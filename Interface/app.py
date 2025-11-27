from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import threading
import requests
import pandas as pd
import numpy as np
from io import StringIO
from datetime import datetime, timedelta
import torch
import torch.nn as nn
from transformers import GPT2Model, GPT2Config
from threading import Lock
import os
from torch.serialization import add_safe_globals

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# Global variables
latest_forecast = None
progress_data = {"status": "idle", "progress": 0, "message": ""}
progress_lock = Lock()

# Model parameters (MUST match training)
D = 768
T_in = 60
T_out = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Model Architecture (EXACTLY as in training) ---
class TokenEncoding(nn.Module):
    def __init__(self, input_dim, embedding_dim, kernel_size=3):
        super().__init__()
        self.conv = nn.Conv1d(input_dim, embedding_dim, kernel_size=kernel_size, padding=1)
    def forward(self, x):
        return self.conv(x.transpose(1, 2)).transpose(1, 2)

class PositionalEncoding(nn.Module):
    def __init__(self, num_patches, embedding_dim):
        super().__init__()
        self.embedding = nn.Embedding(num_patches, embedding_dim)
    def forward(self, x):
        positions = torch.arange(0, x.size(1), device=x.device).expand(x.size(0), -1)
        return self.embedding(positions)

class TemporalEncoding(nn.Module):
    def __init__(self, embedding_dim):
        super().__init__()
        self.temporal_embedding = nn.Linear(1, embedding_dim)
    def forward(self, t_info):
        return self.temporal_embedding(t_info.unsqueeze(-1))

class PatchReconstruction(nn.Module):
    def __init__(self, embedding_dim, patch_length):
        super().__init__()
        self.linear = nn.Linear(embedding_dim, patch_length)
    def forward(self, z):
        return self.linear(z)

class ForecastingModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_encoder = TokenEncoding(1, D)
        self.pos_encoder = PositionalEncoding(T_in, D)
        self.temp_encoder = TemporalEncoding(D)
        self.backbone = GPT2Model(GPT2Config(n_embd=D, n_layer=6, n_head=8))
        self.reconstructor = PatchReconstruction(D, T_out)
    def forward(self, x, temporal_info):
        x_token = self.token_encoder(x)
        x_pos = self.pos_encoder(x_token)
        x_temp = self.temp_encoder(temporal_info)
        x_combined = x_token + x_pos + x_temp
        z = self.backbone(inputs_embeds=x_combined).last_hidden_state[:, -1]
        return self.reconstructor(z)

# --- Load Model & Normalization Stats ---
def load_model_and_stats():
    import numpy as np
    from torch.serialization import add_safe_globals
    
    try:
        # First attempt: Secure loading
        add_safe_globals([
            np.dtype,
            np._core.multiarray.scalar,
            np.dtypes.Float32DType,
            np.float32
        ])
        checkpoint = torch.load("final_model_full.pth", map_location=device, weights_only=True)
    except:
        # Fallback: Only use if you trust the file
        print("WARNING: Using weights_only=False. Only do this if you trust the checkpoint file!")
        checkpoint = torch.load("final_model_full.pth", map_location=device, weights_only=False)
    
    # Initialize model
    model = ForecastingModel().to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Handle numpy types
    data_mean = checkpoint['data_mean'].item() if hasattr(checkpoint['data_mean'], 'item') else checkpoint['data_mean']
    data_std = checkpoint['data_std'].item() if hasattr(checkpoint['data_std'], 'item') else checkpoint['data_std']
    
    return model, data_mean, data_std


model, data_mean, data_std = load_model_and_stats()

@app.route('/')
def index():
    return render_template('index.html')

def predict_weather(lat, lon, param="T2M"):
    with progress_lock:
        progress_data.update({"status": "fetching", "progress": 0, "message": "Fetching 5 years of data..."})

    try:
        # --- Fetch Data ---
        current_date = datetime.now()
        start_date = current_date - timedelta(days=5*365)
        url = (
            f"https://power.larc.nasa.gov/api/temporal/daily/point?"
            f"parameters={param}&community=AG&longitude={lon}&latitude={lat}"
            f"&start={start_date.strftime('%Y%m%d')}&end={current_date.strftime('%Y%m%d')}&format=CSV"
        )
        response = requests.get(url)
        response.raise_for_status()

        # --- Process Data ---
        data_df = pd.read_csv(StringIO(response.text), skiprows=9)
        data_df.columns = [col.strip() for col in data_df.columns]
        data_df['Date'] = pd.to_datetime(
            data_df['YEAR'].astype(str) + data_df['DOY'].astype(str).str.zfill(3), 
            format='%Y%j'
        )
        data_df.set_index('Date', inplace=True)
        series = data_df[param].values.astype(np.float32)

        # --- Normalize with Training Stats ---
        normalized_data = (series - data_mean) / data_std

        # --- Validate Input ---
        if len(normalized_data) < T_in:
            raise ValueError(f"Need at least {T_in} data points, got {len(normalized_data)}")

        # --- Predict ---
        last_window = normalized_data[-T_in:]
        with torch.no_grad():
            last_window_tensor = torch.tensor(last_window, dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(device)
            temporal_info = torch.arange(T_in, device=device).float().unsqueeze(0)
            pred_norm = model(last_window_tensor, temporal_info).cpu().numpy().flatten()
        pred = pred_norm * data_std + data_mean + 3.0

        # --- Format Results ---
        last_date = data_df.index[-1]
        forecast_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=T_out)
        forecast = [{
            "date": d.strftime("%Y-%m-%d"),
            "value": float(v)
        } for d, v in zip(forecast_dates, pred)]

        with progress_lock:
            progress_data.update({"status": "completed", "progress": 100, "message": "Prediction completed."})

        return forecast

    except Exception as e:
        with progress_lock:
            progress_data.update({"status": "error", "message": str(e)})
        raise

@app.route('/forecast', methods=['POST'])
def forecast():
    global latest_forecast
    data = request.json
    latest_forecast = None
    with progress_lock:
        progress_data.update({"status": "processing", "progress": 0, "message": "Starting..."})
    
    def background_task():
        global latest_forecast
        try:
            latest_forecast = predict_weather(
                lat=data['lat'],
                lon=data['lon'],
                param=data.get('property', 'T2M')
            )
        except Exception as e:
            latest_forecast = {"error": str(e)}
    
    threading.Thread(target=background_task).start()
    return jsonify({"status": "processing"})

@app.route('/get-forecast')
def get_forecast():
    if latest_forecast is None:
        return jsonify({"status": "processing"}), 202
    return jsonify(latest_forecast) if isinstance(latest_forecast, list) else jsonify(latest_forecast), 500

@app.route('/get-progress')
def get_progress():
    with progress_lock:
        return jsonify(progress_data)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, threaded=True)
