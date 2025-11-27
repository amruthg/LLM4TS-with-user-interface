# app.py
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

app = Flask(__name__)
CORS(app)

# Global variable for storing forecasts
latest_forecast = None

# Model parameters
D = 768  # Embedding dimension
T_in = 60  # Look-back window
T_out = 10  # Prediction horizon
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- Model Architecture ---
class TokenEncoding(nn.Module):
    def __init__(self, input_dim, embedding_dim, kernel_size=3):
        super().__init__()
        self.conv = nn.Conv1d(input_dim, embedding_dim, kernel_size=kernel_size, padding=1)
    def forward(self, x):
        return self.conv(x.transpose(1, 2)).transpose(1, 2)

class ForecastingModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_encoder = TokenEncoding(1, D)
        self.pos_encoder = nn.Embedding(T_in, D)
        self.temp_encoder = nn.Linear(1, D)
        self.gpt = GPT2Model(GPT2Config(n_embd=D, n_layer=6, n_head=8))
        self.reconstructor = nn.Linear(D, T_out)

    def forward(self, x, temporal_info):
        x_token = self.token_encoder(x)
        x_pos = self.pos_encoder(torch.arange(T_in, device=x.device).expand(x.size(0), -1))
        x_temp = self.temp_encoder(temporal_info.unsqueeze(-1))
        x_combined = x_token + x_pos + x_temp
        z = self.gpt(inputs_embeds=x_combined).last_hidden_state[:, -1]
        return self.reconstructor(z)

def run_weather_model(lat, lon, param="T2M"):
    global latest_forecast
    try:
        # --- Data Fetching ---
        current_date = datetime.now()
        start_date = current_date - timedelta(days=5*365)
        
        url = (
            f"https://power.larc.nasa.gov/api/temporal/daily/point?"
            f"parameters={param}&community=AG&longitude={lon}&latitude={lat}"
            f"&start={start_date.strftime('%Y%m%d')}&end={current_date.strftime('%Y%m%d')}&format=CSV"
        )
        
        response = requests.get(url)
        response.raise_for_status()
        
        # --- Data Processing ---
        data_df = pd.read_csv(StringIO(response.text), skiprows=9)
        data_df.columns = [col.strip() for col in data_df.columns]
        data_df['Date'] = pd.to_datetime(
            data_df['YEAR'].astype(str) + data_df['DOY'].astype(str).str.zfill(3), 
            format='%Y%j'
        )
        data_df.set_index('Date', inplace=True)
        series = data_df[param].values.astype(np.float32)
        
        # --- Normalization ---
        data_mean = series.mean()
        data_std = series.std()
        normalized_data = (series - data_mean) / data_std

        # --- Create Windows ---
        def create_windows(data, look_back, horizon):
            X, y = [], []
            for i in range(len(data) - look_back - horizon):
                X.append(data[i:i+look_back])
                y.append(data[i+look_back:i+look_back+horizon])
            return np.array(X), np.array(y)
        
        X, y = create_windows(normalized_data, T_in, T_out)
        X_tensor = torch.tensor(X, dtype=torch.float32).unsqueeze(-1).to(device)

        # --- Model Setup ---
        model = ForecastingModel().to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=21e-5)
        loss_fn = nn.MSELoss()
        
        # --- Training ---
        model.train()
        for _ in range(25):  # epochs
            optimizer.zero_grad()
            outputs = model(X_tensor, torch.arange(T_in, device=device).float())
            loss = loss_fn(outputs, torch.tensor(y, dtype=torch.float32).to(device))
            loss.backward()
            optimizer.step()

        # --- Prediction ---
        last_window = normalized_data[-T_in:]
        with torch.no_grad():
            model.eval()
            prediction = model(
                torch.tensor(last_window, dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(device),
                torch.arange(T_in, device=device).float()
            ).cpu().numpy().flatten()
        
        # --- Post-processing ---
        dates = [datetime.now() + timedelta(days=i+1) for i in range(T_out)]
        latest_forecast = [{
            "date": d.strftime("%Y-%m-%d"),
            "value": float(v * data_std + data_mean)
        } for d, v in zip(dates, prediction)]
        
    except Exception as e:
        latest_forecast = {"error": str(e)}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/forecast', methods=['POST'])
def forecast():
    global latest_forecast
    data = request.json
    latest_forecast = None  # Reset previous forecast
    
    def background_task():
        run_weather_model(
            lat=data['lat'],
            lon=data['lon'],
            param=data.get('property', 'T2M')
        )
    
    threading.Thread(target=background_task).start()
    return jsonify({"status": "processing"})

@app.route('/get-forecast')
def get_forecast():
    if latest_forecast is None:
        return jsonify({"status": "processing"}), 202
    if isinstance(latest_forecast, list):
        return jsonify(latest_forecast)
    return jsonify(latest_forecast), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, threaded=True)
