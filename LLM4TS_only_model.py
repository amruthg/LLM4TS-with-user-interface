import requests
import pandas as pd
import numpy as np
from io import StringIO
from datetime import datetime, timedelta
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from transformers import GPT2Model, GPT2Config

# Parameters
patch_length = 10
D = 768  # Embedding dimension
batch_size = 64
T_in = 60  # Look-back window
T_out = 10  # Prediction horizon
epochs = 25
learning_rate = 21e-5

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# --- DATA FETCHING AND PREPARATION ---

# Fetch last 5 years of T2M data for Bangalore Rural
district = {"lat": 13.18, "lon": 77.8}
current_date = datetime.now()
start_date = current_date - timedelta(days=(5*365)+4)
start_date_str = start_date.strftime('%Y%m%d')
end_date_str = current_date.strftime('%Y%m%d')
param = "T2M"

url_template = (
    "https://power.larc.nasa.gov/api/temporal/daily/point?"
    "parameters={param}&community=AG&longitude={lon}&latitude={lat}"
    "&start={start}&end={end}&format=CSV"
)
url = url_template.format(
    param=param, lon=district["lon"], lat=district["lat"],
    start=start_date_str, end=end_date_str
)
response = requests.get(url)
if response.status_code != 200:
    raise Exception(f"Failed to fetch data: {response.status_code}")

# Parse T2M data
data_df = pd.read_csv(StringIO(response.text), skiprows=9)
data_df.columns = [col.strip() for col in data_df.columns]
data_df['Date'] = pd.to_datetime(
    data_df['YEAR'].astype(str) + data_df['DOY'].astype(str).str.zfill(3),
    format='%Y%j'
)
data_df.set_index('Date', inplace=True)
t2m_series = data_df["T2M"].values.astype(np.float32)

# --- NORMALIZATION ---
data_mean = t2m_series.mean()
data_std = t2m_series.std()
data = (t2m_series - data_mean) / data_std

# --- SLIDING WINDOW CREATION ---
def create_sliding_windows(data, look_back, forecast_horizon):
    X, y = [], []
    for i in range(len(data) - look_back - forecast_horizon):
        X.append(data[i:i + look_back])
        y.append(data[i + look_back:i + look_back + forecast_horizon])
    X = np.array(X)
    y = np.array(y)
    return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)

X, y = create_sliding_windows(data, T_in, T_out)

# --- TRAIN/TEST SPLIT ---
split_idx = int(0.8 * len(X))
X_train, y_train = X[:split_idx], y[:split_idx]
X_test, y_test = X[split_idx:], y[split_idx:]

# --- DATA LOADER ---
train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# --- MODEL DEFINITION ---
class TokenEncoding(nn.Module):
    def __init__(self, input_dim, embedding_dim, kernel_size=3):
        super(TokenEncoding, self).__init__()
        self.conv = nn.Conv1d(input_dim, embedding_dim, kernel_size=kernel_size, padding=1)
    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.conv(x)
        return x.transpose(1, 2)

class PositionalEncoding(nn.Module):
    def __init__(self, num_patches, embedding_dim):
        super(PositionalEncoding, self).__init__()
        self.embedding = nn.Embedding(num_patches, embedding_dim)
    def forward(self, x):
        batch_size, num_patches, _ = x.size()
        positions = torch.arange(0, num_patches, device=x.device).unsqueeze(0).expand(batch_size, num_patches)
        return self.embedding(positions)

class TemporalEncoding(nn.Module):
    def __init__(self, embedding_dim):
        super(TemporalEncoding, self).__init__()
        self.temporal_embedding = nn.Linear(1, embedding_dim)
    def forward(self, t_info):
        return self.temporal_embedding(t_info.unsqueeze(-1))

class PatchReconstruction(nn.Module):
    def __init__(self, embedding_dim, patch_length):
        super(PatchReconstruction, self).__init__()
        self.linear = nn.Linear(embedding_dim, patch_length)
    def forward(self, z):
        return self.linear(z)

class ForecastingModel(nn.Module):
    def __init__(self, backbone, token_encoder, pos_encoder, temp_encoder, patch_reconstructor):
        super(ForecastingModel, self).__init__()
        self.backbone = backbone
        self.token_encoder = token_encoder
        self.pos_encoder = pos_encoder
        self.temp_encoder = temp_encoder
        self.reconstructor = patch_reconstructor
    def forward(self, x, temporal_info):
        x_token = self.token_encoder(x)
        x_pos = self.pos_encoder(x_token)
        x_temp = self.temp_encoder(temporal_info)
        x_combined = x_token + x_pos + x_temp
        z = self.backbone(inputs_embeds=x_combined).last_hidden_state
        z_last = z[:, -1, :]
        reconstructed = self.reconstructor(z_last.unsqueeze(1))
        return reconstructed.squeeze(1)

# Instantiate Model Components
token_encoder = TokenEncoding(input_dim=1, embedding_dim=D).to(device)
pos_encoder = PositionalEncoding(num_patches=T_in, embedding_dim=D).to(device)
temp_encoder = TemporalEncoding(embedding_dim=D).to(device)
patch_reconstructor = PatchReconstruction(embedding_dim=D, patch_length=T_out).to(device)
config = GPT2Config(n_embd=D, n_layer=6, n_head=8)
gpt2_model = GPT2Model(config).to(device)
model = ForecastingModel(gpt2_model, token_encoder, pos_encoder, temp_encoder, patch_reconstructor).to(device)

# --- TRAINING SETUP ---
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
loss_function = torch.nn.MSELoss()

# --- TRAINING LOOP ---
for epoch in range(epochs):
    model.train()
    total_loss = 0.0
    for batch_data, batch_gt in train_loader:
        batch_data, batch_gt = batch_data.to(device), batch_gt.to(device)
        temporal_info = (
            torch.arange(batch_data.size(1), device=device).float()
            .unsqueeze(0)
            .expand(batch_data.size(0), -1)
        )
        optimizer.zero_grad()
        output = model(batch_data.unsqueeze(-1), temporal_info)
        loss = loss_function(output, batch_gt)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch + 1}/{epochs}, Training Loss: {total_loss / len(train_loader)}")

# --- TEST LOSS ---
model.eval()
with torch.no_grad():
    X_test, y_test = X_test.to(device), y_test.to(device)
    temporal_info_test = torch.arange(X_test.size(1), device=device).float().unsqueeze(0).expand(X_test.size(0), -1)
    predictions = model(X_test.unsqueeze(-1), temporal_info_test)
    test_loss = loss_function(predictions, y_test).item()
    print(f"Test Loss: {test_loss}")

# --- 10-DAY FORECAST FOR NEXT DAYS ---
last_window = data[-T_in:]
last_window_tensor = torch.tensor(last_window, dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(device)
temporal_info = torch.arange(T_in, device=device).float().unsqueeze(0)
model.eval()
with torch.no_grad():
    pred_norm = model(last_window_tensor, temporal_info).cpu().numpy().flatten()
pred = pred_norm * data_std + data_mean
# Print forecast with corresponding dates
last_date = data_df.index[-1]
forecast_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=T_out)
forecast_df = pd.DataFrame({'Date': forecast_dates, 'Forecast_T2M': pred})
print("\nNext 10 days forecast:")
print(forecast_df)
