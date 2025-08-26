import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import numpy as np

from indicators import prepare_features  # Import for features prep

# LSTM Model
class TradingLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=64):
        super(TradingLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        _, (h_n, _) = self.lstm(x)
        out = self.sigmoid(self.fc(h_n[-1]))
        return out

# Dataset for sequences
class SeqDataset(Dataset):
    def __init__(self, features, labels, seq_len):
        self.features = torch.tensor(features, dtype=torch.float32).unfold(0, seq_len, 1).permute(0, 2, 1)
        self.labels = torch.tensor(labels[seq_len-1:], dtype=torch.float32)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

def train_model(df, epochs=50, batch_size=32, seq_len=20):
    features, labels = prepare_features(df, seq_len)
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, shuffle=False)
    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    train_dataset = SeqDataset(X_train, y_train, seq_len)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    model = TradingLSTM(input_size=5)  # Features count
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(epochs):
        model.train()
        for batch_features, batch_labels in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_features).squeeze()
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()

    # Test
    model.eval()
    test_dataset = SeqDataset(X_test, y_test, seq_len)
    test_loader = DataLoader(test_dataset, batch_size=1)
    preds = []
    with torch.no_grad():
        for feat, _ in test_loader:
            preds.append(model(feat).item() > 0.5)
    acc = accuracy_score(y_test[seq_len-1:], preds)
    print(f'Accuracy: {acc:.2f}')

    torch.save(model.state_dict(), 'lstm_model.pt')
    np.save('scaler.npy', scaler)
    return model, scaler

def load_model(input_size):
    model = TradingLSTM(input_size)
    model.load_state_dict(torch.load('lstm_model.pt'))
    model.eval()
    return model

def get_ml_prediction(model, scaler, recent_df, seq_len=20):
    features, _ = prepare_features(recent_df, seq_len)
    if len(features) < seq_len:
        return 0.5
    scaled = scaler.transform(features[-seq_len:].reshape(1, seq_len, -1))
    with torch.no_grad():
        pred = model(torch.tensor(scaled, dtype=torch.float32)).item()
    return pred
