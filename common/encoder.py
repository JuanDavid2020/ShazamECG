import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock1D(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, kernel_size=5):
        super().__init__()
        padding = kernel_size // 2
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=kernel_size, padding=padding, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)

# =====================================================================
# 1. BEAT ENCODER (12 Leads x 200 muestras -> 128D)
# Función: Extraer características morfológicas del latido (QRS)
# Patologías Target: LBBB, RBBB, PVC, etc.
# =====================================================================
class BeatEncoder(nn.Module):
    def __init__(self, embedding_dim=128):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv1d(12, 32, kernel_size=7, padding=3), nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(2))
        self.layer1 = ResidualBlock1D(32, 64, stride=2, kernel_size=5)
        self.layer2 = ResidualBlock1D(64, 128, stride=2, kernel_size=5)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(nn.Linear(128, embedding_dim), nn.BatchNorm1d(embedding_dim))

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        return F.normalize(self.fc(self.pool(x).squeeze(-1)), dim=1)


# =====================================================================
# 2. RHYTHM ENCODER (12 Leads x 2000 muestras -> 128D)
# Función: Extraer características temporales a largo plazo (8 segundos)
# Patologías Target: AFIB, Flutter, Bradicardia, Taquicardia, etc.
# =====================================================================
class RhythmEncoder(nn.Module):
    def __init__(self, embedding_dim=128):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv1d(12, 32, kernel_size=15, padding=7), nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(4))
        self.layer1 = ResidualBlock1D(32, 64, stride=2, kernel_size=11)
        self.layer2 = ResidualBlock1D(64, 128, stride=2, kernel_size=7)
        self.layer3 = ResidualBlock1D(128, 256, stride=2, kernel_size=5)
        self.layer4 = ResidualBlock1D(256, 256, stride=2, kernel_size=5)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(nn.Linear(256, embedding_dim), nn.BatchNorm1d(embedding_dim))

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return F.normalize(self.fc(self.pool(x).squeeze(-1)), dim=1)
