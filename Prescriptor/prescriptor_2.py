import torch
import torch.nn as nn

# class TimeBiLSTM(nn.Module):
#     def __init__(self, hidden_dim, num_layers=2):
#         super().__init__()
#         self.bilstm = nn.LSTM(
#             hidden_dim, 
#             hidden_dim, 
#             num_layers=num_layers, 
#             batch_first=True, 
#             bidirectional=True
#         )
#         # Mapping from 2*hidden_dim back to hidden_dim
#         self.fc = nn.Linear(2 * hidden_dim, hidden_dim)
#         self.norm = nn.LayerNorm(hidden_dim)
        
#     def forward(self, x):
#         # x shape: (batch, seq_len, hidden_dim)
#         lstm_out, _ = self.bilstm(x)  # lstm_out: (batch, seq_len, 2*hidden_dim)
#         lstm_out = self.fc(lstm_out)  # (batch, seq_len, hidden_dim)
#         # Residual connection and normalization
#         return self.norm(lstm_out)

# class TemporalConvBlock(nn.Module):
#     def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
#         super().__init__()
#         self.conv = nn.Conv1d(
#             in_channels,
#             out_channels,
#             kernel_size,
#             stride=stride,
#             padding=0 if stride > 1 else 'same',
#             padding_mode='replicate'
#         )
#         self.activation = nn.GELU()
        
#     def forward(self, x):
#         # x shape: (batch, channels, seq_len)
#         return self.activation(self.conv(x))

# class CryptoModel(nn.Module):
#     def __init__(
#         self,
#         input_dim_1m,
#         input_dim_1d,
#         hidden_dim=16,
#         output_dim=8
#     ):
#         super().__init__()
        
#         # 1. Temporal Convolution for 1-minute data (reducing sequence length by 10x)
#         self.tcn_1m = nn.Sequential(
#             TemporalConvBlock(input_dim_1m, hidden_dim//2, kernel_size=10, stride=10),
#             TemporalConvBlock(hidden_dim // 2, hidden_dim)
#         )
        
#         # 2. Temporal Convolution for 1-day data
#         self.tcn_1d = nn.Sequential(
#             TemporalConvBlock(input_dim_1d, hidden_dim//2, kernel_size=1),
#             TemporalConvBlock(hidden_dim // 2, hidden_dim)
#         )
        
#         # 3. Replace time-based attention with BiLSTM modules
#         self.time_bilstm_1m = TimeBiLSTM(hidden_dim)
#         self.time_bilstm_1d = TimeBiLSTM(hidden_dim)
        
#         # 4. Timeframe interaction module
#         self.timeframe_interaction = nn.Sequential(
#             nn.Linear(hidden_dim * 2, hidden_dim),
#             nn.GELU(),
#         )
        
#         # 5. Output layers
#         self.out_layers = nn.Sequential(
#             nn.Linear(hidden_dim, output_dim)
#         )
        
#     def forward(self, x_1m, x_1d):
#         # 1. Initial temporal convolution
#         x_1m = self.tcn_1m(x_1m.transpose(1, 2)).transpose(1, 2)
#         x_1d = self.tcn_1d(x_1d.transpose(1, 2)).transpose(1, 2)

#         # print(x_1m.shape)

#         # 2. Apply BiLSTM modules
#         x_1m = self.time_bilstm_1m(x_1m)
#         x_1d = self.time_bilstm_1d(x_1d)
        
#         # 3. Extract last timestep features
#         x_1m = x_1m[:, -1, :]  # (batch, hidden_dim)
#         x_1d = x_1d[:, -1, :]  # (batch, hidden_dim)
        
#         # 4. Combine timeframe information
#         combined = torch.cat([x_1m, x_1d], dim=-1)  # (batch, hidden_dim * 2)
#         timeframe_features = self.timeframe_interaction(combined)
        
#         # 5. Final prediction
#         return self.out_layers(timeframe_features)
    


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

# 하나의 Temporal Block (두 개의 dilated convolution과 residual connection 포함)
class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size,
                               stride=1, padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()

        # self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1,
        #                          self.conv2, self.chomp2, self.relu2)

        self.net = nn.Sequential(self.conv1, self.relu1,
                                 self.conv2, self.relu2)

    def forward(self, x):
        out = self.net(x)
        return out

# TCN 모델: 여러 TemporalBlock을 층으로 쌓음. 각 층마다 stride를 지정할 수 있음.
class TCN(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2, strides=None):
        """
        num_inputs: 입력 채널 수
        num_channels: 각 층의 출력 채널 수 리스트 (예: [25, 25, 50])
        kernel_size: 컨볼루션 필터 크기
        dropout: 드롭아웃 비율
        strides: 각 층에 적용할 stride 리스트. 지정하지 않으면 모두 1로 사용.
        """
        super(TCN, self).__init__()
        layers = []
        num_levels = len(num_channels)
        if strides is None:
            strides = [1] * num_levels
        for i in range(num_levels):
            dilation_size = 2 ** i  # dilated rate를 2의 거듭제곱으로 증가
            stride = strides[i]
            padding = (kernel_size - 1) * dilation_size
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers.append(
                TemporalBlock(in_channels, out_channels, kernel_size,
                              stride=stride, dilation=dilation_size,
                              padding=padding, dropout=dropout)
            )
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

class CryptoModelTCN(nn.Module):
    def __init__(self, input_dim_1m, input_dim_1d, hidden_dim=16, output_dim=8):
        super().__init__()
        # TCN for 1-minute data (입력 길이 600)
        # 3개의 레이어를 사용하여 stride [5, 4, 5]로 진행:
        # 600 -> 120 -> 30
        # kernel_size는 5로 확장하여 receptive field를 넓힘
        self.tcn_1m = TCN(input_dim_1m, [hidden_dim] * 3, kernel_size=5, dropout=0.2, strides=[5, 4, 5])
        
        # TCN for 1-day data (입력 길이 60; kernel_size는 기존대로 3)
        # 2개의 레이어를 사용하여 stride [2, 2]: 60 -> 30 -> 15
        self.tcn_1d = TCN(input_dim_1d, [hidden_dim] * 2, kernel_size=3, dropout=0.2, strides=[2, 2])
        
        # LSTM을 사용하여 시간 차원을 축소 (batch_first=True: 입력 shape -> (batch, seq_len, feature))
        self.lstm_1m = nn.LSTM(input_size=hidden_dim, hidden_size=hidden_dim, batch_first=True)
        self.lstm_1d = nn.LSTM(input_size=hidden_dim, hidden_size=hidden_dim, batch_first=True)
        
        # 두 타임프레임 특징 결합 및 상호작용 모듈
        self.timeframe_interaction = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
        )
        
        # 최종 예측 레이어
        self.out_layers = nn.Sequential(
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, x_1m, x_1d):
        # x_1m: (batch, seq_len=1440, channels=input_dim_1m)
        # x_1d: (batch, seq_len=60, channels=input_dim_1d)
        # Conv1d는 (batch, channels, seq_len) 형태를 요구하므로 transpose 수행
        x_1m = x_1m.transpose(1, 2)  # -> (batch, input_dim_1m, 1440)
        x_1d = x_1d.transpose(1, 2)  # -> (batch, input_dim_1d, 60)
        
        # 각 타임프레임별 TCN 적용 (stride로 입력 길이 감소)
        x_1m = self.tcn_1m(x_1m)   # -> (batch, hidden_dim, 30)
        x_1d = self.tcn_1d(x_1d)   # -> (batch, hidden_dim, 약 15)
        
        # LSTM에 넣기 위해 time dimension을 앞쪽으로 이동: (batch, seq_len, feature)
        x_1m = x_1m.transpose(1, 2)  # -> (batch, 30, hidden_dim)
        x_1d = x_1d.transpose(1, 2)  # -> (batch, 약 15, hidden_dim)

        # LSTM을 통한 시퀀스 정보 요약 (마지막 타임스텝의 은닉 상태 사용)
        out_1m, (h_n_1m, _) = self.lstm_1m(x_1m)  # h_n_1m: (1, batch, hidden_dim)
        out_1d, (h_n_1d, _) = self.lstm_1d(x_1d)  # h_n_1d: (1, batch, hidden_dim)
        
        # h_n의 첫 번째 차원 제거
        feat_1m = h_n_1m.squeeze(0)  # (batch, hidden_dim)
        feat_1d = h_n_1d.squeeze(0)  # (batch, hidden_dim)
        
        # 타임프레임별 특징 결합 및 상호작용
        combined = torch.cat([feat_1m, feat_1d], dim=-1)  # (batch, hidden_dim*2)
        features = self.timeframe_interaction(combined)
        
        # 최종 예측 출력
        return self.out_layers(features)