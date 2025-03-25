import torch
from torch import nn
import torch.nn.init as init
import torch.nn.functional as F

# from prescriptor_2 import CryptoLSTM

class ConvFullyBase(nn.Module):
    def __init__(self, 
                 input_dim: int, 
                 hidden_dim: int, 
                 output_dim: int,
                 group_size: int):
        super(ConvFullyBase, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.group_size = group_size

        self.pos_emb = nn.Embedding(self.group_size*3, 2)
        # Task 1 layers
        self.fc1= nn.Conv1d(in_channels=(self.input_dim+2)*self.group_size,
                            out_channels=self.hidden_dim*group_size,
                            kernel_size=1,
                            stride=1,
                            groups=self.group_size)
        # Final MLP layer
        self.fc_final= nn.Conv1d(in_channels=self.hidden_dim*self.group_size,
                            out_channels=self.output_dim*group_size,
                            kernel_size=1,
                            stride=1,
                            groups=self.group_size)
        # self.relu = nn.ReLU(inplace=True)
        self.tanh = nn.Tanh()

    def forward(self, x: torch.Tensor, x_cate: torch.Tensor):
        x_cate = self.pos_emb(x_cate)
        
        x = torch.concat([x, x_cate], dim=-1)
        if x.ndim == 2:
            x = x.unsqueeze(dim=0)
            
        x_shape = x.shape
        B = x_shape[0]
        x = x.reshape(B, -1, 1)
        # Task 1 forward pass
        x1 = self.fc1(x)
        # x1 = self.relu(x1)
        x1 = self.tanh(x1)

        # Pass through final MLP layer
        out = self.fc_final(x1)
        out = out.reshape(B, x_shape[1], -1)
        out = self.tanh(out)
        
        return out

# class BaseLSTMModel(nn.Module):
#     def __init__(self, small_input_dim, large_input_dim, fc_hidden_size, 
#                  small_lstm_hidden_dim, large_lstm_hidden_dim, output_dim, 
#                  num_layers=2, volume_index=-1, bidirectional=False):
#         """
#         Args:
#             small_input_dim (int): 작은 입력 branch 피처 수 (Volume 포함)
#             large_input_dim (int): 큰 입력 branch 피처 수 (Volume 포함)
#             fc_hidden_size (int): 각 branch의 초기 FC 레이어 출력 차원 (Volume 제외)
#             small_lstm_hidden_dim (int): 작은 branch LSTM의 hidden state 차원
#             large_lstm_hidden_dim (int): 큰 branch LSTM의 hidden state 차원
#             output_dim (int): 최종 출력 차원
#             num_layers (int): LSTM 레이어 수 (두 branch 모두 동일)
#             volume_index (int): 입력 시퀀스 내 Volume 피처의 인덱스 (여기서는 -1)
#             bidirectional (bool): 양방향 LSTM 사용 여부
#         """
#         super(BaseLSTMModel, self).__init__()
#         self.volume_index = volume_index  # 항상 -1
        
#         # 양방향 여부에 따른 hidden dimension 보정
#         self.small_num_directions = 2 if bidirectional else 1
#         self.large_num_directions = 2 if bidirectional else 1
        
#         # 초기 FC 레이어는 volume 피처를 제외한 차원으로 처리함.
#         self.small_feature_dim = small_input_dim - 1  # volume 제외
#         self.large_feature_dim = large_input_dim - 1  # volume 제외
        
#         # 작은 입력 branch 초기 FC (LeakyReLU, inplace=True)
#         self.small_fc = nn.Sequential(
#             nn.Linear(self.small_feature_dim, fc_hidden_size),
#             nn.LeakyReLU(inplace=True)
#         )
        
#         # 큰 입력 branch 초기 FC (LeakyReLU, inplace=True)
#         self.large_fc = nn.Sequential(
#             nn.Linear(self.large_feature_dim, fc_hidden_size),
#             nn.LeakyReLU(inplace=True)
#         )
        
#         # 작은 branch LSTM
#         self.small_lstm = nn.LSTM(
#             input_size=fc_hidden_size,
#             hidden_size=small_lstm_hidden_dim,
#             num_layers=num_layers,
#             batch_first=True,
#             dropout=0,  # dropout 제거
#             bidirectional=bidirectional
#         )
        
#         # 큰 branch LSTM
#         self.large_lstm = nn.LSTM(
#             input_size=fc_hidden_size,
#             hidden_size=large_lstm_hidden_dim,
#             num_layers=num_layers,
#             batch_first=True,
#             dropout=0,  # dropout 제거
#             bidirectional=bidirectional
#         )
        
#         # 마지막 FC 레이어: 두 branch의 마지막 LSTM 출력을 concat하여 최종 예측 산출
#         self.last_fc = nn.Linear(
#             small_lstm_hidden_dim * self.small_num_directions + 
#             large_lstm_hidden_dim * self.large_num_directions, 
#             output_dim
#         )
    
class BaseLSTMModel(nn.Module):
    def __init__(self, small_input_dim, large_input_dim, fc_hidden_size, 
                 small_lstm_hidden_dim, large_lstm_hidden_dim, output_dim, 
                 num_layers=2, volume_index=-1, bidirectional=False):
        """
        Args:
            small_input_dim (int): 작은 입력 branch의 전체 피처 수 (Volume 포함)
            large_input_dim (int): 큰 입력 branch의 전체 피처 수 (Volume 포함)
            fc_hidden_size (int): 각 branch의 초기 FC 레이어 출력 차원 (Volume 제외)
            small_lstm_hidden_dim (int): 작은 branch LSTM의 hidden state 차원
            large_lstm_hidden_dim (int): 큰 branch LSTM의 hidden state 차원
            output_dim (int): 최종 출력 차원
            num_layers (int): LSTM 레이어 수 (두 branch 모두 동일)
            volume_index (int): 입력 시퀀스 내에서 Volume 피처의 인덱스 (여기서는 -1)
            bidirectional (bool): 양방향 LSTM 사용 여부
        """
        super(BaseLSTMModel, self).__init__()
        self.volume_index = volume_index  # 항상 -1

        # 양방향 여부에 따른 hidden dimension 보정
        self.small_num_directions = 2 if bidirectional else 1
        self.large_num_directions = 2 if bidirectional else 1

        # 초기 FC 레이어는 volume 피처를 제외한 차원으로 처리함.
        self.small_feature_dim = small_input_dim - 1  # volume 제외
        self.large_feature_dim = large_input_dim - 1  # volume 제외

        # 작은 입력 branch 초기 FC (LeakyReLU, inplace=True)
        self.small_fc = nn.Sequential(
            nn.Linear(self.small_feature_dim, fc_hidden_size),
            nn.LeakyReLU(inplace=True)
        )
        # 큰 입력 branch 초기 FC (LeakyReLU, inplace=True)
        self.large_fc = nn.Sequential(
            nn.Linear(self.large_feature_dim, fc_hidden_size),
            nn.LeakyReLU(inplace=True)
        )

        # volume 처리 FC layers: volume 피처는 스칼라이므로 입력 차원은 1이며,
        # 두 FC를 통과하여 fc_hidden_size 차원의 벡터로 변환
        self.small_volume_fc = nn.Sequential(
            nn.Linear(1, fc_hidden_size),
            nn.LeakyReLU(inplace=True),
            nn.Linear(fc_hidden_size, fc_hidden_size)
        )
        self.large_volume_fc = nn.Sequential(
            nn.Linear(1, fc_hidden_size),
            nn.LeakyReLU(inplace=True),
            nn.Linear(fc_hidden_size, fc_hidden_size)
        )

        # 작은 branch LSTM
        self.small_lstm = nn.LSTM(
            input_size=fc_hidden_size,
            hidden_size=small_lstm_hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0,  # dropout 제거
            bidirectional=bidirectional
        )
        # 큰 branch LSTM
        self.large_lstm = nn.LSTM(
            input_size=fc_hidden_size,
            hidden_size=large_lstm_hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=0,  # dropout 제거
            bidirectional=bidirectional
        )
        # 마지막 FC 레이어: 두 branch의 마지막 LSTM 출력을 concat하여 최종 예측 산출
        self.last_fc = nn.Linear(
            small_lstm_hidden_dim * self.small_num_directions + 
            large_lstm_hidden_dim * self.large_num_directions, 
            output_dim
        )

    def forward(self, small_x, large_x):
        """
        Args:
            small_x (Tensor): 작은 입력 branch, shape (batch_size, seq_len, small_input_dim)
            large_x (Tensor): 큰 입력 branch, shape (batch_size, seq_len, large_input_dim)
        Returns:
            out (Tensor): 최종 예측, shape (batch_size, 1, output_dim)
        """
        # ===== 작은 branch 처리 =====
        # volume 피처 분리 (마지막 피처) 및 나머지 피처 분리
        small_features = small_x[:, :, :-1]    # (batch, seq_len, small_input_dim - 1)
        small_volume = small_x[:, :, -1].unsqueeze(-1)  # (batch, seq_len, 1)

        # 피처 FC 처리
        small_fc_out = self.small_fc(small_features)  # (batch, seq_len, fc_hidden_size)
        # Volume FC 처리
        small_volume_weight = self.small_volume_fc(small_volume)  # (batch, seq_len, fc_hidden_size)
        # 두 출력을 element-wise 곱하여 volume weighting 적용
        small_weighted = small_fc_out * small_volume_weight

        small_lstm_out, _ = self.small_lstm(small_weighted)
        small_last = small_lstm_out[:, -1, :]  # 마지막 시점의 출력

        # ===== 큰 branch 처리 =====
        large_features = large_x[:, :, :-1]    # (batch, seq_len, large_input_dim - 1)
        large_volume = large_x[:, :, -1].unsqueeze(-1)  # (batch, seq_len, 1)

        large_fc_out = self.large_fc(large_features)  # (batch, seq_len, fc_hidden_size)
        large_volume_weight = self.large_volume_fc(large_volume)  # (batch, seq_len, fc_hidden_size)
        large_weighted = large_fc_out * large_volume_weight

        large_lstm_out, _ = self.large_lstm(large_weighted)
        large_last = large_lstm_out[:, -1, :]

        # 두 branch의 출력을 concat
        concat_out = torch.cat([small_last, large_last], dim=-1)
        # 마지막 FC 레이어를 통해 최종 예측
        out = self.last_fc(concat_out)
        return out.unsqueeze(dim=1)

    def forward(self, small_x, large_x):
        """
        Args:
            small_x (Tensor): 작은 입력 branch, shape (batch_size, seq_len, small_input_dim)
            large_x (Tensor): 큰 입력 branch, shape (batch_size, seq_len, large_input_dim)
        Returns:
            out (Tensor): 최종 예측, shape (batch_size, 1, output_dim)
        """
        # 작은 branch 처리
        # volume feature 분리: 마지막 feature만 별도 사용, fc는 나머지 feature들 사용
        small_features = small_x[:, :, :-1]  # (batch, seq_len, small_input_dim - 1)
        small_volume = small_x[:, :, -1]       # (batch, seq_len)
        
        small_fc_out = self.small_fc(small_features)  # (batch, seq_len, fc_hidden_size)
        # volume 기반 가중치: softmax로 정규화
        small_volume_weights = F.softmax(small_volume, dim=1).unsqueeze(-1)  # (batch, seq_len, 1)
        small_weighted = small_fc_out * small_volume_weights
        
        small_lstm_out, _ = self.small_lstm(small_weighted)
        small_last = small_lstm_out[:, -1, :]  # 마지막 시점의 출력
        
        # 큰 branch 처리
        large_features = large_x[:, :, :-1]  # (batch, seq_len, large_input_dim - 1)
        large_volume = large_x[:, :, -1]       # (batch, seq_len)
        
        large_fc_out = self.large_fc(large_features)  # (batch, seq_len, fc_hidden_size)
        large_volume_weights = F.softmax(large_volume, dim=1).unsqueeze(-1)  # (batch, seq_len, 1)
        large_weighted = large_fc_out * large_volume_weights
        
        large_lstm_out, _ = self.large_lstm(large_weighted)
        large_last = large_lstm_out[:, -1, :]
        
        # 두 branch의 출력을 concat
        concat_out = torch.cat([small_last, large_last], dim=-1)
        
        # 마지막 FC 레이어를 통해 최종 예측
        out = self.last_fc(concat_out)
        return out.unsqueeze(dim=1)
    
class Prescriptor(nn.Module):
    def __init__(self, 
                 basic_block: nn.Module, 
                 small_input_dim: int, 
                 large_input_dim: int,
                 fc_hidden_size: int,
                 small_lstm_hidden_dim: int, 
                 large_lstm_hidden_dim: int,
                 output_dim: int,
                 after_input_dim: int,
                 after_hidden_dim: int,
                 after_output_dim: int, 
                 num_blocks: int = 1):
        super(Prescriptor, self).__init__()
        
        if basic_block == None:
            # self.base_network = BaseLSTMModel
            self.base_network = CryptoLSTM
        else:
            self.base_network = basic_block
            
        # self.layers = nn.ModuleList([self.base_network(small_input_dim, large_input_dim, fc_hidden_size, small_lstm_hidden_dim, large_lstm_hidden_dim, output_dim, ) for _ in range(num_blocks)])
        self.layers = nn.ModuleList([self.base_network(small_input_dim, large_input_dim, fc_hidden_size, output_dim=output_dim) for _ in range(num_blocks)])
        self.after_layers = ConvFullyBase(after_input_dim, after_hidden_dim, after_output_dim, group_size=num_blocks)
        self.num_blcoks = num_blocks
        
        
    def forward(self, small_x, large_x):
        outputs = [layer(small_x, large_x) for layer in self.layers]
        outputs = torch.concat(outputs)
        return outputs
    
    def base_forward(self, small_x, large_x):
        outputs = [layer(small_x, large_x) for layer in self.layers]
        outputs = torch.stack(outputs, dim=0).half().cpu()
        return outputs
    
    # def base_forward(self, small_x, large_x):
    #     # 각 layer의 forward를 torch.jit.fork로 병렬 실행
    #     futures = [torch.jit.fork(layer, small_x, large_x) for layer in self.layers]
    #     outputs = [torch.jit.wait(f) for f in futures]
    #     # 결과를 stacking 후 half 타입으로 변환하고 CPU로 이동
    #     outputs = torch.stack(outputs, dim=0).half().cpu()
    #     return outputs

    def after_forward(self, x, x_cate):
        return self.after_layers(x, x_cate)
    

class ChromosomeSelectorModel(nn.Module):
    def __init__(self, large_input_dim, hidden_dim, num_chromosomes, num_layers=2):
        super(ChromosomeSelectorModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Define the LSTM layers
        self.large_lstm = nn.LSTM(large_input_dim, hidden_dim, num_layers, batch_first=True)
        
        # Define the output layer
        self.fc = nn.Linear(hidden_dim, num_chromosomes)
    
    def forward(self, large_x):
        large_out, _ = self.large_lstm(large_x)
        large_out = large_out[:, -1, :]
        
        # Output scores for each chromosome
        scores = self.fc(large_out)
        return scores