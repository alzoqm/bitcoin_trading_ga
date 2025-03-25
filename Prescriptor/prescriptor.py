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

class BaseConvFully(nn.Module):
    def __init__(self, 
                 input_dim: int, 
                 hidden_dim: int, 
                 output_dim: int,
                 group_size: int):
        super(BaseConvFully, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.group_size = group_size
 
        # Task 1 layers
        self.fc1 = nn.Sequential(
            nn.Conv1d(
                in_channels=self.input_dim* self.group_size,
                out_channels=self.hidden_dim * 4 * self.group_size,
                kernel_size=1,
                stride=1,
                groups=self.group_size
            ),
            nn.GELU(),
            nn.Conv1d(
                in_channels=self.hidden_dim * 4 * self.group_size,
                out_channels=self.hidden_dim * 2 * self.group_size,
                kernel_size=1,
                stride=1,
                groups=self.group_size
            ),
            nn.GELU(),
            nn.Conv1d(
                in_channels=self.hidden_dim * 2 * self.group_size,
                out_channels=self.hidden_dim * self.group_size,
                kernel_size=1,
                stride=1,
                groups=self.group_size
            ),
            nn.GELU(),
        )

        # Final MLP layer
        self.fc_final= nn.Conv1d(in_channels=self.hidden_dim*self.group_size,
                            out_channels=self.output_dim*group_size,
                            kernel_size=1,
                            stride=1,
                            groups=self.group_size)
        
    def forward(self, x: torch.Tensor):
        # x shape: [B, input_dim] (예: [100, 52])
        B = x.shape[0]
        
        # 1. 마지막 차원을 추가하여 [B, input_dim, 1]로 만듭니다.
        x = x.unsqueeze(-1)  # shape: [100, 52, 1]
        
        # 2. group_size 만큼 채널을 복제하여 [B, input_dim * group_size, 1]로 만듭니다.
        #    여기서 self.group_size = 30 이므로 최종 채널 수는 52*30 = 1560이 됩니다.
        x = x.repeat(1, self.group_size, 1)  # shape: [100, 1560, 1]
        
        # 3. Task 1: 연속된 Conv1d와 GELU 활성화 레이어를 통과합니다.
        x1 = self.fc1(x)
        
        # 4. 최종 MLP layer를 통과합니다.
        out = self.fc_final(x1)
        
        # 5. 최종 출력을 reshape하여 원하는 형태로 만듭니다.
        #    여기서는 [B, input_dim, -1] 형태로 reshape 하는 예시입니다.
        out = out.reshape(B, -1, self.output_dim)  # 예: [100, 52, 8]
        return out


    
class Prescriptor(nn.Module):
    def __init__(self, 
                 input_dim: int, 
                 fc_hidden_size: int,
                 output_dim: int,
                 after_input_dim: int,
                 after_hidden_dim: int,
                 after_output_dim: int, 
                 num_blocks: int = 1):
        super(Prescriptor, self).__init__()

        self.base_layers = BaseConvFully(input_dim, fc_hidden_size, output_dim, num_blocks)
        self.after_layers = ConvFullyBase(after_input_dim, after_hidden_dim, after_output_dim, group_size=num_blocks)
        self.num_blcoks = num_blocks
        
        
    def forward(self, small_x):
        outputs = self.base_layers(small_x)
        return outputs
    
    def base_forward(self, small_x):
        outputs = self.base_layers(small_x)
        return outputs
 

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