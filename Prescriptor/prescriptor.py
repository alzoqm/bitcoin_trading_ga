import torch
from torch import nn
import torch.nn.init as init

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
        
        return out


class BaseLSTMModel(nn.Module):
    def __init__(self, small_input_dim, large_input_dim, hidden_dim, output_dim, num_layers=2):
        super(BaseLSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Define the LSTM layer
        self.small_lstm = nn.LSTM(small_input_dim, hidden_dim, num_layers, batch_first=True)
        self.large_lstm = nn.LSTM(large_input_dim, hidden_dim, num_layers, batch_first=True)
        # self.lstm = LSTM(input_dim, hidden_dim, num_layers)
        
        # Define the output layer
        self.fc = nn.Linear(hidden_dim*2, output_dim)
    
    def forward(self, small_x, large_x):
        # h0 = torch.zeros(self.num_layers, small_x.size(0), self.hidden_dim).to(small_x.device)
        # c0 = torch.zeros(self.num_layers, small_x.size(0), self.hidden_dim).to(small_x.device)
        # small_out, _ = self.small_lstm(small_x, (h0, c0))
        small_out, _ = self.small_lstm(small_x)
        small_out = small_out[:, -1, :]

        # h0 = torch.zeros(self.num_layers, large_x.size(0), self.hidden_dim).to(large_x.device)
        # c0 = torch.zeros(self.num_layers, large_x.size(0), self.hidden_dim).to(large_x.device)
        # large_out, _ = self.large_lstm(large_x, (h0, c0))
        large_out, _ = self.large_lstm(large_x)
        large_out = large_out[:, -1, :]
        
        out = torch.concat([small_out, large_out], dim=-1)
        
        # Decode the hidden state of the last time step
        out = self.fc(out)
        return out.unsqueeze(dim=1)

class Prescriptor(nn.Module):
    def __init__(self, 
                 basic_block: nn.Module, 
                 base_small_input_dim: int, 
                 base_large_input_dim: int,
                 base_hidden_dim: int, 
                 base_output_dim: int,
                 after_input_dim: int,
                 after_hidden_dim: int,
                 after_output_dim: int, 
                 num_blocks: int = 1):
        super(Prescriptor, self).__init__()
        
        if basic_block == None:
            self.base_network = BaseLSTMModel
        else:
            self.base_network = basic_block
            
        self.layers = nn.ModuleList([self.base_network(base_small_input_dim, base_large_input_dim, base_hidden_dim, base_output_dim) for _ in range(num_blocks)])
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