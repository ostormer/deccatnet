import torch.nn as nn
import torch

class RecurrentEncoder(nn.Module):

    class GRU_Block(nn.Module):
        """Simple GRU block with layernorm and residual connection
        """
        def __init__(self, dim) -> None:
            super().__init__()
            self.ln = nn.LayerNorm(dim)
            self.gru = nn.GRU(dim, dim * 2)
        
        def forward(self, x):
            x0 = self.ln(x)
            x0 = self.gru(x0)
            return torch.add(x, x0)


    def __init__(self, in_size=256, n_blocks=2) -> None:
        super().__init__()

        self.gru0 = nn.GRU(input_size=in_size, hidden_size=in_size*2)
        self.gru1 = nn.GRU(input_size=in_size//2, hidden_size=in_size)
        self.gru2 = nn.GRU(input_size=in_size//4, hidden_size=in_size//2)

        self.dense0 = nn.Linear(in_features=in_size*3, out_features=128)
        self.relu0 = nn.ReLU()

        self.gru_blocks = []
        for i in range(n_blocks):
            self.gru_blocks.append(self.GRU_Block(128))

        self.dense_final = nn.Linear(in_features=128, out_features=4)

        
    def forward(self, x):
        x0 = self.gru0(x)

        x1 = nn.functional.interpolate(x, scale_factor=0.5, mode='nearest')
        x1 = self.gru1(x1)
        x1 = nn.functional.interpolate(x1, scale_factor=2)

        x2 = nn.functional.interpolate(x, scale_factor=0.25, mode='nearest')
        x2 = self.gru2(x2)
        x2 = nn.functional.interpolate(x2, scale_factor=4)

        x = torch.cat((x0, x1, x2))

        x = self.dense0(x)
        x = self.relu0(x)

        for gru in self.gru_blocks:
            x = gru(x)
        
        x = self.dense_final(x)
        return x


class ConvolutionalEncoder(nn.Module):

    class CONV_Block(nn.Module):

        def __init__(self) -> None:
            super().__init__()