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


    def __init__(
        self,
        in_dim=256,
        out_dim=4,
        n_blocks=2,
    ) -> None:
        super().__init__()

        self.gru0 = nn.GRU(input_size=in_dim, hidden_size=in_dim*2)
        self.gru1 = nn.GRU(input_size=in_dim//2, hidden_size=in_dim)
        self.gru2 = nn.GRU(input_size=in_dim//4, hidden_size=in_dim//2)

        self.dense0 = nn.Linear(in_features=in_dim*3, out_features=128)
        self.relu0 = nn.ReLU()

        self.gru_blocks = []
        for i in range(n_blocks):
            self.gru_blocks.append(self.GRU_Block(128))

        self.dense_final = nn.Linear(in_features=128, out_features=out_dim)

        
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

    class ConvBlock(nn.Module):

        def __init__(self, dim, kernel_size) -> None:
            """1d convolution block with residual shortcut connection

            Args:
                dim (int): input and output dimension
                kernel_size (int): convolution kernel size
            """
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(dim, dim),
                nn.ReLU(),
                nn.BatchNorm1d(dim),
                nn.ReflectionPad1d(kernel_size//2),
                nn.Conv1d(dim+kernel_size, dim, kernel_size)
            )
        
        def forward(self, x):
            x0 = self.net(x)
            return torch.add(x0, x)


    def __init__(
        self,
        in_dim=128,
        out_dim=4,
        conv_out_channels=[100, 100, 50],
        kernel_sizes=[128, 64, 16],
        n_blocks=[4],
        block_kernel_size=64,
    ) -> None:
        super().__init__()

        self.rp0 = nn.ReflectionPad1d(kernel_sizes[0]//2)
        self.conv0 = nn.Conv1d(
            in_dim + kernel_sizes[0],
            out_channels=conv_out_channels[0],
            kernel_size=kernel_sizes[0])

        self.rp1 = nn.ReflectionPad1d(kernel_sizes[1]//2)
        self.conv1 = nn.Conv1d(
            in_dim + kernel_sizes[1],
            out_channels=conv_out_channels[1],
            kernel_size=kernel_sizes[1])

        self.rp2 = nn.ReflectionPad1d(kernel_sizes[2]//2)
        self.conv2 = nn.Conv1d(
            in_dim + kernel_sizes[2],
            out_channels=conv_out_channels[2],
            kernel_size=kernel_sizes[2])

        dim = sum(conv_out_channels)

        self.conv_blocks = []
        for i in range(n_blocks):
            self.conv_blocks.append(self.ConvBlock(dim, block_kernel_size))

        self.net_end = nn.Sequential(
            nn.ReLU(),
            nn.BatchNorm1d(dim),
            nn.ReflectionPad1d(block_kernel_size//2),
            nn.Conv1d(dim, out_dim, block_kernel_size)
        )
    
    def forward(self, x):
        x0 = self.rp0(x)
        x0 = self.conv0(x0)

        x1 = self.rp1(x)
        x1 = self.conv1(x1)

        x2 = self.rp2(x)
        x2 = self.conv2(x2)

        x = torch.cat((x0, x1, x2))

        for conv_block in self.conv_blocks:
            x = conv_block(x)
        
        x = self.net_end(x)
        return x

