import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange, Reduce

class SeqCLRRecurrentEncoder(torch.nn.Module):
    """
    see Figure 4.A of proceedings.mlr.press/v136/mohsenvand20a/mohsenvand20a.pdf for diagram
    """
    def __init__(self, num_channels, temporal_len, dropout_rate=0.5, embed_dim=100):
        super(SeqCLRRecurrentEncoder, self).__init__()
        self.TEMPORAL_DIM = 2
        self.BATCH_DIM = 0
        self.CHANNEL_DIM = 1

        self.gru_1 = torch.nn.GRU(num_channels, 256)
        self.gru_2 = torch.nn.GRU(num_channels, 128)
        self.gru_3 = torch.nn.GRU(num_channels, 64)

        self.x_linear_1 = torch.nn.Linear(256+128+64, 128)
        self.x_relu_1 = torch.nn.ReLU()
        self.h_linear_1 = torch.nn.Linear(256+128+64, 128)
        self.h_relu_1 = torch.nn.ReLU()

        self.x_layer_norm_1 = torch.nn.LayerNorm(128)
        self.h_layer_norm_1 = torch.nn.LayerNorm(128)

        self.gru_4 = torch.nn.GRU(128, 128)
        
        self.x_layer_norm_2 = torch.nn.LayerNorm(128)
        self.h_layer_norm_2 = torch.nn.LayerNorm(128)
        
        self.gru_5 = torch.nn.GRU(128, 128)

        self.final_linear = torch.nn.Linear(128, embed_dim)

        self.num_channels = num_channels
        self.temporal_len = temporal_len
        self.dropout_rate = dropout_rate
        self.embed_dim = embed_dim
    
    def forward(self, x):
        orig_temporal_len = x.size(self.TEMPORAL_DIM)
        orig_batch_num = x.size(self.BATCH_DIM)
        x = x.permute(self.BATCH_DIM, self.CHANNEL_DIM, self.TEMPORAL_DIM)
        
        # prepare x for processing
        # see https://discuss.pytorch.org/t/how-to-downsample-a-time-series/78485
        # and https://discuss.pytorch.org/t/f-interpolate-weird-behaviour/36088
        x_down_1 = torch.nn.functional.interpolate(x, size=(x.size(self.TEMPORAL_DIM)//2))
        intermed_temporal_len = x_down_1.size(self.TEMPORAL_DIM)
        x_down_2 = torch.nn.functional.interpolate(x_down_1, size=(x_down_1.size(self.TEMPORAL_DIM)//2))
        
        x = x.permute(self.TEMPORAL_DIM, self.BATCH_DIM, self.CHANNEL_DIM)
        x_down_1 = x_down_1.permute(self.TEMPORAL_DIM, self.BATCH_DIM, self.CHANNEL_DIM)
        x_down_2 = x_down_2.permute(self.TEMPORAL_DIM, self.BATCH_DIM, self.CHANNEL_DIM)

        # embed x using various gru modules
        x_embed_1, hidden_1 = self.gru_1(x) # TO-DO: do we need to remember this hidden, or not?
        x_embed_2, hidden_2 = self.gru_2(x_down_1)# TO-DO: do we need to remember this hidden, or not?
        x_embed_3, hidden_3 = self.gru_3(x_down_2)# TO-DO: do we need to remember this hidden, or not?

        # upsample the two smaller embeddings, with x_embed_3 requiring two upsamples to reach the appropriate size ***Note: https://pytorch.org/docs/stable/notes/randomness.html
        x_embed_2 = torch.nn.functional.interpolate(x_embed_2.permute(1,2,0), size=(orig_temporal_len)).permute(2, 0, 1)
        # hidden_2 = torch.nn.functional.interpolate(hidden_2.permute(1,2,0), size=(orig_temporal_len)).permute(2, 0, 1)# TO-DO: do we need to remember this hidden, or not?

        x_embed_3 = torch.nn.functional.interpolate(x_embed_3.permute(1,2,0), size=(intermed_temporal_len)).permute(2, 0, 1)
        x_embed_3 = torch.nn.functional.interpolate(x_embed_3.permute(1,2,0), size=(orig_temporal_len)).permute(2, 0, 1)
        # hidden_3 = torch.nn.functional.interpolate(hidden_3.permute(1,2,0), size=(intermed_temporal_len)).permute(2, 0, 1)# TO-DO: do we need to remember this hidden, or not?
        # hidden_3 = torch.nn.functional.interpolate(hidden_3.permute(1,2,0), size=(orig_temporal_len)).permute(2, 0, 1)# TO-DO: do we need to remember this hidden, or not?

        # combine embeddings
        x = torch.cat(tuple([x_embed_1, x_embed_2, x_embed_3]), dim=2)
        h = torch.cat(tuple([hidden_1, hidden_2, hidden_3]), dim=2)# TO-DO: do we need to remember this hidden, or not?

        x = self.x_linear_1(x)
        h = self.h_linear_1(h)
        x = self.x_relu_1(x)
        h = self.h_relu_1(h)

        # first residual block pass
        x_hat = self.x_layer_norm_1(x)
        h_hat = self.h_layer_norm_1(h)# TO-DO: do we need to remember this hidden, or not?
        x_hat, h_hat = self.gru_4(x, h)# TO-DO: do we need to remember this hidden, or not?
        x = x + x_hat
        h = h + h_hat# TO-DO: do we need to remember this hidden, or not?

        # second residual block pass
        x_hat = self.x_layer_norm_2(x)
        # h_hat = self.h_layer_norm_2(h)# TO-DO: do we need to remember this hidden, or not?
        x_hat, _ = self.gru_5(x, h) # x_hat, h_hat = self.gru_5(x, h)# TO-DO: do we need to remember this hidden, or not?
        x = x + x_hat
        # h = h + h_hat
        
        # final output generation
        x = self.final_linear(x) # TO-DO: should I include the final h in this as well (via concatenation)??? if so, need to change final_linear and preceding code in forward pass
        x = x.permute(1, 2, 0)
        return x, torch.mean(x, dim=2) # x is used for upstream task and torch.mean(x, dim=1) for downstream

class SeqCLRConvolutionalResidualBlock(torch.nn.Module):
    """
    see Figure 4.B of proceedings.mlr.press/v136/mohsenvand20a/mohsenvand20a.pdf for diagram
    """
    def __init__(self, D=250, K=64, dropout_rate=0.5):
        super(SeqCLRConvolutionalResidualBlock, self).__init__()
        self.BATCH_DIM = 0
        self.CHANNEL_DIM = 1
        self.TEMPORAL_DIM = 2

        self.linear = torch.nn.Linear(D, D)
        self.sequential = torch.nn.Sequential(
            torch.nn.ReLU(), 
            torch.nn.BatchNorm1d(D), 
            torch.nn.ReflectionPad1d(((K//2)-1, K//2)), 
            torch.nn.Conv1d(D, D, K)
        )
        pass
    
    def forward(self, x):
        x_hat = self.linear(x.permute(self.BATCH_DIM, 2, 1)).permute(self.BATCH_DIM, 2, 1)
        x_hat = self.sequential(x_hat)
        return x + x_hat

class SeqCLRConvolutionalEncoder(torch.nn.Module):
    """
    see Figure 4.B of proceedings.mlr.press/v136/mohsenvand20a/mohsenvand20a.pdf for diagram
    """
    def __init__(self, num_channels, temporal_len, dropout_rate=0.5, embed_dim=100):
        super(SeqCLRConvolutionalEncoder, self).__init__()
        self.BATCH_DIM = 0
        self.CHANNEL_DIM = 2
        self.TEMPORAL_DIM = 1

        self.K128 = 64
        self.K64 = 32
        self.K16 = 8

        self.D_INTERNAL_100 = 20
        self.D_INTERNAL_50 = 10
        self.D_INTERNAL_250 = 50
        self.D_OUT = embed_dim

        self.conv_block_1 = torch.nn.Sequential(
            torch.nn.ReflectionPad1d(((self.K128//2)-1, self.K128//2)), 
            torch.nn.Conv1d(num_channels, self.D_INTERNAL_100, self.K128)
        )
        self.conv_block_2 = torch.nn.Sequential(
            torch.nn.ReflectionPad1d(((self.K64//2)-1, self.K64//2)), 
            torch.nn.Conv1d(num_channels, self.D_INTERNAL_100, self.K64)
        )
        self.conv_block_3 = torch.nn.Sequential(
            torch.nn.ReflectionPad1d(((self.K16//2)-1, self.K16//2)), 
            torch.nn.Conv1d(num_channels, self.D_INTERNAL_50, self.K16)
        )

        self.res_block_1 = SeqCLRConvolutionalResidualBlock(self.D_INTERNAL_250, self.K64)
        self.res_block_2 = SeqCLRConvolutionalResidualBlock(self.D_INTERNAL_250, self.K64)
        self.res_block_3 = SeqCLRConvolutionalResidualBlock(self.D_INTERNAL_250, self.K64)
        self.res_block_4 = SeqCLRConvolutionalResidualBlock(self.D_INTERNAL_250, self.K64)

        self.final_relu = torch.nn.ReLU()
        self.final_batch_norm = torch.nn.BatchNorm1d(self.D_INTERNAL_250)
        self.final_reflective_padding = torch.nn.ReflectionPad1d(((self.K64//2)-1, self.K64//2))
        self.final_conv_layer = torch.nn.Conv1d(self.D_INTERNAL_250, self.D_OUT, self.K64)

        # self.final_linear = torch.nn.Linear(self.D_OUT*temporal_len, self.D_OUT) # added this layer (not in paper) for comparisson purposes in generalizing to other tasks

        self.num_channels = num_channels
        self.temporal_len = temporal_len
        self.dropout_rate = dropout_rate
        self.embed_dim = embed_dim
        pass
    
    def forward(self, x):
        orig_batch_num = x.size(0)
        x = x.permute(self.BATCH_DIM, self.CHANNEL_DIM, self.TEMPORAL_DIM)

        # embed x using various convolutional modules
        x_embed_1 = self.conv_block_1(x)
        x_embed_2 = self.conv_block_2(x)
        x_embed_3 = self.conv_block_3(x)

        # combine embeddings
        x = torch.cat(tuple([x_embed_1, x_embed_2, x_embed_3]), dim=1)

        # first residual block pass
        x_hat = self.res_block_1(x)
        x = x + x_hat

        # second residual block pass
        x_hat = self.res_block_2(x)
        x = x + x_hat
        
        # third residual block pass
        x_hat = self.res_block_3(x)
        x = x + x_hat
        
        # fourth residual block pass
        x_hat = self.res_block_4(x)
        x = x + x_hat
        
        # final output generation
        x = self.final_relu(x)
        x = self.final_batch_norm(x)
        x = self.final_reflective_padding(x)
        x = self.final_conv_layer(x)

        # x = self.final_linear(x.view(orig_batch_num, -1))

        return x, torch.mean(x, dim=2) # x is used for upstream task and torch.mean(x, dim=1) for downstream

class SeqCLRDecoder(torch.nn.Module):
    """
    see Figure 4.C of proceedings.mlr.press/v136/mohsenvand20a/mohsenvand20a.pdf for diagram
    """
    def __init__(self, num_channels, temporal_len, dropout_rate=0.5, num_upstream_decode_features=32):
        super(SeqCLRDecoder, self).__init__()
        self.TEMPORAL_DIM = 2
        self.BATCH_DIM = 0
        self.CHANNEL_DIM = 1

        self.out_features_256 = 32 # 256
        self.out_features_128 = 16 # 128
        self.out_features_64 = 8 # 64
        self.bdlstm_1 = torch.nn.LSTM(num_channels, self.out_features_256, bidirectional=True)
        self.bdlstm_2 = torch.nn.LSTM(num_channels, self.out_features_128, bidirectional=True)
        self.bdlstm_3 = torch.nn.LSTM(num_channels, self.out_features_64, bidirectional=True)

        self.x_linear_1 = torch.nn.Linear(2*2*self.out_features_256+2*2*self.out_features_128+2*2*self.out_features_64, self.out_features_128)
        self.x_relu_1 = torch.nn.ReLU()
        # self.h_linear_1 = torch.nn.Linear(2*2*256+2*2*128+2*2*64, 128)
        # self.h_relu_1 = torch.nn.ReLU()
        # self.c_linear_1 = torch.nn.Linear(2*2*256+2*2*128+2*2*64, 128)
        # self.c_relu_1 = torch.nn.ReLU()

        self.final_linear = torch.nn.Linear(self.out_features_128, num_upstream_decode_features)

        self.num_channels = num_channels
        self.temporal_len = temporal_len
        self.dropout_rate = dropout_rate
        self.num_upstream_decode_features = num_upstream_decode_features
    
    def forward(self, x):
        # print("SeqCLRDecoder.forward: x shape == ", x.shape)
        # orig_temporal_len = x.size(self.TEMPORAL_DIM)
        # orig_batch_num = x.size(self.BATCH_DIM)
        
        # prepare x for processing
        # see https://discuss.pytorch.org/t/how-to-downsample-a-time-series/78485
        # and https://discuss.pytorch.org/t/f-interpolate-weird-behaviour/36088
        x_down_1 = torch.nn.functional.interpolate(x, size=(x.size(self.TEMPORAL_DIM)//2))
        # print("SeqCLRDecoder.forward: x_down_1 shape == ", x_down_1.shape)
        # intermed_temporal_len = x_down_1.size(self.TEMPORAL_DIM)
        x_down_2 = torch.nn.functional.interpolate(x_down_1, size=(x_down_1.size(self.TEMPORAL_DIM)//2))
        # print("SeqCLRDecoder.forward: x_down_2 shape == ", x_down_2.shape)
        
        x = x.permute(self.TEMPORAL_DIM, self.BATCH_DIM, self.CHANNEL_DIM)
        # print("SeqCLRDecoder.forward: x shape == ", x.shape)
        x_down_1 = x_down_1.permute(self.TEMPORAL_DIM, self.BATCH_DIM, self.CHANNEL_DIM)
        # print("SeqCLRDecoder.forward: x_down_1 shape == ", x_down_1.shape)
        x_down_2 = x_down_2.permute(self.TEMPORAL_DIM, self.BATCH_DIM, self.CHANNEL_DIM)
        # print("SeqCLRDecoder.forward: x_down_2 shape == ", x_down_2.shape)

        # embed x using various gru modules
        x_embed_1, (h_1, c_1) = self.bdlstm_1(x) # TO-DO: do we need to remember this hidden, or not?
        x_embed_2, (h_2, c_2) = self.bdlstm_2(x_down_1)# TO-DO: do we need to remember this hidden, or not?
        x_embed_3, (h_3, c_3) = self.bdlstm_3(x_down_2)# TO-DO: do we need to remember this hidden, or not?

        x_embed_1 = x_embed_1.permute(1, 0, 2)
        x_embed_2 = x_embed_2.permute(1, 0, 2)
        x_embed_3 = x_embed_3.permute(1, 0, 2)

        # combine embeddings
        x = torch.cat(tuple([x_embed_1[:,0,:], 
                             x_embed_1[:,-1,:], 
                             x_embed_2[:,0,:], 
                             x_embed_2[:,-1,:], 
                             x_embed_3[:,0,:], 
                             x_embed_3[:,-1,:]]), 
                      dim=1
        )
        # print("x shape == ", x.shape)
        # h = torch.cat(tuple([h_1, h_2, h_3]), dim=2)# TO-DO: do we need to remember this hidden, or not?
        # c = torch.cat(tuple([c_1, c_2, c_3]), dim=2)# TO-DO: do we need to remember this hidden, or not?

        x = self.x_linear_1(x)
        # h = self.h_linear_1(h)
        # c = self.c_linear_1(c)
        x = self.x_relu_1(x)
        # h = self.h_relu_1(h)
        # c = self.c_relu_1(c)
        
        # final output generation
        out = self.final_linear(x) # TO-DO: should I include the final h & c in this as well (via concatenation)??? if so, need to change final_linear and preceding code in forward pass
        # raise NotImplementedError()
        return out


# Simplified encoder
class PhaseSwapFCN(torch.nn.Module):
    """
    See Section 3 of arxiv.org/pdf/2009.07664.pdf for description and
    see Figure 1.b in arxiv.org/pdf/1611.06455.pdf for most of diagram
    """
    def __init__(self, num_channels, dropout_rate=0.5, embed_dim=100):
        super(PhaseSwapFCN, self).__init__()
        self.BATCH_DIM = 0
        self.CHANNEL_DIM = 2
        self.TEMPORAL_DIM = 1

        self.K8 = 8
        self.K5 = 5
        self.K3 = 3

        self.D_INTERNAL_128 = 128
        self.D_INTERNAL_256 = 256
        self.D_OUT = embed_dim

        # self.conv_block_1 = torch.nn.Sequential(
        #     torch.nn.ReflectionPad1d(((self.K128//2)-1, self.K128//2)), 
        #     torch.nn.Conv1d(num_channels, self.D_INTERNAL_100, self.K128)
        # )
        # self.conv_block_2 = torch.nn.Sequential(
        #     torch.nn.ReflectionPad1d(((self.K64//2)-1, self.K64//2)), 
        #     torch.nn.Conv1d(num_channels, self.D_INTERNAL_100, self.K64)
        # )
        # self.conv_block_3 = torch.nn.Sequential(
        #     torch.nn.ReflectionPad1d(((self.K16//2)-1, self.K16//2)), 
        #     torch.nn.Conv1d(num_channels, self.D_INTERNAL_50, self.K16)
        # )

        self.conv_block_1 = torch.nn.Sequential(
            torch.nn.ReflectionPad1d(((self.K8//2)-1, self.K8//2)), 
            torch.nn.Conv1d(num_channels, self.D_INTERNAL_128, self.K8), 
            torch.nn.BatchNorm1d(self.D_INTERNAL_128, track_running_stats=False), 
            torch.nn.ReLU()
        )
        self.conv_block_2 = torch.nn.Sequential(
            torch.nn.ReflectionPad1d(((self.K5//2), self.K5//2)), 
            torch.nn.Conv1d(self.D_INTERNAL_128, self.D_INTERNAL_256, self.K5), 
            torch.nn.BatchNorm1d(self.D_INTERNAL_256, track_running_stats=False), 
            torch.nn.ReLU()
        )
        self.conv_block_3 = torch.nn.Sequential(
            torch.nn.ReflectionPad1d(((self.K3//2), self.K3//2)), 
            torch.nn.Conv1d(self.D_INTERNAL_256, self.D_OUT, self.K3), # note: orig architecture uses self.D_INTERNAL_128 instead of self.D_OUT - made substitution to compare with other baselines using same decoder (01/18/2021)
            torch.nn.BatchNorm1d(self.D_OUT, track_running_stats=False), # note: orig architecture uses self.D_INTERNAL_128 instead of self.D_OUT - made substitution to compare with other baselines using same decoder (01/18/2021)
            torch.nn.ReLU()
        )

        self.avg_pool = torch.nn.AvgPool1d(self.D_OUT)#, padding=self.D_INTERNAL_128//2) # note: orig architecture uses self.D_INTERNAL_128 instead of self.D_OUT - made substitution to compare with other baselines using same decoder (01/18/2021)
        pass
    
    def forward(self, x):
        # print("\nPhaseSwapFCN: x.shape == ", x.shape)
        x = x.permute(self.BATCH_DIM, self.CHANNEL_DIM, self.TEMPORAL_DIM)
        # print("PhaseSwapFCN: x.shape == ", x.shape)
        x = self.conv_block_1(x)
        # print("PhaseSwapFCN: x.shape == ", x.shape)
        x = self.conv_block_2(x)
        # print("PhaseSwapFCN: x.shape == ", x.shape)
        x = self.conv_block_3(x)
        # print("PhaseSwapFCN: x_resid.shape == ", x.shape)
        x = self.avg_pool(x)
        # print("PhaseSwapFCN: out.shape == ", x.shape)
        return x, torch.mean(x, dim=2)#self.TEMPORAL_DIM) # x is used for upstream task and torch.mean(x, dim=2) for downstream

class PhaseSwapUpstreamDecoder(torch.nn.Module):
    """
    See Section 3 of arxiv.org/pdf/2009.07664.pdf for description
    """
    def __init__(self, hidden_dim, temporal_len, dropout_rate=0.5, decode_dim=1):
        super(PhaseSwapUpstreamDecoder, self).__init__()
        self.BATCH_DIM = 0
        self.CHANNEL_DIM = 1
        self.TEMPORAL_DIM = 2
        
        self.linear = torch.nn.Linear(hidden_dim*(temporal_len//hidden_dim), decode_dim)
        pass
    
    def forward(self, x):
        # print("\nPhaseSwapUpstreamDecoder: x.shape == ", x.shape)
        x = x.view(x.size(0), -1)
        # print("PhaseSwapUpstreamDecoder: x.shape == ", x.shape)
        x = self.linear(x)
        # print("PhaseSwapUpstreamDecoder: x.shape == ", x.shape)
        # x = torch.nn.functional.softmax(x)
        # print("PhaseSwapUpstreamDecoder: x.shape == ", x.shape)
        return x

class SQNet(torch.nn.Module):
    """
    see proceedings.mlr.press/v136/mohsenvand20a/mohsenvand20a.pdf
    """
    def __init__(self, encoder_type, num_channels, temporal_len, dropout_rate=0.5, embed_dim=100, num_upstream_decode_features=32):
        super(SQNet, self).__init__()

        if encoder_type == "recurrent":
            self.embed_model = SeqCLRRecurrentEncoder(num_channels, temporal_len, dropout_rate=dropout_rate, embed_dim=embed_dim)
            self.decode_model = SeqCLRDecoder(embed_dim, 
                                     temporal_len, 
                                     dropout_rate=dropout_rate, 
                                     num_upstream_decode_features=num_upstream_decode_features
            )
        elif encoder_type == "convolutional":
            self.embed_model = SeqCLRConvolutionalEncoder(num_channels, temporal_len, dropout_rate=dropout_rate, embed_dim=embed_dim)
            self.decode_model = SeqCLRDecoder(embed_dim, 
                                     temporal_len, 
                                     dropout_rate=dropout_rate, 
                                     num_upstream_decode_features=num_upstream_decode_features
            )
        elif encoder_type == "simplified":
            self.embed_model = PhaseSwapFCN(num_channels, dropout_rate=dropout_rate, embed_dim=embed_dim)
            self.decode_model = PhaseSwapUpstreamDecoder(embed_dim, temporal_len, dropout_rate=dropout_rate, decode_dim=num_upstream_decode_features)
        else:
            raise ValueError("encoder_type "+str(encoder_type)+" not supported")

        self.encoder_type = encoder_type
        pass
    
    def forward(self, x): 
        x, _ = self.embed_model(x)
        x = self.decode_model(x)
        return x




class Projector(nn.Module):
    def __init__(self, params):
        for key in params:
            self.key = params[key]

    def forward(self, x_in):
        pass

class Transformer(nn.Module):
    def __init__(self,params):
        super().__init__()
        for key in params:
            self.key = params[key]

    def forward(self):
        pass

class Convolution(nn.Module):
    def __init__(self, params):
        # self.patch_size = patch_size
        super().__init__()
        for key in params:
            self.key = params[key]

        # self.kernel_size_1 = (1,25) and self.kernel_size_2 = (22,1), self.drop_r = 0.5, self.pool_kernel, self.pool_stirde = (1, 75), (1, 15)
        self.shallownet = nn.Sequential(
            nn.Conv2d(self.in_channels, self.hidden_size, self.kernel_size_1, (1, 1)),
            nn.Conv2d(self.hidden_size, self.hidden_size, self.kernel_size_2, (1, 1)),
            nn.BatchNorm2d(self.hidden_size),
            nn.ELU(),
            nn.AvgPool2d(self.pool_kernel, self.pool_stride),
            # pooling acts as slicing to obtain 'patch' along the time dimension as in ViT
            nn.Dropout(self.drop_r),
        )

        self.projection = nn.Sequential(
            nn.Conv2d(40, self.emb_size, (1, 1), stride=(1, 1)),  # transpose, conv could enhance fiting ability slightly
            Rearrange('b e (h) (w) -> b (h w) e'),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, _, _, _ = x.shape
        x = self.shallownet(x)
        x = self.projection(x)
        return x

class Encoder(nn.Sequential):
    def __init__(self, conv_params, trans_params):
        self.conv = Convolution(conv_params)
        self.trans = Transformer(trans_params)

    def forward(self, x_in):
        x = self.conv(x_in)
        x_out = self.trans(x)
        return  x_out

class TransPreTrain(nn.Module):
    def __init__(self, conv_params,trans_params, project_params):
        self.encoder = Encoder(conv_params, trans_params)
        self.projector = Projector(project_params)

    def forward(self, x_in):
        x = self.encoder(x_in)
        x_out = self.projector(x)
        return x_out
