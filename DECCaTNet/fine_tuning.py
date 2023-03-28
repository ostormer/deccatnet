import torch
import torch.nn as nn

from .DECCaTNet_model import Encoder


class EncodingClassifier(nn.Module):
    def __init__(self):
        pass

    def forward(self, X):
        return X


class DownstreamNet(nn.Module):
    def __init__(self, params):
        """
        encoder_path
        channel_group_size = 2
        n_channel_groups
        encoder_out_size
        n_classes = 2
        """
        # self.patch_size = patch_size
        super().__init__()
        for key in params:
            self.key = params[key]

        self.encoder = Encoder()
        self.encoder.load_state_dict(self.encoder_path)
        self.encoder.requires_grad_(False)

        # trans_layer = nn.TransformerEncoderLayer(d_model=1024, nhead=8)
        # self.transformer = nn.TransformerEncoder(encoder_layer=trans_layer, num_layers=6)

        self.classifier = nn.Sequential(
            nn.Linear(in_features=self.encoder_out_size * self.n_channel_groups, out_features=256),
            nn.ReLU(),
            nn.Linear(in_features=256, out_features=64),
            nn.ReLU(),
            nn.Linear(in_features=64, out_features=self.n_classes),
            nn.LogSoftmax(self.n_classes)
        )

    def forward(self, X):
        print(X.shape)
        # Split input into chunks that fit into the encoder
        # TODO: Decide what to do if n_channels does not fit into encoder size (Not even)
        # Do we discard the last channels? Do we make a overlap
        encoder_inputs = torch.split(X, self.channel_group_size, dim=0)
        encoder_out = []

        # Run each group/pair of channels through the encoder
        for group in encoder_inputs:
            print(group.shape)
            encoder_out.append(self.encoder(group))  # Encode group

        combined_encodings = torch.flatten(torch.concat(encoder_out, dim=0))
        x = self.classifier_net(combined_encodings)
        return x


def run_downstream_task(dataset, batch_size, train_split, pretrained_encoder_path, temperature, learning_rate,
                        weight_decay, num_workers, max_epochs, batch_print_freq, save_dir_model, model_file_name):
