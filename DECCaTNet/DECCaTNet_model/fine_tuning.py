import torch
import torch.nn as nn
from tqdm import tqdm

from DECCaTNet.preprocessing.preprocess import _make_adjacent_pairs
from .DECCaTNet_model import Encoder


class EncodingClassifier(nn.Module):
    def __init__(self):
        pass

    def forward(self, X):
        return X


class FineTuneNet(nn.Module):
    def __init__(self, channel_groups, params):
        """
        encoder_path
        channel_group_size = 2
        embedding_size
        n_classes = 2
        """
        # self.patch_size = patch_size
        super().__init__()
        self.encoder_path = params["encoder_path"]
        self.channel_group_size = params["channel_group_size"]
        self.channel_groups = channel_groups  # Channel groups defined by names
        self.n_channel_groups = len(self.channel_groups)  # Number of channel groups
        self.embedding_size = params["embedding_size"]
        self.n_classes = params["n_classes"]

        self.encoder = Encoder(emb_size=self.embedding_size)
        self.encoder.load_state_dict(torch.load(self.encoder_path))
        self.encoder.requires_grad_(False)

        self.ordered_channels = []  # TODO: Ask Styrk about channel ordering in preprocessed files. Is it arbitrary?
        # TODO: Make dict witch translates channel names to index in preprocessed files
        # TODO: Use that dict to define channel groups by indexes instead of channel names


        # trans_layer = nn.TransformerEncoderLayer(d_model=1024, nhead=8)
        # self.transformer = nn.TransformerEncoder(encoder_layer=trans_layer, num_layers=6)

        self.classifier = nn.Sequential(
            nn.Linear(in_features=self.embedding_size * self.n_channel_groups, out_features=256),
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
        # TODO: Define splits from channel group index list from init to reduce run time
        encoder_inputs = torch.split(X, self.channel_group_size, dim=0)
        if X.shape[2] % self.n_channel_groups != 0:
            encoder_inputs.drop
        encoder_out = []

        # Run each group/pair of channels through the encoder
        for group in encoder_inputs:
            print(group.shape)
            encoder_out.append(self.encoder(group))  # Encode group

        combined_encodings = torch.flatten(torch.concat(encoder_out, dim=0))
        x = self.classifier_net(combined_encodings)
        return x


def run_fine_tuning(dataset, params):
    epochs = params["epochs"]
    dataset_channels = sorted(
        ['EEG F4', 'EEG P4', 'EEG T6', 'EEG P3', 'EEG C4', 'EEG C3', 'EEG T3', 'EEG T5', 'EEG O1', 'EEG FP1', 'EEG A2',
         'EEG T1', 'EEG T2', 'EEG F7', 'EEG FZ', 'EEG O2', 'EEG A1', 'EEG CZ', 'EEG F8', 'EEG T4', 'EEG PZ', 'EEG F3',
         'EEG FP2'])
    channel_groups = _make_adjacent_pairs(sorted(dataset_channels))

    model = FineTuneNet(channel_groups, params)

    split = dataset.split("train")
    train = split["True"]
    test = split["False"]

    train_loader = torch.utils.data.DataLoader(train, batch_size=params["batch_size"], shuffle=params["shuffle"],
                                               num_workers=1)

    # print(train.datasets[0].targets_from)
    # print(train.datasets[0].__getitem__(0))

    for epoch in range(epochs):
        print('epoch number: ', epoch, 'of: ', epochs)
        for X, y, crop_inds in tqdm(train_loader, position=0, leave=True):
            enc = model(X)
        print("Encoded!")
