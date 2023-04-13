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
    def __init__(self, channel_groups, ds_channel_order, params):
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

        self.ds_channel_order = ds_channel_order
        # Make dict witch translates channel names to index in preprocessed files
        self.channel_index = {}
        for i, ch in enumerate(self.ds_channel_order):
            self.channel_index[ch] = i
        # Use channel index dict to define channel groups by indexes instead of channel names
        self.channel_index_groups = []
        for group in self.channel_groups:
            self.channel_index_groups.append([self.channel_index[ch] for ch in group])

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
        print("New sample:!!!")
        print(X.shape)
        X = X[:, None, :, :]
        # Split input into chunks that fit into the encoder
        # TODO: Decide what to do if n_channels does not fit into encoder size (Not even)
        # Do we discard the last channels? Do we make a overlap
        # TODO: Define splits from channel group index list from init to reduce run time
        channel_group_tensors = []

        for indexes in self.channel_index_groups:
            print(indexes)
            # Select the correct channel for each window in batch, concatenate those to new tensor
            channel_group_tensors.append(
                torch.concat([X[..., i:i+1, :] for i in indexes], dim=-2)
            )
            print(channel_group_tensors[-1].shape)



        # TODO: compare speed of the above with ll
        # encoder_inputs = torch.split(X, self.channel_group_size, dim=0)

        encoder_out = []
        # Run each group/pair of channels through the encoder
        for group in channel_group_tensors:
            print(f"Group shape: {group.shape}")
            encoder_out.append(self.encoder(group))  # Encode group

        print(encoder_out[0].shape)
        X_encoded = torch.concat(encoder_out, dim=0)
        print(X_encoded.shape)
        X_encoded = torch.reshape(X_encoded, (X_encoded.shape[0], -1))
        X_encoded = torch.flatten(X_encoded)
        x = self.classifier(X_encoded)
        return x


def run_fine_tuning(dataset, params):
    epochs = params["epochs"]
    ds_channel_order = dataset.datasets[0].windows.ch_names
    for windows_ds in dataset.datasets:
        assert windows_ds.windows.ch_names == ds_channel_order
    print("All recordings have the correct channel order")

    channel_groups = _make_adjacent_pairs(ds_channel_order)
    model = FineTuneNet(channel_groups, ds_channel_order, params)

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
