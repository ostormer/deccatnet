import torch
import torch.nn as nn
from tqdm import tqdm

from .DECCaTNet_model import Encoder


class EncodingClassifier(nn.Module):
    def __init__(self):
        pass

    def forward(self, X):
        return X


class FineTuneNet(nn.Module):
    def __init__(self, params):
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
        self.embedding_size = params["embedding_size"]
        self.n_classes = params["n_classes"]

        self.encoder = Encoder(emb_size=self.embedding_size)
        self.encoder.load_state_dict(torch.load(self.encoder_path))
        self.encoder.requires_grad_(False)

        self.n_channel_groups = 8  # TODO: SET TO CORRECT VALUE FROM LOADING A SAMPLE

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
        encoder_inputs = torch.split(X, self.channel_group_size, dim=0)
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

    model = FineTuneNet(params)

    split = dataset.split("train")
    print(split)
    train = split["True"]
    test = split["False"]
    print(train.description)
    print(test.description)

    train_loader = torch.utils.data.DataLoader(train, batch_size=params["batch_size"], shuffle=params["shuffle"],
                                               num_workers=1)

    for epoch in range(epochs):
        print('epoch number: ', epoch, 'of: ', epochs)
        # for batch in train_loader:
        #     print(batch)
        for X, y in tqdm(train_loader, position=0, leave=True):
            print(f"X: {X}\n\n"
                  f"y: {y}")
