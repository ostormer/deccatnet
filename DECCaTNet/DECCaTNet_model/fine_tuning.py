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
                torch.concat([X[..., i:i + 1, :] for i in indexes], dim=-2)
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


def n_correct_preds(y_pred, y):
    num_correct = (torch.argmax(y_pred, dim=1) == y).float().sum().item()
    num_total = len(y)
    return num_correct, num_total


def run_fine_tuning(dataset, params):
    epochs = params["epochs"]
    learning_rate = params['lr_rate']
    weight_decay = params['weight_decay']
    num_workers = params['num_workers']
    ds_channel_order = dataset.datasets[0].windows.ch_names
    for windows_ds in dataset.datasets:
        assert windows_ds.windows.ch_names == ds_channel_order
    print("All recordings have the correct channel order")

    channel_groups = _make_adjacent_pairs(ds_channel_order)
    model = FineTuneNet(channel_groups, ds_channel_order, params)

    split = dataset.split("train")
    train = split["True"]
    test = split["False"]
    # TODO: get train/val/test set, will use test as val for now

    train_loader = torch.utils.data.DataLoader(train, batch_size=params["batch_size"], shuffle=params["shuffle"],
                                               num_workers=num_workers)
    val_loader = torch.utils.data.DataLoader(test, batch_size=params['batch_size'], shuffle=False,
                                             num_workers=num_workers)
    test_loader = torch.utils.data.DataLoader(test, batch_size=params['batch_size'], shuffle=False,
                                             num_workers=num_workers)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if torch.cuda.is_available():
        model.cuda()
        print("Moved model to CUDA")

    loss_func = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,
                                 weight_decay=weight_decay)  # TODO: check out betas for Adam and if Adam is the best choice

    loss = []
    train_acc = []
    val_acc = []
    for epoch in range(epochs):
        model.train()  # tells Pytorch Backend that model is trained (for example set dropout and have correct batchNorm)
        train_loss = 0
        correct_train_preds = 0
        num_train_preds = 0
        print('epoch number: ', epoch, 'of: ', epochs)
        for x, y, crop_inds in tqdm(train_loader, position=0, leave=True):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()

            # forward pass
            pred = model(x)
            # compute loss
            loss = loss_func(pred, y)
            # update weights
            loss.backward()
            optimizer.step()

            correct, number = n_correct_preds(pred, y)
            correct_train_preds += correct
            num_train_preds += number

            # track loss
            train_loss += loss.item()

            # free up cuda memory
            del x
            del y
            del pred
            torch.cuda.empty_cache()

        correct_eval_preds = 0
        num_eval_preds = 0
        with torch.no_grad(): #detach all gradients from tensors
            model.eval()  # tell model it is evaluation time
            for x, y, crops_inds in tqdm(val_loader, position=0, leave=True):

                x, y = x.to(device), y.to(device)
                # get predictions
                pred = model(x)
                # get correct preds and number of preds
                correct, number = n_correct_preds(pred, y)
                #update preds
                correct_eval_preds += correct
                num_eval_preds += number

                #cleare up memory
                del x
                del y
                del pred
                torch.cuda.empty_cache()

        # calculate accuracies and losses and update
        loss.append(train_loss/len(train_loader))
        train_acc.append(correct_train_preds/num_train_preds)
        val_acc.append(correct_eval_preds/num_eval_preds)


    # testing
    correct_test_preds = 0
    num_test_preds= 0
    with torch.no_grad(): # detach gradients
        model.eval() # set correct flags for evaluation/testing
        for x, y, crops_inds in tqdm(test_loader, position=0, leave=True):
            x, y = x.to(device), y.to(device)
            # get predictions
            pred = model(x)
            # get correct preds and number of preds
            correct, number = n_correct_preds(pred, y)
            # update preds
            correct_test_preds += correct
            num_test_preds += number

            # cleare up memory
            del x
            del y
            del pred
            torch.cuda.empty_cache()

    test_acc =
    # TODO Fininsh test acc, make it a function and implement k-fold validation strategy, could have both and check which is best, split into two functions to make is clean/nice
        # TODO: add all things from contrastive_framework as saving, pickle dumping etc
        # TODO: implement testing
        # TODO: remeber that some datasets (Abnormal/Normal) is already splitted, guessing this is implemented by Oskar.
        # TODO: implement pickel file

        print("Encoded!")
