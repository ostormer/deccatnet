from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler, label_binarize
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score
import torch
from torch.utils.data import DataLoader, Dataset
import numpy as np

from inceptionTime import InceptionTime

class NpDataset(Dataset):
    def __init__(self, x, y):
        assert x.shape[0] == y.shape[0]
        self.x = torch.tensor(x_test).float()
        self.y = torch.tensor(y).type(torch.LongTensor)
        self.classes = np.unique(y)
        self.num = 0
    
    def __len__(self):
        return len(self.x)

    def __getitem__(self, i):
        return self.x[i], self.y[i]

    def __iter__(self):
        return self

    def __next__(self):
        if self.num >= len(self.x):
            raise StopIteration
        else:
            self.num += 1
            return self.__getitem__(self.num - 1)


def train(model, train_loader, val_loader, epochs=5, loss_fn=torch.nn.CrossEntropyLoss()):
    # Move to GPU if available
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(device)
    model.to(device)

    optimizer = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()), lr=0.001)
    min_valid_loss = np.inf
    for epoch in range(epochs):
        train_loss = 0.0
        print(f"Starting epoch {epoch}")
        for i, (data, labels) in enumerate(train_loader):
            # Transfer Data to GPU if available
            data, labels = data.to(device), labels.to(device)

            # Clear the gradients
            optimizer.zero_grad()
            # Forward Pass
            target = model(data)
            # Find the Loss
            loss = loss_fn(target, labels)
            # Calculate gradients
            loss.backward()
            # Update Weights
            optimizer.step()
            # Calculate Loss
            train_loss += loss.item()

            if i % 1000 == 999:
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {train_loss / 2000:.3f}')
                train_loss = 0.0

        valid_loss = 0.0
        model.eval()     # Optional when not using Model Specific layer
        for data, labels in val_loader:
            # Transfer Data to GPU if available
            data, labels = data.to(device), labels.to(device)

            # Forward Pass
            target = model(data)
            # Find the Loss
            loss = loss_fn(target, labels)
            # Calculate Loss
            valid_loss += loss.item()

        print(f'Epoch {epoch + 1} \t\t Training Loss: {train_loss / len(train_loader)} \t\t Validation Loss: {valid_loss / len(val_loader)}')

        if min_valid_loss > valid_loss:
            print(f'Validation Loss Decreased({min_valid_loss:.6f}--->{valid_loss:.6f}) \t Saving The Model')
            min_valid_loss = valid_loss

            # Saving State Dict
            torch.save(model.state_dict(), 'saved_model.pth')

x = np.vstack((np.load("data/sequenced_data_for_VAE_length-160_stride-10_pt1.npy"),
               np.load("data/sequenced_data_for_VAE_length-160_stride-10_pt2.npy")))
y = np.load("data/sequenced_data_for_VAE_length-160_stride-10_targets.npy")
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=69420)

scaler = RobustScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

data_train = NpDataset(x_train, y_train)
data_test = NpDataset(x_test, y_test)
loader_train = DataLoader(data_train)
loader_test = DataLoader(data_test)

model = InceptionTime(n_classes=4)
model.cuda()
step_size = 300

# train(model, loader_train, loader_test, epochs=1)
model.load_state_dict(torch.load("saved_model.pth"))
model.cpu()
model.eval()

x_test = torch.tensor(x_test).float()
with torch.no_grad():
    x_pred = np.argmax(model(x_test).detach(), axis=1)
print(x_pred)
print("F1: ", f1_score(y_true=y_test, y_pred=x_pred, average="macro"))
print("Accuracy: ", accuracy_score(y_true=y_test, y_pred=x_pred))
cf = confusion_matrix(y_true=y_test, y_pred=x_pred)
print("Confusion matrix:\n")
print(cf)
