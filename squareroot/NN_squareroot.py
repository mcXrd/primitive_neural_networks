# %%
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter
import numpy as np

writer = SummaryWriter()

# %%
class SquareRootDataset(Dataset):

    # Constructor with defult values
    def __init__(self, length=1000, transform=None):
        self.len = length
        self.np_x = np.random.uniform(low=0, high=100, size=self.len)
        self.np_y = np.sqrt(self.np_x)
        self.x = torch.from_numpy(self.np_x)
        self.x = self.x.view(-1, 1)
        self.y = torch.from_numpy(self.np_y)
        self.y = self.y.view(-1, 1)
        self.transform = transform

    # Getter
    def __getitem__(self, index):
        sample = self.x[index], self.y[index]
        if self.transform:
            sample = self.transform(sample)
        return sample

    # Get Length
    def __len__(self):
        return self.len


# %%

# %%
class NN(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(1, 10)
        self.linear2 = nn.Linear(10, 50)
        self.linear3 = nn.Linear(50, 1000)
        self.linear4 = nn.Linear(1000, 1000)
        #raise Exception(len(self.linear4.weight))
        self.linear5 = nn.Linear(1000, 1000)
        self.linear6 = nn.Linear(1000, 1)
        self.bn1 = nn.BatchNorm1d(10)
        self.bn2 = nn.BatchNorm1d(50)
        self.bn3 = nn.BatchNorm1d(1000)
        self.bn4 = nn.BatchNorm1d(1000)
        self.bn5 = nn.BatchNorm1d(1000)

        self.dropout = nn.Dropout(0.2)

        #self.conv = nn.Conv1d(in_channels=16, out_channels=4, kernel_size=10, stride=1, padding_mode='zeros')

    # Prediction
    def forward(self, x):
        x = self.linear1(x)
        x = self.bn1(x)
        x = torch.relu(x)

        x = self.linear2(x)
        x = self.bn2(x)
        x = torch.relu(x)

        x = self.linear3(x)
        x = self.bn3(x)
        x = torch.relu(x)

        x = self.linear4(x)
        x = self.bn4(x)
        x = torch.relu(x)
        x = self.dropout(x)

        x = self.linear5(x)
        x = self.bn5(x)
        x = torch.relu(x)

        x = self.linear6(x)
        return x


# %%
sqdataset = SquareRootDataset(length=200)
train_loader = torch.utils.data.DataLoader(dataset=sqdataset, batch_size=8)
validation_loader = torch.utils.data.DataLoader(dataset=sqdataset, batch_size=8)
# %%

model = NN()
model = model.float()
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
# optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.001)
criterion = nn.MSELoss()


# %%
def train_model(model, train_loader, validation_loader, optimizer, n_epochs=None):
    # global variable
    N_test = len(sqdataset)
    accuracy_list = []
    loss_list = []
    for epoch in range(n_epochs):
        for x, y in train_loader:
            model.train()
            optimizer.zero_grad()
            z = model(x.float())
            loss = criterion(z, y.float())
            writer.add_scalar("Loss/train", loss, epoch)
            loss.backward()
            optimizer.step()
            loss_list.append(loss.data)
            correct = 0
        # perform a prediction on the validation data
        for x_test, y_test in validation_loader:
            model.eval()
            z = model(x_test.float())
            # print(model(torch.Tensor(2).float()))
            _, yhat = torch.max(z.data, 1)
            correct = torch.mean(y_test - yhat)
            accuracy = correct / N_test
            writer.add_scalar("Accuracy/train", correct, epoch)
            accuracy_list.append(accuracy)
    return accuracy_list, loss_list


# %%
train_model(model, train_loader, validation_loader, optimizer, n_epochs=50)  # %%

model.eval()
tt = torch.Tensor([[2.0], [20.0], [30.0], [60.0], [70.0], [51.0], [90.0], [30000.0]])
print("sr of 2 and 3")
print(np.sqrt(tt))
print(model(torch.Tensor(tt).float()))
