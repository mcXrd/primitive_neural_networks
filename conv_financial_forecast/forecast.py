# %%
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.tensorboard import SummaryWriter
import pandas as pd

writer = SummaryWriter()
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)


class CoinsDataset(Dataset):

    # Constructor with defult values
    def __init__(self, df, transform=None):
        self.df = df
        self.len = len(self.df) - 1
        self.transform = transform

    # Getter
    def __getitem__(self, index):
        sample = torch.from_numpy(self.df.iloc[index].values), torch.from_numpy(self.df.iloc[index + 1].values)
        sample = sample[0].to(device), sample[1].to(device)
        if self.transform:
            sample = self.transform(sample)
        return sample

    # Get Length
    def __len__(self):
        return self.len


df = pd.read_hdf("binance_data_sep_16.hdf")
df = df.dropna()
coindataset = CoinsDataset(df=df)
BATCH_SIZE = 6
train_loader = torch.utils.data.DataLoader(dataset=coindataset, batch_size=BATCH_SIZE, drop_last=True)
validation_loader = torch.utils.data.DataLoader(dataset=coindataset, batch_size=BATCH_SIZE, drop_last=True)


model = torch.nn.Sequential(
    torch.nn.Linear(253, 253),
    nn.BatchNorm1d(253),
    torch.nn.ReLU(),
    torch.nn.Linear(253, 253),
    nn.BatchNorm1d(253),
    torch.nn.ReLU(),
    torch.nn.Linear(253, 1000),
    nn.BatchNorm1d(1000),
    torch.nn.ReLU(),
    torch.nn.Linear(1000, 1000),
    nn.Dropout(p=0.1),
    torch.nn.ReLU(),
    torch.nn.Linear(1000, 253),
    nn.Dropout(p=0.1),
    torch.nn.ReLU(),
    torch.nn.Linear(253, 253),
)


model = model.float()
model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)
# optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.001)
criterion = nn.MSELoss()


# %%
def train_model(model, train_loader, validation_loader, optimizer, n_epochs=None):
    # global variable
    N_test = len(coindataset)
    accuracy_list = []
    loss_list = []
    for epoch in range(n_epochs):
        for x, y in train_loader:
            a = 1
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
            a = y_test.size()
            b = yhat.size()
            c = z.data.size()
            a = 1
            correct = torch.mean(z.data - y_test)
            accuracy = correct / N_test
            writer.add_scalar("Accuracy/train", correct, epoch)
            accuracy_list.append(accuracy)
    return accuracy_list, loss_list


# %%
train_model(model, train_loader, validation_loader, optimizer, n_epochs=200)  # %%

model.eval()
tt = torch.Tensor([[2.0], [20.0], [30.0], [60.0], [70.0], [51.0], [90.0], [30000.0]])
# print("sr of 2 and 3")
# print(np.sqrt(tt))
# print(model(torch.Tensor(tt).float()))

last_row = torch.from_numpy(df.iloc[-BATCH_SIZE:].values).float()

print("last row")
print(last_row[0][0])
print("prediction")
print(model(last_row.to(device))[0][0])

print("last row")
print(last_row[5][0])
print("prediction")
print(model(last_row.to(device))[5][0])
