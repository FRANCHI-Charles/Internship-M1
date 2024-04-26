import torch
import torch.nn as nn
from sklearn import datasets

n_iter = 100
learning_rates = 0.01

class Mylinear(nn.Module):

    def __init__(self, n_input, n_hidden):
        super().__init__()
        self.lin1 = nn.Linear(n_input, n_hidden)
        self.lin2 = nn.Linear(n_hidden,1)

    def forward(self, x):
        return torch.sigmoid(self.lin2(self.lin1(x)))
        

#### Create Artificial dataset
npXdata, npydata = datasets.make_classification(n_samples=100, n_features=3, n_informative=2, n_redundant=0, random_state=42)

Xdata = torch.from_numpy(npXdata).type(torch.float32)
ydata = torch.from_numpy(npydata).type(torch.float32).view(-1,1)

print(Xdata[:4])



#### Train the model

model = Mylinear(3,4)
loss = nn.BCELoss()

optimizer = torch.optim.SGD(model.parameters(), lr=learning_rates)
for epoch in range(n_iter):
    y_predict = model(Xdata)

    l = loss(y_predict, ydata)

    l.backward()

    optimizer.step()

    optimizer.zero_grad()

    if (epoch+1) %10 == 0:
        print(f"epoch {epoch +1} : loss = {l}, {list(model.parameters())}")


with torch.no_grad():
    y_pred = model(Xdata).round()

    acc = y_pred.eq(ydata).sum()/ ydata.shape[0]

    print(acc.item())