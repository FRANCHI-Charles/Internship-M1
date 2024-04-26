import torch
import torch.nn as nn
import torch.utils
import torch.utils.data
import torchvision
import matplotlib.pyplot as plt
from tqdm import tqdm

torch.manual_seed(42)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

##### hyperparameters
learning_rates = 0.005
batch_size = 100
test_size = 3000
n_epoch = 10
hidden_layer = 50
numlayers = 2
output_number = 10


##### Load the data
datatrain = torchvision.datasets.MNIST("./mnist", train=True, transform=torchvision.transforms.ToTensor())
datatest = torchvision.datasets.MNIST("./mnist", train=False, transform=torchvision.transforms.ToTensor())
# print(datatrain.data.shape)
# print(datatrain.targets.shape)
# print(datatest.data.shape)

train_load = torch.utils.data.DataLoader(datatrain, batch_size=batch_size, shuffle=True)
test_load = torch.utils.data.DataLoader(datatest, batch_size=test_size)
testing = next(iter(test_load))

print(datatrain[0][0].shape)

##### Neural networks
class myRNN(nn.Module):

    def __init__(self, hidden_layer, num_layers):
        super().__init__()
        self.layers = num_layers
        self.hid = hidden_layer
        self.rnn = nn.GRU(datatrain.data.shape[2], hidden_size=hidden_layer, num_layers=num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_layer, output_number)

    def forward(self, x):
        h0 = torch.zeros(self.layers, x.shape[0], self.hid).to(device)
        out, _ = self.rnn(x, h0)
        return self.linear(out[:,-1,:])
    
    def predict(self, x):
        to_probas = self.forward(x)
        return torch.argmax(to_probas, dim=1)


##### Training

model = myRNN(hidden_layer, numlayers).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rates)

for epoch in range(n_epoch):
    batch = iter(train_load)

    for _ in tqdm(range(len(train_load)), desc=f"Epoch {epoch+1}...", ncols=75):
        features, labels = next(batch)
        #print(features.shape)
        features = features[:,0].to(device)


        predict = model(features)
        loss = criterion(predict, labels.to(device))
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        acc = torch.sum(model.predict(testing[0][:,0].to(device)) == testing[1].to(device)).item() / test_size
        print(f"{epoch+1}-th loop : loss = {loss:.4f} ; test acc = {acc:.4f}")
    
quit()
##### Visualization

for i in range(15):
    plt.subplot(3,5, i+1)
    plt.imshow(testing[0][i:i+1,0,:,0].view(28,28), cmap="Blues")
    plt.title(f"Predicted class = {model.predict(testing[0][i:i+1,0,:,0]).item()}\nReal one = {testing[1][i]}")

plt.show()



