from torch.utils.data import Dataset, DataLoader, random_split
import torch
import torch.nn as nn
from tqdm import tqdm # for loading bar

### Usefull functions
from utils import N_LETTERS, line_to_tensor_size, load_data #, findmax

### Reproducibility
torch.manual_seed(689)


### Hyperparameters
PAR_PATH = "./data/parameters.pt" # saved parameters path
MAXSIZE = 19 # maximum length of a name in the dataset (print(findmax(category_lines)) answer is 19)

HIDDEN_SIZE = 20
N_LAYERS = 2

LR = 0.01
N_EPOCH = 15
BATCH_SIZE = 150
TESTSIZE = 0.25

### GPU avaible ?
device = torch.device("cpu")#"cuda" if torch.cuda.is_available() else "cpu")
# My GPU is too old...


### GRU Neural Network
class Model(nn.Module):

    def __init__(self, all_cat, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.gru = nn.GRU(N_LETTERS, HIDDEN_SIZE, N_LAYERS, batch_first=True, device=device)
        self.toclass = nn.Linear(HIDDEN_SIZE, len(all_cat), device=device)

    def forward(self, x, truelen):
        "`truelen` is a list of the real length of the sequence : that way we can recover the good prediction along the gru"
        h0 = torch.zeros(N_LAYERS, x.shape[0], HIDDEN_SIZE).to(device)
        out, _ = self.gru(x.to(device), h0)
        out = torch.stack([out[i, truelen[i] -1, :] for i in range(out.shape[0])]) #extract only the require prediction y for each batch
        return self.toclass(out)
        
    def predict(self, x, truelen):
        return torch.argmax(self(x, truelen), dim = 1)
    
    def strpredict(self, name:str):
        tensorname = line_to_tensor_size(name, len(name))[None,:,:] # put a batch size of 0
        prediction = self.predict(tensorname, [len(name)])
        return all_categories[prediction.item()]


### Dataset
class Names(Dataset):
    def __init__(self, cat_names, all_cat) -> None:
        #super().__init__() not needed
        self.X = list()
        self.namelen = list() #len of the original names
        self.y = list()
        for i in range(len(all_cat)): # put every categories in one list to have a real dataset
            cat = all_cat[i]
            self.X += [line_to_tensor_size(name, size=MAXSIZE) for name in cat_names[cat]]
            self.namelen += [len(name) for name in cat_names[cat]]
            self.y += [torch.tensor([i]) for _ in range(len(cat_names[cat]))]

        #assert len(self.X) == len(self.y)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, index):
        return self.X[index], self.namelen[index], self.y[index]
    
    def __iter__(self):
        self.indicies = torch.randperm(len(self.X))
        self.i = -1
        return self
    
    def __next__(self):
        self.i += 1
        if self.i == len(self.X):
            self.i = 0
        return self[self.indicies[self.i]]

### Load the data
category_lines, all_categories = load_data()
data = Names(category_lines, all_categories)

traindata, testdata = random_split(data, [1-TESTSIZE, TESTSIZE]) # traintest split
testdata = next(iter(DataLoader(testdata, len(testdata)))) #transform class Subset as tuple of tensors

loader = DataLoader(traindata, batch_size=BATCH_SIZE, shuffle=True)

model = Model(all_categories).to(device)
loss = nn.CrossEntropyLoss()
optim = torch.optim.Adam(model.parameters(), lr=LR)


if input("Do you want to re-train the model (Yes/Y/yes/y): ") in ["Yes", "yes", "Y", "y"]:
    for epoch in range(N_EPOCH):
        dataiter = iter(loader)
        for _ in tqdm(range(len(loader)), desc=f"Epoch {epoch+1}...", ncols=75):
            trainX, trainlen, trainy = next(dataiter)
            trainy = trainy.reshape(-1).to(device)

            out = model(trainX, trainlen)
            
            error = loss(out, trainy)
            error.backward()

            optim.step()
            optim.zero_grad()

        with torch.no_grad():
            predict = model.predict(testdata[0], testdata[1])

            acc = torch.sum(predict == testdata[2].reshape(-1), dim=0) * 100 / len(testdata[2])
            
            print(f"Epoch {epoch+1} done ! Test accuracy : {acc:.4f}")


        torch.save(model.state_dict(), PAR_PATH)

else:
    model.load_state_dict(torch.load(PAR_PATH))
    model.eval()
    print("Model succesfully loaded !")

while True:
    name = input("Name to predict ('q' to quit):")

    if name == 'q':
        break
    try:
        print(f"It mays be {model.strpredict(name)} !")
    except Exception as e:
        print(e)
        