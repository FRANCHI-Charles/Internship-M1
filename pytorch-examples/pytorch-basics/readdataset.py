from torch.utils.data import Dataset, DataLoader
import torch
import torchvision
from pandas import read_csv
from sklearn.preprocessing import LabelEncoder

torch.manual_seed(42)

class Pistachios(Dataset):

    def __init__(self, transform = None) -> None:
        #super().__init__() not needed
        self.transform = transform
        load = read_csv("./data/Pistachio.csv", sep=",")
        lab = LabelEncoder()

        self.x = torch.tensor(load.drop("Class", axis=1).values, dtype=torch.float32)
        self.y = torch.tensor(lab.fit_transform(load["Class"]), dtype=torch.int)


    def __len__(self):
        return self.y.shape[0]
    

    def __getitem__(self, index):
        value = self.x[index]
        if self.transform:
            value = self.transform(value)
        return value, self.y[index]
    

data = Pistachios()

dataload = DataLoader(data, batch_size=4, shuffle=True)

test = iter(dataload)

print(next(test))
print(next(test))