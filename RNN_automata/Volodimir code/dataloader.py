import numpy as np
# import tensorflow as tf
import torch
from torch.utils.data import Dataset
import os

MAX_TIMESTEPS = 100

def data_proces(path,name):
    data = []
    labels = []
    with open(path+name) as file:
        for line in file.readlines():
            labeled_line = line.split()
            data.append(labeled_line[0])
            labels.append(labeled_line[1])
    return data, labels

def get_max_length(data):
    return max([len(x) for x in data])
    

def splitter(word): # not usefull if don't need a mutable
    res = []
    for j in word:
        res.append(j)
    return res


def alphabet_extractor(data):
    alphabet = []
    for elem in data:
        for letter in elem:
            alphabet.append(letter)
    alphabet = set(alphabet)
    return sorted(list(alphabet))
    

def integer_embeder(alphabet,letter):
    return alphabet.index(letter)


def label_encoder(label):
    return label == 'TRUE'


def letter_one_hot_encoder(one_hot_list,alphabet,letter):
    indx = integer_embeder(alphabet,letter)
    return one_hot_list[indx]



# ### test zone
# data, labels = data_proces(path,name)
# alphabet = alphabet_extractor(data)
# print(alphabet)
# word = data[0]
# print(word)
# for j in word:
#     integ = integer_embeder(alphabet,j)
#     print(integ)

# ### end test zone


def preprocess(line, nb_class): #In the preprocess, the x is the sentence, and the y is the sentence shifted one element left
    return letter_one_hot_encoder(nb_class, line), line[1:] # get 2 instead of 3 parameters


def get_vocabulary_size(data): #get the vocabulary size, will be useful for the embedding layer
    alphabet = alphabet_extractor(data)
    return len(alphabet)


def load_spice_dataset(sample_name:str, path="data/Small/"): #Load the spice sample with the number given as args contained in the folder $path
    data, labels = data_proces(path, name) # name not define
    nb_class = get_vocabulary_size(data)
    lines = list(map(lambda x : preprocess(x, nb_class), lines)) #we preprocess every lines
    return lines

class Dataset(torch.utils.data.Dataset):#type: ignore
    def __init__(self,path,name,length=None):
        self.X, self.y = data_proces(path,name)
        self.alphabet = alphabet_extractor(self.X)
        self.target_alphabet = ['TRUE','FALSE']
        self.one_hot = []
        for j in range(len(self.alphabet)):
            self.one_hot.append(np.eye(len(self.alphabet))[:,j])
        
        if length == None:
            self.padding_length = get_max_length(self.X)
        else:
            self.padding_length = length
    def print_one_hot_alphabet(self):
        print(self.one_hot)
    
    def __len__(self):
        return len(self.X)
    def __getitem__(self, index):
        return self.to_one_hot(self.X[index]), self.label_to_one_hot(self.y[index])#, dtype=torch.float32)
    def to_one_hot(self, x):
        x_one_hot = []
        for letter in x:
            x_one_hot.append(letter_one_hot_encoder(self.one_hot,self.alphabet,letter))
        for _ in range(self.padding_length - len(x)):
            x_one_hot.append(np.zeros((len(self.alphabet),)))
        x_one_hot.append(len(x)*np.ones((len(self.alphabet),)))
        x_one_hot = np.array(x_one_hot)
        return torch.tensor(x_one_hot, dtype=torch.float32)

    def label_to_one_hot(self,y):
        y_one_hot = label_encoder(y)
        return torch.tensor(y_one_hot, dtype=torch.int16)
    def alphabet_len(self):
        return len(self.alphabet)