import torch
import torch.nn as nn
from DFA2SRN import machine_process
from automata import DFA, TorchData

class STATS():
    def __init__(self,lis_stats):
        ### this class is designed to store any kind of statistics typed as lists. It takes as input a list of strings
        self.names = lis_stats
        self.dict  = {} 
        for elem in lis_stats:
            self.dict[elem] = []
    
        
    def push(self,name,element):
        self.dict[name].append(element)

    def get_data(self):
        ls = []
        for elem in self.names:
            ls.append(self.dict[elem])
        return ls
    
# class customSRN(nn.Module):

#     def __init__(self, hidden_dim, input_dim, output_dim, seq_length, num_layers=1, bidirectional=False, activation='tanh', device='cpu', dtype=torch.float32):
#         super().__init__()
#         self.seqlen = seq_length
#         self.hidden = hidden_dim
#         self.input  = input_dim
#         self.output = output_dim
#         self.layers = num_layers
#         self.bidire = bidirectional
#         self.dtype = dtype
#         self.device = device
#         self.beta   = 1
#         self.feta   = 0
#         self.Fbeta = {'value':[],'position':[]}
#         if activation == 'sigmoid':
#             self.activ = nn.Sigmoid()
#         elif activation == 'tanh':
#             self.activ = nn.Tanh()
#         else:
#             self.activ = nn.ReLU()

#         self.U      = nn.Linear(self.input,  self.hidden).to(dtype=torch.float32, device = device)
#         self.W      = nn.Linear(self.hidden, self.hidden).to(dtype=torch.float32, device = device)
#         self.fc     = nn.Linear(self.hidden, self.output).to(dtype=torch.float32, device = device)

#     def forward(self,x):
#         self.Fbeta = {'value':[],'position':[]}
#         pad_len = x[:,-1,:]
#         x = x[:,:-1,:].to(dtype = self.dtype,device=self.device)  
#         h = torch.zeros(x.size(0), self.hidden).to(dtype=self.dtype,device=self.device) # bqatch x hiddendi
#         outs = torch.zeros(x.size(0),self.output,self.seqlen-1).to(dtype = self.dtype,device=self.device) 
#         mask = torch.zeros(x.size(0),self.seqlen-1).to(dtype = self.dtype,device=self.device)
#         x = torch.transpose(x,2,1)
        
#         for j in range(x.shape[2]-1): 
#             mask[:,j]= (pad_len[:,0]==j*torch.ones((x.size(0),)).to(dtype = self.dtype, device=self.device)) ## cration of the padding mask
#             y1 = self.U(x[:,:,j]) #bqtch x hidden
#             y2 = self.W(h)
#             h = self.activ(y1+y2) # bTCH X HIDDEN
#             out = self.fc(h) # OUTPUT LOGITS
#             outs[:,:,j] = self.activ(out)
#             ab = torch.abs(h)
#             # print(ab.shape)
#             beta = torch.min(ab)
#             feta = torch.max(ab).detach()
#             A = (beta<=self.beta)
#             B = (feta>=self.feta)
#             mi, index = ab.min(dim=1)
#             self.Fbeta['value'].append(mi.detach().cpu().numpy())
#             self.Fbeta['position'].append(index.detach().cpu().numpy())
#             self.beta = beta*A + self.beta*(not A)
#             self.feta = feta*B + self.feta*(not B)
#         for k in range(outs.shape[1]):
#             outs[:,k,:][mask==0] = 0        ## aplying the padding mask
#         outs = torch.sum(outs,-1) ## extracting the usfull classification 
#         return outs.to(dtype=torch.float64,device=self.device)
    
#     def get_beta(self):
#         return self.beta

#     def reset_beta(self):
#         self.beta = 1
#         self.feta = 0

#     def get_Fbeta(self):
#         return self.Fbeta
    
#     def reset_Fbeta(self):
#         self.Fbeta = {'value':[],'position':[]}

#     def get_matrixW(self):
#         return self.state_dict()['W.weight']


def lang_names(file_path):
    "Read the .txt file with the names of the dataset to use."
    names = []
    extensions = []
    with open(file_path) as file:
        readnames = True
        for line in file.readlines():
            if readnames:
                if line != '/n':
                    names.append(line[:-1])
                else:
                    readnames = False
            else:
                extensions.append(line[:-1])
            
    return names, extensions



def data_load(file_path) -> tuple[list[str], list[int]] :
    "Load the automaton dataset."
    data = []
    labels = []
    with open(file_path) as file:
        for line in file.readlines():
            labeled_line = line.split()
            data.append(labeled_line[0])
            labels.append(labeled_line[1])

    labels = [1 if label == 'TRUE' else 0 for label in labels]

    return data, labels

# def get_max_length(data) -> int:
#     "Max sequence length in data."
#     return max([len(x) for x in data])


def alphabet_extractor(data) -> str:
    alphabet = []
    for elem in data:
        for lettre in elem:
            alphabet.append(lettre)
    alphabet = "".join(sorted(set(alphabet)))
    return alphabet


def att_to_DFA(file_path, alphabet):
    "Load .att files as a dict of DFA class."
    transitions, finites = machine_process(file_path)
    output = DFA(transitions.T, finites, letters=alphabet)

    return output


def data_automata_loading(att_path:str, data_path:str, name:str, data_ext:list = [""], return_automata = False):
    """
    Load the data as a DFA object and a TorchData object.

    Parameters
    ----------
    att_path : str
        Path to the .att file
    data_path : str
        Path to the dataset (.txt) file
    name : str
        Name of the automaton used (same for .att and dataset)
    data_ext : list[str]
        Extension names for the dataset.
    return_automata : Bool
        Choose to return the DFA or not. The DFA will have the last data in self.data
    """
    datas = dict()
    alphabet = ""
    for e in data_ext:
        datas[e] = data_load(data_path + name + e + ".txt")
        alphabet = "".join(sorted(set(alphabet + alphabet_extractor(datas[e][0]))))
    
    automaton = att_to_DFA(att_path + name + ".att", alphabet)
    for e in data_ext:
        automaton.data = datas[e]
        datas[e] = TorchData(automaton)

    return (datas, automaton) if return_automata else datas
