import torch
import torch.nn as nn
from torch.utils.data import Dataset
import numpy as np
import matplotlib.pyplot as plt
from DFA2SRN import machine_process
from automata import DFA, TorchData


STATS_NAMES = ["l2", "linf", "targetdist", "linedist", "losses", "acc"]


def lang_names(file_path):
    "Read the .txt file with the names of the dataset to use."
    names = []
    extensions = []
    with open(file_path) as file:
        readnames = True
        for line in file.readlines():
            if readnames:
                if line != '\n':
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
    output = DFA(transitions.T, finites, probas="equal", letters=alphabet)

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



def stochastic(dataset:Dataset, mini_batch:int):
    batch_index = np.random.choice(range(len(dataset)), mini_batch)
    data = list()
    lengths = list()
    labels = list()
    for elem in batch_index:
        data.append(dataset[elem][0])
        lengths.append(dataset[elem][1])
        labels.append(dataset[elem][2])
        
    return torch.stack(data), lengths, labels


### Compute stats

def initstats(names:list[str]):
    dico = dict()
    for name in names:
        dico[name] = list()
    return dico

def linedist(point, linepoint):
    """Compute the distance between the `point` and the line form by the 2 points `linepoint` and 0."""
    return torch.sqrt(torch.abs(torch.linalg.vector_norm(point)**2 - (torch.sum(point * linepoint) / torch.linalg.vector_norm(linepoint))**2)).item()
    

def stats(net:nn.Module,target,lr):
    "Returns the L2 and the L_\infty norm of the gradient and the distance to target."
    norm_2_acc = 0
    norm_inf_acc = []
    flatten_target = torch.cat([target[0].flatten(), target[2].flatten(), target[4].flatten()])
    flatten_param = torch.zeros(flatten_target.shape)
    last_position = 0
    for parameters in net.parameters():
        gr = parameters.grad.flatten()
        norm_2_acc += (torch.linalg.norm(gr,ord=2).item())**2
        norm_inf_acc.append(torch.linalg.norm(gr,ord=float('inf')).item())
        with torch.no_grad():
            parameters = parameters.flatten()
            flatten_param[last_position : last_position + torch.numel(parameters)] = parameters
            last_position += torch.numel(parameters)
        disttoline = linedist(flatten_param, flatten_target)
        target_dist = torch.linalg.vector_norm(flatten_target - flatten_param).item()
        norm_2 = lr*norm_2_acc**(0.5)
        norm_inf = max(norm_inf_acc) * lr
        return norm_2, norm_inf, disttoline, target_dist
    

def plot_stats(*stats:tuple[dict], plotaspects = ["r", "g", "b"], labels=["Unconstrained SRN", "Constrained SRN"]):
    "Plot the stats L2, Linf, distance to target, distance to line target, losses and accuracy."
    _, axes = plt.subplots(2,3, figsize = (12,6))

    if len(stats) > len(plotaspects):
        raise ValueError("Not enough plotaspects.")
    

    for i, name in enumerate(STATS_NAMES):
        for statsdict, layout, label in zip(stats, plotaspects, labels):
            axes[i%2,i//2].plot(statsdict[name], layout, label=label)
        axes[i%2,i//2].set_title(name)
        axes[i%2,i//2].legend()

    plt.show()
