import torch
from torch.utils.data import DataLoader
import numpy as np

from DFA2SRN import dfa2srn, sigmoid_to_tanh
import automata as dfa

SIZE = 10000
MEANLEN = 12
RANDOM = 42

device = torch.device("cpu")

### Automaton

unique_transitions = np.array([[1,2,3,4],
                               [1,-1,-1,-1],
                                [-1,2,-1,-1],
                                [-1,-1,3,-1],
                                [-1,-1,-1,4],
                                [-1,-1,-1,-1]])
unique_finites = np.array([1,1,1,1,1,0])

uniquedfa = dfa.DFA(unique_transitions, unique_finites)


### Dataset

uniquedfa.dataset(SIZE, meanlen=MEANLEN, random_state=RANDOM)

dataset = dfa.TorchData(uniquedfa)

batch, lenghts, labels = next(iter(DataLoader(dataset, len(dataset), shuffle=True)))

### Model

target = dfa2srn(unique_transitions.T, unique_finites)
target = sigmoid_to_tanh(target)

print(target)

model = dfa.AutomataRNN(uniquedfa, device=device)

model.set_parameters(target)

predicted = model(batch, lenghts)
print(predicted, labels)

accuracy = torch.sum(predicted == labels).item() / len(labels)

print(accuracy)