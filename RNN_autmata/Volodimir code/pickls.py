import pickle
import torch


import io

class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else:
            return super().find_class(module, name)

#contents = pickle.load(f) becomes...
# contents = CPU_Unpickler(f).load()

with open('perf_dict_saturate.pickle', 'rb') as handle:
    perf_dict = CPU_Unpickler(handle).load()

for key, value in perf_dict.items():
    print(value['test_acc'])


