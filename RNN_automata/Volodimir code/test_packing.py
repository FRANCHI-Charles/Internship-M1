import torch
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

seq = torch.tensor([[1, 2, 0], [3, 0, 0], [4, 5, 6]])
lens = [2, 1, 3]
packed = pack_padded_sequence(seq, lens, batch_first=True, enforce_sorted=False)
print(packed)

seq_unpacked, lens_unpacked = pad_packed_sequence(packed, batch_first=True)
print(seq_unpacked)
print(lens_unpacked)