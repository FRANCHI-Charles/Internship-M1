# data: https://download.pytorch.org/tutorial/data.zip
import io
import os
import unicodedata
import string
import glob

import tensorflow as tf
import random

# alphabet small + capital letters + " .,;'"
ALL_LETTERS = string.ascii_letters + " .,;'"
N_LETTERS = len(ALL_LETTERS)

# Turn a Unicode string to plain ASCII, thanks to https://stackoverflow.com/a/518232/2809427
def unicode_to_ascii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in ALL_LETTERS
    )

def load_data():
    # Build the category_lines dictionary, a list of names per language
    category_lines = {}
    all_categories = []
    
    def find_files(path):
        return glob.glob(path)
    
    # Read a file and split into lines
    def read_lines(filename):
        lines = io.open(filename, encoding='utf-8').read().strip().split('\n')
        return [unicode_to_ascii(line) for line in lines]
    
    for filename in find_files('data/names/*.txt'):
        category = os.path.splitext(os.path.basename(filename))[0]
        all_categories.append(category)
        
        lines = read_lines(filename)
        category_lines[category] = lines
        
    return category_lines, all_categories


def findmax(category_lines):
    maxlen = 0
    for key in category_lines.keys():
        candidate = max([len(line) for line in category_lines[key]])
        if candidate > maxlen:
            maxlen = candidate

    return maxlen

# print(findmax(category_lines)) : answer is 16
"""
To represent a single letter, we use a “one-hot vector” of 
size <1 x n_letters>. A one-hot vector is filled with 0s
except for a 1 at index of the current letter, e.g. "b" = <0 1 0 0 0 ...>.

To make a word we join a bunch of those into a
2D matrix <line_length x 1 x n_letters>.

That extra 1 dimension is because PyTorch assumes
everything is in batches - we're just using a batch size of 1 here.
"""

# Find letter index from all_letters, e.g. "a" = 0
def letter_to_index(letter):
    return ALL_LETTERS.find(letter)

# Just for demonstration, turn a letter into a <1 x n_letters> TensorPAR_PATH = "./data/parameters.pt" # saved parameters path
def letter_to_tensor(letter):
    return tf.sparse.SparseTensor(indices=[[letter_to_index(letter)]], values=[1], dense_shape=(N_LETTERS,))

# Turn a line into a <line_length x n_letters>,
# or an array of one-hot letter vectors
def line_to_tensor(line):
    return line_to_tensor_size(line, len(line))

def line_to_tensor_size(line, size):
    "size version : no batch dimension"
    indices = list()
    for i, letter in enumerate(line):
        indices.append([i, letter_to_index(letter)])
    return tf.sparse.SparseTensor(indices, [1 for _ in indices], dense_shape=(size, N_LETTERS))


def random_training_example(category_lines, all_categories):
    
    def random_choice(a):
        random_idx = random.randint(0, len(a) - 1)
        return a[random_idx]
    
    category = random_choice(all_categories)
    line = random_choice(category_lines[category])
    category_tensor = tf.constant([all_categories.index(category)])
    line_tensor = line_to_tensor(line)
    return category, line, category_tensor, line_tensor

def training_example_i(category_lines, all_categories, category, index):
    category_tensor = tf.constant([all_categories.index(category)])
    line_tensor = line_to_tensor(category_lines[category][index])
    return category_tensor, line_tensor



def load_transform(maxlen, maxsize=10000):
    category_lines, all_categories = load_data()

    names = list()
    labels = list()
    lengths = list()

    for i, category in enumerate(all_categories):
        numberlines = len(category_lines[category])
        for j, name in enumerate(category_lines[category]):
            if maxsize <= j:
                numberlines = maxsize
                break
            names.append(line_to_tensor_size(name, maxlen))
            lengths.append(len(name))
            
        
        labels += [i for _ in range(numberlines)]

    return names, lengths, labels, all_categories
        
        


if __name__ == '__main__':
    print(ALL_LETTERS)
    print(unicode_to_ascii('Ślusàrski'))
    
    category_lines, all_categories = load_data()
    print(category_lines['Italian'][:5])
    
    print(letter_to_tensor('J')) # [1, 57]
    print(line_to_tensor('Jones').size()) # [5, 1, 57]PAR_PATH = "./data/parameters.pt" # saved parameters path