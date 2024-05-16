# data: https://download.pytorch.org/tutorial/data.zip
import io
import os
import unicodedata
import string
import glob

import tensorflow as tf
import keras

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
    names = list()
    labels = list()
    all_categories = []
    
    def find_files(path):
        return glob.glob(path)
    
    # Read a file and split into lines
    def read_lines(filename):
        lines = io.open(filename, encoding='utf-8').read().strip().split('\n')
        return [unicode_to_ascii(line) for line in lines]
    
    for i, filename in enumerate(find_files('data/names/*.txt')):
        category = os.path.splitext(os.path.basename(filename))[0]
        all_categories.append(category)
        
        lines = read_lines(filename)
        names += lines
        labels += [i for _ in lines]
        
    return names, labels, all_categories


# Find letter index from all_letters, e.g. "a" = 0
def letter_to_index(letter):
    return ALL_LETTERS.find(letter)
# Find letter index from all_letters, e.g. "a" = 0
def letter_to_index(letter):
    return ALL_LETTERS.find(letter)
def name_to_list(name):
    return [letter_to_index(letter) for letter in name]

def preprocessing(data, masked_value = 2):
    "name to list of int to one-hot to padding and mask"
    listint = [name_to_list(name) for name in data]
    onehot = list()
    for name in listint:
        onehot.append(tf.one_hot(name, N_LETTERS))
    padded = keras.utils.pad_sequences(onehot, padding="post", value= masked_value)
    return padded


def train_test_split(data, labels, testprop:float=0.2):
    indices = tf.random.shuffle(range(len(data)))
    shdata = tf.gather(data, indices)
    shlabels = tf.gather(labels, indices)

    threshold = int((1-testprop) * len(data))
    traindata = shdata[:threshold]
    trainlabels = shlabels[:threshold]
    testdata = shdata[threshold:]
    testlabels = shlabels[threshold:]

    return traindata, trainlabels, testdata, testlabels