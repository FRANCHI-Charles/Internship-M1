from automata import *
import numpy as np
from time import time

SEED = 689
MEANLEN = 10
SIZE = 10000

def principal(even:bool=False, sink:bool=False, fix:bool=False, unique:bool=False):
    output = tuple()
    if even:
        t0 = time()
        print("Generate dataset 'even'...")
        even_transitions = np.array([[1,0],
                                    [0,1]])
        even_finites = np.array([1,0])

        output += (DFA(even_transitions, even_finites, probas="equal"),)
        output[-1].dataset(SIZE, meanlen=MEANLEN, random_state=SEED)
        print(f"Done! in {time() - t0:.4f} sec")

    if sink:
        t0 = time()
        print("Generate dataset 'sink'...")
        sink_transitions = np.array([[-1,1,-1],
                                    [-1,2,-1],
                                    [3,-1,2],
                                    [-1,-1,-1],
                                    [-1,-1,-1]])
        sink_finites = np.array([0,0,0,1,0])

        sink_probas = np.array([[0,1,0],
                                [0,1,0],
                                [1/(MEANLEN-2),0,(MEANLEN-3)/(MEANLEN -2)],
                                [1/3,1/3,1/3],
                                [1/3,1/3,1/3]])

        output += (DFA(sink_transitions, sink_finites, probas=sink_probas),)
        output[-1].dataset(SIZE, meanlen=MEANLEN, random_state=SEED)
        print(f"Done! in {time() - t0:.4f} sec")

    if fix:
        t0 = time()
        print("Generate 'fix'...")
        fix_transitions = np.array([[1,1,1,1,1],
                                    [2,2,2,2,2],
                                    [3,3,3,3,3],
                                    [4,4,4,4,4],
                                    [5,5,5,5,5],
                                    [5,5,5,5,5]])
        fix_finites = np.array([0,0,0,0,1,0])

        output += (DFA(fix_transitions, fix_finites, probas = "equal"),)
        output[-1].dataset(SIZE, meanlen=4, random_state=SEED)
        print(f"Done! in {time() - t0:.4f} sec")

    if unique:
        t0 = time()
        print("Generate 'unique'...")
        unique_transitions = np.array([[1,2,3,4],
                                    [1,-1,-1,-1],
                                    [-1,2,-1,-1],
                                    [-1,-1,3,-1],
                                    [-1,-1,-1,4],
                                    [-1,-1,-1,-1]])
        unique_finites = np.array([1,1,1,1,1,0])

        output += (DFA(unique_transitions, unique_finites),)
        output[-1].dataset(SIZE, meanlen=MEANLEN, random_state=SEED)
        print(f"Done! in {time() - t0:.4f} sec")

    return output
