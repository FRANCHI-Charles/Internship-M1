import numpy as np
import string
from torch.utils.data import Dataset
from torch import tensor, float32
from warnings import warn


class DFA:
    """
    Class of deterministic automaton.
    
    Attributes
    ----------
    initstate : int
        State in which the automaton start.

    transition : np.ndarray
        2D matrix of transitions.
        It has as shape (Q,S), with:
                - Q, number of states
                - S, number of symbols of the automaton

    finites : np.ndarray
        A 1D binary array reprenting if the corresponding state
            is terminal (1) or not (0).

    probas : np.ndarray
        Matrix of probabilites for generation of correct
        and incorrect words. 

    letters : str
        List of the order of the symbol 

    Methods
    -------
    check,
    generate,
    dataset
    """
    
    def __init__(self, transition:np.ndarray, finites:np.ndarray, initstate:int=0, probas:np.ndarray|str=None, letters:str=None):
        """
        Parameters
        ----------
        transition : np.ndarray
            A 2D array representing the transitions of the automaton.
            It has as shape (Q,S), with:
                - Q, number of states
                - S, number of symbols of the automaton

        finites : np.ndarray
            A 1D binary array reprenting if the corresponding state
            is terminal (1) or not (0).

        initstate : int
            The initial state where the automaton starts.
            Default is 0.

        probas : np.ndarray or str, optional
            How to generate words.
            If None :
                It supposed the last state (last rows of `transition`)
                is the 'bin' state, a sink which is rejected.
                That way, from every item of the transition matrix,
                -1 means impossible symbol
            If np.ndarray :
                A 2D array with probas to generate accepted and rejected words.
                It has as shape (Q,S), like `transition`.
            If "equal" :
                All characters have as many chance to occur since each states
                (so bigger probability to have a rejected word)

        letters : str, optional
            Alphabet corresponding to the automaton.
            Default is latin alphabet.
            NOTE : if the number of symbols of the automaton is greater than 26,
                it has to be determinate. 
        """
        self.transition = transition
        self.initstate = initstate
        self.probas = probas
        self.finites = finites
        self.letters = letters

        self.data = None # last dataset generated
        
    
    @property
    def transition(self):
        return self._transition
    
    @transition.setter
    def transition(self, matrix):
        if isinstance(matrix, np.ndarray) and len(matrix.shape) == 2:
            if np.any(matrix >= matrix.shape[0]) or np.any(matrix < -1):
                raise ValueError("A value in `tranisition` is not a possible state.")
            self._transition = matrix.copy()
            self._transition[self._transition == (matrix.shape[0]-1)] = -1 # change possible extreme value as -1 for algorithm simplicity
        else:
            raise TypeError("`transisition` must be a 2D numpy array.")
        
    @property
    def finites(self):
        return self._finites
    
    @finites.setter
    def finites(self, array):
        if isinstance(array, np.ndarray) and len(array.shape) == 1:
            if array.shape[0] != self.transition.shape[0]:
                raise ValueError("The number of states of `finites` must be the same as `transition`")
            if array[-1] == 1 and np.all(self.probas[-1] == 0): # check if probas come from "None" option
                raise ValueError("Such `probas` has been set with None, the bins sink (last state) must be rejected, but got 1.")
            self._finites = array.copy()
        else:
            raise TypeError("`finites` must be a 1D numpy array.")
        
    @property
    def initstate(self):
        return self._initstate
    
    @initstate.setter
    def initstate(self, value):
        if isinstance(value, int):
            if value >= self.transition.shape[0] or value < -1:
                raise ValueError("`initstate` is not a valid state number.")
            self._initstate = value
        else:
            raise TypeError("`inistate` must be int.")
        
    @property
    def probas(self):
        return self._probas

    @probas.setter
    def probas(self, value):
        if isinstance(value, np.ndarray):
            if value.shape == self.transition.shape: # if probas is defined, just copy it
                self._probas = value.copy()
            else:
                raise ValueError("`probas` does not have the same shape as `transition`.")
        elif value == "equal": # if equal, do equal proba for the all matrix
            self._probas = np.ones(self.transition.shape) / self.transition.shape[1]
        elif value is None: # else, do equi-probabilities on element which don't lead to the bin
            if np.any(self.transition[-1] != -1):
                warn("The last row of `transition` is not a sink! Switch `probas` to 'equal'...")
                self.probas = "equal"
                return
            nobinselements = (self.transition != -1) # descriminate elements going to the sink
            for array in nobinselements:
                if np.sum(array, axis=0) == 0: # if all the elements are going to the sink, do equiprobability
                    array[:] = True
            nbrelements = np.sum(nobinselements, axis=1).reshape(-1,1)
            self._probas = nobinselements / nbrelements
        else:
            raise TypeError("`probas` has to be a 2D numpy array, 'equal' or None.")

    @property
    def letters(self):
        return self._letters
    
    @letters.setter
    def letters(self, strings):
        if isinstance(strings, str):
            if len(set(strings)) != len(strings):
                raise ValueError("The symbols in `letters` are not unique!")
            if len(strings) < self.transition.shape[1]:
                raise ValueError("Not enough symbols in `letters`.")
            self._letters = strings[:self.transition.shape[1]]
        elif strings is None:
            if len(string.ascii_lowercase) < self.transition.shape[1]:
                raise ValueError("Not enough symbols in `letters`.")
            self._letters = string.ascii_lowercase[:self.transition.shape[1]]
        else:
            raise TypeError("`letters` have to be str or None.")


        
    def _word_to_coord(self, word:str) -> list:
        return [self.letters.rfind(e) for e in word]

    def check(self, word:str) -> int:
        """
        Check if a word is accepted by the automaton or not.
        
        Parameters
        ----------
        word : str
            word to check.

        Returns
        -------
        binary (int)
            0 if rejected, 1 if accepted
        """
        wordlist = self._word_to_coord(word)
        if -1 in wordlist:
            return 0
        
        state = self._initstate
        for i in wordlist:
            state = self.transition[state,i]

        return self.finites[state]
    

    def generate(self, number:int, meanlen:int = 6, random_state:int = None) -> tuple[list[str], list[int]]:
        """
        Generate a list of accepted and rejected word.
        
        Parameters
        ----------
        number : int
            number of words to generate

        meanlen : int
            Mean length of a word. The length of words will follow
            a Poisson law with lambda = meanlen.

        random_state : int, optional
            Fix the random state for reproducibility.
            You can directly give a random Generator object from numpy.
            
        Returns
        -------
        list
            List of words generated
        list
            Corresponding binary list of accepted (1) or rejected (0)
        """
        rng = self._rng(random_state)
        words = list()
        labels = list()
        for _ in range(number):
            word, acc_rej = self._one_generation(meanlen, rng)
            words.append(word)
            labels.append(acc_rej)
        return words, labels

    def _one_generation(self, meanlen, rng):
        """The algorithm to determine the length of the word is the one find with
        the inverse transform sampling method."""
        word = ""
        p = 1
        state = self.initstate
        p *= rng.random()
        threshold = np.exp(-meanlen)
        while p > threshold:
            isymbol = rng.choice(np.arange(self.transition.shape[1]), p=self.probas[state]) # choose the next symbol in terms of the probas
            word += self.letters[isymbol] #add the symbol to the word
            state = self.transition[state, isymbol] # go next state
            p *= rng.random()
        return word, self.finites[state]
    
    def _rng(self, random_state):
        if isinstance(random_state, int):
            return np.random.default_rng(random_state)
        elif isinstance(random_state, np.random._generator.Generator):
            return random_state
        else:
            return np.random
    

    def dataset(self, length:int, prop:float = 0.5, meanlen:int = 6, max_try:int = None, equalneg:bool = True, random_state:int=None) -> tuple[list[str], list[int]]:
        """
        Generate words and labels, with a total of `length` elements
        with `prop`*`length` positive elements.
        First generate all the positive elements, then select or
        generate negative elements.

        Parameters
        ----------
        length : int
            Final size of the dataset

        prop : float between [0,1], default is 0.5
            Proportion of positive elements

        meanlen : int, default is 6
            Mean length of a word. The length of words will follow
            a Poisson law with lambda = meanlen.

        max_try : int, optional
            The maximum number of word generated to create the dataset in the good proportions.
            Default is : int(length / (1-prop)) * 10

        equalneg : bool, default is True
            If True, when generating negative elements,
            used uniform distribution for symbol selection.

        random_state : int, optional
            Fix the random state for reproducibility.
            You can directly give a random Generator object from numpy.

        Returns
        -------
        list
            List of generated words
        list
            List of words class (1 accepted, 0 rejected)
        """
        rng = self._rng(random_state)

        if prop <0 or prop > 1:
            raise ValueError("`prop` has to be a float in [0,1].") 

        if max_try is None:
            max_try = int(length/(1-prop)) * 10
        elif max_try < length:
            raise ValueError("`max_try` has to be greater than the length.")
        
        n_accepted = int(prop * length)
        n_actual_accepted = 0
        words = list()
        labels = list()
        for itry in range(max_try): # while the n_accepted is not the one required of max_try not reach
            word, label = self._one_generation(meanlen, rng)
            n_actual_accepted += label
            words.append(word)
            labels.append(label)
            if n_accepted == n_actual_accepted:
                break

        if itry == max_try-1:
            words, labels = self._zeroselect(length, words, labels, rng) # remove extrazeros
            warn(f"`max_try` limit reach. Final prop = {sum(labels)/len(labels)}")
        else:
            if len(labels) - length >= 0: # if we have generated enough elements
                words, labels = self._zeroselect(length, words, labels, rng) # remove extra zeros
            else: # else keep generating
                if equalneg: # put probas as equal probas
                    temp_probas = self.probas.copy()
                    self.probas = "equal"
                for itry2 in range(itry+1, max_try):
                    word, label = self._one_generation(meanlen, rng)
                    if label == 0: # keep only negative elements (missing ones)
                        words.append(word)
                        labels.append(label)
                    if len(labels) == length:
                        break
                if equalneg:
                    self._probas = temp_probas # reset probas

                if itry2 == max_try-1:
                    lastwords, lastlabels = self.generate(length - len(labels), meanlen, rng)
                    words += lastwords
                    labels += lastlabels
                    warn(f"`max_try` limit reach. Final prop = {sum(labels)/len(labels)}")

        self.data = (words, labels)
        return self.data
    
    def _zeroselect(self, length, words, labels, rng):
        zerolabels = np.nonzero(np.array(labels) == 0)[0] # zero indices
        to_remove = list(rng.permutation(zerolabels)[:len(labels) - length]) # select a random part

        words = list(np.delete(np.array(words), to_remove)) 
        labels = list(np.delete(np.array(labels), to_remove))

        return words, labels
    

    def word_to_matrix(self, word, length:int=None) -> list[list[int]]:
        """
        One-hot encoding the word in the automaton letters.

        Parameters
        ----------
        word : str
            Automaton word to encode.

        length : int or None
            If set, the length of the return list will be this parameters (all adding list are 0 lists).

        Returns
        -------
        list of lists of int
            One-hot encoded word.
        """
        if isinstance(word, str):
            if length is None:
                return [[1 if symbol == letter else 0 for symbol in self.letters] for letter in word]
            else:
                return [[1 if symbol == letter else 0 for symbol in self.letters] for letter in word] + [[0 for _ in self.letters] for _ in range(length - len(word))]
        else:
            raise TypeError("`word` should be a string.")



        

class TorchData(Dataset):
    """
    Transform an automaton dataset in one-hot letters torch Dataset.
    """

    def __init__(self, automaton: DFA) -> None:
        self.words, self.lengths = self._to_tensor(automaton)
        self.labels = automaton.data[1]

    def _to_tensor(self, automaton:DFA):
        datas = list()
        lengths = list()
        maxlength = self._maxlength(automaton.data[0])
        for word in automaton.data[0]:
            coded = [[1 if symbol == letter else 0 for symbol in automaton.letters] for letter in word]
            lengths.append(len(word))
            coded += [[0 for _ in automaton.letters] for _ in range(maxlength - len(coded))]
            datas.append(tensor(coded, dtype=float32))
        return datas, lengths

    def _maxlength(self, wordslist):
        maxlength = 0
        for i in wordslist:
            maxlength = max(maxlength, len(i))
        return maxlength

    
    def __getitem__(self, index) -> tuple[list,int,int]:
        return self.words[index], self.lengths[index], self.labels[index]
    
    def __len__(self) -> int:
        return len(self.labels)


if __name__ == "__main__":
    transitions1 = np.array([[2,1,0],
                            [3,1,-1],
                            [0,1,-1],
                            [-1,-1,-1]])
    finites1 = np.array([0,1,0,0])

    transitions2 = np.array([[1,0,0,0],
                            [1,2,0,0],
                            [0,0,0,0]])
    finites2 = np.array([0,0,1])

    transitions3 = np.array([[1,2],
                             [2,0],
                             [2,2]])
    finites3 = np.array([1,0,0])
    
    
    automat = DFA(transitions1, finites1)
    print(automat.transition)
    print(automat.initstate)
    print(automat.finites)
    print(automat.probas)
    print(automat.letters)
    print(type(automat.probas[0,0]))

    print(automat.check("abab"), automat.check("aba"))

    rng = np.random.default_rng(42)

    #words, labels = automat.generate(20, random_state=rng)
    words, labels = automat.dataset(500, prop = 0.5, random_state=8)
    print(len(words), len(labels), sum(labels))
    print(type(words[0]), type(labels[0]))

    datatorch = TorchData(automat)
    size = len(datatorch[0][0])
    print(datatorch[1])
    for a,_,_ in datatorch:
        assert len(a) == size