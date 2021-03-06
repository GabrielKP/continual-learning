# Grammar & Datasets
# TODO: random sequence generation, filter ungrammatical and create arbitrary testset of ungrammatical seqs
# TODO: unconstraint max length
# TODO: grammaticality loss fix

import itertools
import random
import torch
import copy

from torch import nn
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader

PAD_TOKEN = 0   # ugly but works for now
START_TOKEN = 1
END_TOKEN = 2

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class GrammarGen():
    """
    Generates Grammar sequences from grammars
    Grammars are dictionaries:
    - always have START
    - all paths lead eventually to END
    - Entries starting with the same letter
      have same output
    """

    def __init__(self, grammar=None):
        if grammar is None:
            self.grammar = {
                'START': ['A'],
                'A': ['D', 'C1'],
                'C1': ['G1', 'F'],
                'G1': ['F'],
                'D': ['C1'],
                'F': ['C2', 'END'],
                'C2': ['END', 'G2'],
                'G2': ['END']
            }
        else:
            self.grammar = grammar
        self._initOutput()

    def _initOutput(self):
        """
        Creates Mapping from Stimulus to Output
        """
        stimulusToNumber = {'END': END_TOKEN,
                            'START': START_TOKEN, 'PAD': PAD_TOKEN}
        cores = {'END': END_TOKEN, 'START': START_TOKEN,
                 'PAD': PAD_TOKEN}  # keep track of same output letters
        i = 3
        for stimulus in self.grammar:
            if stimulus == 'START':
                continue
            core = stimulus[0]
            if core not in cores:
                cores[core] = i
                i += 1
            stimulusToNumber[stimulus] = cores[core]

        # Mapping Stimulus to Number (C1 -> 4, C2 -> 4, A -> 3)
        self.stim2nbr = stimulusToNumber
        # Mapping Core-Stimulus to Number (C -> 4, A -> 3)
        self.cores = cores
        # Convert Grammar to stimulus Numbers
        self.number_grammar = dict()

        # rules in grammar are in form stimA -> simsB with stimsB being a list
        for stimA, stimsB in self.grammar.items():

            converted_stim = self.stim2nbr[stimA]

            # Handle case where stimulus can be present twice or more in grammar
            if converted_stim in self.number_grammar:
                # Merge different possibiliteis for stimsB
                self.number_grammar[converted_stim].extend(
                    [self.stim2nbr[stim] for stim in stimsB])
                # Make sure nothing is double
                self.number_grammar[converted_stim] = list(
                    set(self.number_grammar[converted_stim]))
                continue

            self.number_grammar[converted_stim] = [
                self.stim2nbr[stim] for stim in stimsB]

        self.out2stim = {v: k for k, v in self.cores.items()}
        # variable to check how many sequences have been generated for the grammaticality test
        self.grammCheckMaxLen = -1

    def __len__(self):
        return len(self.cores)

    def stim2seqs(self, stimuliSequences):
        seqs = []
        for stimulusSequence in stimuliSequences:
            seqs.append([self.cores[stimulus]
                        for stimulus in stimulusSequence])
        return seqs

    def generate(self, n):
        ret = []
        count = 0
        hashtrack = set()
        while count < n:
            token = []
            current = 'START'
            while current != 'END':
                # Append current
                if current != 'START':
                    token.append(self.stim2nbr[current])
                # Choose next
                r = random.randint(0, len(self.grammar[current]) - 1)
                current = self.grammar[current][r]
            # Check if seq is already inside
            tokenhash = ''.join([str(x) for x in token])
            if tokenhash not in hashtrack:
                hashtrack.add(tokenhash)
                ret.append((token, ))
                count += 1

        return ret

    def seqs2stim(self, seqs):
        return [[self.out2stim[nbr] for nbr in seq] for seq in seqs]

    def generateUngrammatical(self, n):
        pass

    def transitionProbabilities(self):
        pass

    def generateAllGrammatical(self, maxlen=float('inf')):
        def genAllHelp(seq, current):
            if current == 'END':
                return [seq]
            if len(seq) >= maxlen:
                return []
            # Append Current
            if current != 'START':
                seq.append(self.stim2nbr[current])
            # Generate next possibilities
            options = range(len(self.grammar[current]))
            ret = [(genAllHelp(copy.copy(seq), self.grammar[current][i]))
                   for i in options]
            return itertools.chain(*ret)
        return set([tuple(seq) for seq in genAllHelp([], 'START')])

    def isGrammatical(self, seqs):
        """Check for grammaticality for a list of sequences"""
        maxlen = max([len(seq) for seq in seqs])
        if self.grammCheckMaxLen < maxlen:
            self.allGrammatical = self.generateAllGrammatical(maxlen)
            self.grammCheckMaxLen = maxlen

        return [tuple(seq) in self.allGrammatical for seq in seqs]


def shiftStimuli(ggen, seqs):
    """
    Creates shifted grammar
    """
    new_vocab_size = 2 * len(ggen) - 3  # PADTOKEN;STARTTOKEN;ENDTOKEN;

    vocab_without_TOKENS = len(ggen) - 3
    shifted_seqs = []
    for seq in seqs:
        shifted_seqs.append(
            ([stim + vocab_without_TOKENS for stim in seq if stim not in [PAD_TOKEN, START_TOKEN, END_TOKEN]]))

    return shifted_seqs, new_vocab_size


class SequenceDataset(Dataset):
    """
    Dataset for Sequences
    """

    def __init__(self, seqs):
        """
        https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
        Args:
            size (int): amount of sequences generated
            grammar (dict): dictionary specifying the grammar
        """
        self.seqs = seqs

    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        return self.seqs[idx]


def collate_batch(batch):
    """
    https://pytorch.org/tutorials/beginner/text_sentiment_ngrams_tutorial.html
    Basically:
    1. create tensor for all labels in a batch
    2. smash all sequences together into a list
    """
    label_list, seq_list = [], []
    for (_seq) in batch:
        label_list.append(-1)
        _seq = [1] + _seq + [2]
        processed_seq = torch.tensor(_seq, dtype=torch.int32)
        seq_list.append(processed_seq)
    label_list = torch.tensor(label_list, dtype=torch.float)
    return label_list.to(device), seq_list


def get_dl(bs, sequences, shuffle=True):
    train_ds = SequenceDataset(sequences)
    return DataLoader(train_ds, batch_size=bs, shuffle=shuffle, collate_fn=collate_batch)


def get_data(train_ds, valid_ds, bs):
    return (
        DataLoader(train_ds, batch_size=bs, shuffle=True,
                   collate_fn=collate_batch),
        DataLoader(valid_ds, batch_size=bs * 2, collate_fn=collate_batch),
    )
