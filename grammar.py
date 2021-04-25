# Grammar & Datasets
# TODO: random sequence generation, filter ungrammatical and create arbitrary testset of ungrammatical seqs
# TODO: unconstraint max length
# TODO: grammaticality loss fix
# TODO: run training without validation set

import random
import torch

from torch import nn
from torch.utils.data.dataset import Dataset
from torch.utils.data import DataLoader

PAD_TOKEN = 0   # ugly but works for now
START_TOKEN = 1
END_TOKEN = 2

device = torch.device( 'cuda' if torch.cuda.is_available() else 'cpu' )

class GrammarGen():
    """
    Generates Grammar sequences from grammars
    Grammars are dictionaries:
    - always have START
    - all paths lead eventually to END
    - Entries starting with the same letter
      have same output
    """

    def __init__(self, grammar=None ):
        if grammar is None:
            self.grammar = {
                'START': ['A'],
                'A': ['D','C1'],
                'C1': ['G1','F'],
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
        stimulusToOutput = { 'END': END_TOKEN, 'START': START_TOKEN, 'PAD': PAD_TOKEN }
        cores = { 'END': END_TOKEN, 'START': START_TOKEN,'PAD': PAD_TOKEN } # keep track of same output letters
        i = 3
        for stimulus in self.grammar:
            if stimulus == 'START':
                continue
            core = stimulus[0]
            if core not in cores:
                cores[core] = i
                i += 1
            stimulusToOutput[stimulus] = cores[core]

        # Mapping Stimulus to Number (C1 -> 4, C2 -> 4, A -> 3)
        self.stim2out = stimulusToOutput
        # Mapping Core-Stimulus to Number (C -> 4, A -> 3)
        self.cores = cores
        # Convert Grammar to stimulus Numbers
        self.number_grammar = dict()

        # rules in grammar are in form stimA -> simsB with stimsB being a list
        for stimA, stimsB in self.grammar.items():

            converted_stim = self.stim2out[stimA]

            # Handle case where stimulus can be present twice or more in grammar
            if converted_stim in self.number_grammar:
                # Merge different possibiliteis for stimsB
                self.number_grammar[converted_stim].extend( [ self.stim2out[stim] for stim in stimsB ] )
                # Make sure nothing is double
                self.number_grammar[converted_stim] = list( set( self.number_grammar[converted_stim] ) )
                continue

            self.number_grammar[converted_stim] = [ self.stim2out[stim] for stim in stimsB ]


    def __len__(self):
        return len(self.cores)

    def stim2seqs(self, stimuliSequences):
        seqs = []
        for label, stimulusSequence in stimuliSequences:
            seqs.append( ( label, [ self.cores[stimulus] for stimulus in stimulusSequence ] ) )
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
                    token.append( self.stim2out[current] )
                # Choose next
                r = random.randint( 0, len( self.grammar[current] ) - 1 )
                current = self.grammar[current][r]
            # Check if seq is already inside
            tokenhash = ''.join( [ str(x) for x in token ] )
            if tokenhash not in hashtrack:
                hashtrack.add( tokenhash )
                ret.append( ( 1, token, ) )
                count += 1

        return ret

    def seqs2stim(self, seqs):
        o2r = { v: k for k, v in self.cores.items() }
        return [ [ o2r[stim] for stim in seq ] for _,seq in seqs ]

    def generateUngrammatical(self, n):
        pass

    def transitionProbabilities(self):
        pass


def createShiftedGrammar():
    """
    Creates shifted grammar
    """
    pass


class SequenceDataset( Dataset ):
    """
    Dataset for Sequences
    """

    def __init__( self, seqs ):
        """
        https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
        Args:
            size (int): amount of sequences generated
            grammar (dict): dictionary specifying the grammar
        """
        self.seqs = seqs

    def __len__(self):
        return len( self.seqs )

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
    for (_label, _seq) in batch:
        label_list.append( _label )
        _seq = [1] + _seq + [2]
        processed_seq = torch.tensor( _seq, dtype=torch.int32 )
        seq_list.append( processed_seq )
    label_list = torch.tensor( label_list, dtype=torch.float )
    return label_list.to( device ), seq_list

# def collate_batch(batch):
#     """
#     https://pytorch.org/tutorials/beginner/text_sentiment_ngrams_tutorial.html
#     Basically:
#     1. create tensor for all labels in a batch
#     2. Add start (0) and end token (1) to each sequennce
#     3. Mash sequences together into a list
#     4. Pad smaller sequences with 0
#     """
#     label_list, seq_list = [], []
#     for (_label, _seq) in batch:
#         label_list.append( _label )
#         _seq = [1] + _seq + [2]
#         processed_seq = torch.tensor( _seq, dtype=torch.int32 )
#         seq_list.append( processed_seq )
#     label_list = torch.tensor( label_list, dtype=torch.float )
#     seq_list = nn.utils.rnn.pad_sequence( seq_list, batch_first=True )
#     return label_list.to( device ), seq_list


def get_data(train_ds, valid_ds, bs):
    return (
        DataLoader( train_ds, batch_size=bs, shuffle=True, collate_fn=collate_batch ),
        DataLoader( valid_ds, batch_size=bs * 2, collate_fn=collate_batch ),
    )


def get_trainstimuliSequence():
    return [
        ( 1, ['A','C','F'], ),
        ( 1, ['A','C','F','C','G'], ),
        ( 1, ['A','C','G','F'], ),
        ( 1, ['A','C','G','F','C','G'], ),
        ( 1, ['A','D','C','F'], ),
        ( 1, ['A','D','C','F','C'], ),
        ( 1, ['A','D','C','F','C','G'], ),
        ( 1, ['A','D','C','G','F','C','G'], ),
    ]


def get_trainstimuliExtendedSequence():
    return [
        ( 1, ['A','C','F'], ),
        ( 1, ['A','C','F','C','G'], ),
        ( 1, ['A','C','G','F'], ),
        ( 1, ['A','C','G','F','C','G'], ),
        ( 1, ['A','D','C','F'], ),
        ( 1, ['A','D','C','F','C'], ),
        ( 1, ['A','D','C','F','C','G'], ),
        ( 1, ['A','D','C','G','F','C','G'], ),
        ( 0, ['A','C','F','G'] ),
        ( 0, ['A','D','G','F','C'] ),
        ( 0, ['A','C','C','G'] ),
        ( 0, ['A','G','F','C','G'] ),
        ( 0, ['A','D','C','F','G'] ),
        ( 0, ['A','D','C','G','F','G','C'], ),
    ]


def get_correctStimuliSequence():
    return [
        ( 1, ['A','C','F','C','G'], ),
        ( 1, ['A','D','C','F','C'], ),
        ( 1, ['A','C','G','F','C'], ),
        ( 1, ['A','D','C','G','F'], ),
    ]


def get_incorrectStimuliSequence():
    return [
        ( 0, ['A','D','C','F','G'], ),
        ( 0, ['A','D','F','C','G'], ),
        ( 0, ['A','D','G','C','F'], ),
        ( 0, ['A','D','G','F','C'], ),
        ( 0, ['A','G','C','F','G'], ),
        ( 0, ['A','G','F','G','C'], ),
        ( 0, ['A','G','D','C','F'], ),
        ( 0, ['A','G','F','D','C'], ),
    ]


def get_teststimuliSequence():
    return [
        ( 1, ['A','C','F','C','G'], ),
        ( 1, ['A','D','C','F','G'], ),
        ( 1, ['A','C','G','F','C'], ),
        ( 1, ['A','D','C','G','F'], ),
        ( 0, ['A','D','C','F','G'], ),
        ( 0, ['A','D','F','C','G'], ),
        ( 0, ['A','D','G','C','F'], ),
        ( 0, ['A','D','G','F','C'], ),
        ( 0, ['A','G','C','F','G'], ),
        ( 0, ['A','G','F','G','C'], ),
        ( 0, ['A','G','D','C','F'], ),
        ( 0, ['A','G','F','D','C'], ),
    ]
