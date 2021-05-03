# Grammar & Datasets
# TODO: random sequence generation, filter ungrammatical and create arbitrary testset of ungrammatical seqs
# TODO: unconstraint max length
# TODO: grammaticality loss fix

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


def shiftStimuli( ggen, seqs ):
    """
    Creates shifted grammar
    """
    new_vocab_size = 2 * len( ggen ) - 3 # PADTOKEN;STARTTOKEN;ENDTOKEN;

    vocab_without_TOKENS = len( ggen ) - 3
    shifted_seqs = []
    for _, seq in seqs:
        shifted_seqs.append( (1, [ stim + vocab_without_TOKENS for stim in seq if stim not in [ PAD_TOKEN, START_TOKEN, END_TOKEN ] ] ) )

    return shifted_seqs, new_vocab_size


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

def get_dl(bs, sequences, shuffle=True):
    train_ds = SequenceDataset( sequences )
    return DataLoader( train_ds, batch_size=bs, shuffle=shuffle, collate_fn=collate_batch )


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


def g1():
    return {
        'START': ['#1'],
        '#1': ['W1','N1'],
        'W1': ['S1','S2','W2'],
        'N1': ['P1','P2','N2'],
        'S1': ['S2',],
        'S2': ['W2',],
        'W2': ['S3','Z'],
        'P1': ['P2',],
        'P2': ['N2',],
        'N2': ['P3','Z'],
        'S3': ['P1','P2','N2'],
        'Z': ['#2',],
        'P3': ['S3','Z'],
        '#2': ['END']
    }


def g1_train():
    return [
        ( 1, ['#','W','S','W','Z','#'], ),
        ( 1, ['#','W','S','S','W','Z','#'], ),
        ( 1, ['#','W','S','W','S','N','P','Z','#'], ),
        ( 1, ['#','W','S','W','S','P','N','Z','#'], ),
        ( 1, ['#','W','W','S','P','N','P','Z','#'], ),
        ( 1, ['#','W','W','S','N','P','S','N','Z','#'], ),
        ( 1, ['#','W','W','S','P','N','Z','#'], ),
        ( 1, ['#','W','W','S','N','Z','#'], ),
        ( 1, ['#','W','S','S','W','S','N','Z','#'], ),
        ( 1, ['#','W','S','W','S','N','Z','#'], ),
        ( 1, ['#','N','P','P','N','Z','#'], ),
        ( 1, ['#','N','N','Z','#'], ),
        ( 1, ['#','N','P','P','N','P','Z','#'], ),
        ( 1, ['#','N','N','P','S','P','N','P','Z','#'], ),
        ( 1, ['#','N','P','N','P','S','N','P','Z','#'], ),
        ( 1, ['#','N','N','P','S','P','P','N','Z','#'], ),
        ( 1, ['#','N','P','P','N','P','S','N','Z','#'], ),
        ( 1, ['#','N','N','P','S','N','Z','#'], ),
    ]


def g1_test_gr():
    return [
        ( 1, ['#','W','W','Z','#'], ),
        ( 1, ['#','W','W','S','N','P','Z','#'], ),
        ( 1, ['#','W','S','S','W','S','N','P','Z','#'], ),
        ( 1, ['#','W','W','S','P','P','N','P','Z','#'], ),
        ( 1, ['#','W','W','S','P','N','Z','#'], ),
        ( 1, ['#','W','S','W','S','N','Z','#'], ),
        ( 1, ['#','W','S','W','S','N','P','Z','#'], ),
        ( 1, ['#','W','S','S','W','S','P','N','Z','#'], ),
        ( 1, ['#','W','W','S','P','P','N','Z','#'], ),
        ( 1, ['#','W','S','W','S','P','P','N','Z','#'], ),
        ( 1, ['#','N','P','N','Z','#'], ),
        ( 1, ['#','N','N','P','S','N','P','Z','#'], ),
        ( 1, ['#','N','N','P','Z','#'], ),
        ( 1, ['#','N','P','N','P','S','N','Z','#'], ),
        ( 1, ['#','N','N','P','S','P','N','Z','#'], ),
        ( 1, ['#','N','P','N','P','S','P','N','Z','#'], ),
        ( 1, ['#','W','W','S','P','P','N','P','Z','#'], ),
    ]


def g1_test_ugr():
    return [
        ( 0, ['#','W','Z','W','Z','#'], ),
        ( 0, ['#','W','S','W','N','P','Z','#'], ),
        ( 0, ['#','W','W','S','N','W','Z','#'], ),
        ( 0, ['#','W','S','W','N','P','S','N','Z','#'], ),
        ( 0, ['#','N','Z','P','P','N','Z','#'], ),
        ( 0, ['#','N','N','S','P','N','P','Z','#'], ),
        ( 0, ['#','N','S','P','P','N','Z','#'], ),
        ( 0, ['#','N','N','W','S','P','P','N','Z','#'], ),
        ( 0, ['#','W','S','Z','#'], ),
        ( 0, ['#','W','W','S','P','W','S','N','Z','#'], ),
        ( 0, ['#','W','S','W','P','P','N','Z','#'], ),
        ( 0, ['#','W','S','W','S','Z','#'], ),
        ( 0, ['#','W','N','P','S','P','P','N','Z','#'], ),
        ( 0, ['#','N','P','N','S','P','N','Z','#'], ),
        ( 0, ['#','N','N','Z','S','P','S','N','Z','#'], ),
        ( 0, ['#','N','W','S','W','Z','#'], ),
        ( 0, ['#','N','P','N','W','Z','#'], ), #NPP, end, NPL start
        ( 0, ['#','W','S','S','N','Z','#'], ),
        ( 0, ['#','W','S','P','S','P','N','Z','#'], ),
        ( 0, ['#','W','W','S','P','P','Z','#'], ),
        ( 0, ['#','W','S','S','N','P','N','P','Z','#'], ),
        ( 0, ['#','N','P','P','S','N','Z','#'], ),
        ( 0, ['#','N','P','S','S','P','S','N','Z','#'], ),
        ( 0, ['#','N','N','P','S','N','N','P','Z','#'], ),
        ( 0, ['#','N','P','N','P','N','Z','#'], ),
        ( 0, ['#','N','P','P','Z','#'], ),
        ( 0, ['#','N','P','S','P','Z','#'], ),
        ( 0, ['#','N','P','N','P','S','S','W','Z','#'], ),
        ( 0, ['#','N','P','S','P','P','N','Z','#'], ),
        ( 0, ['#','W','W','S','N','N','Z','#'], ),
        ( 0, ['#','W','S','W','S','S','N','Z','#'], ),
        ( 0, ['#','W','S','N','Z','#'], ),
        ( 0, ['#','W','S','P','P','N','Z','#'], ),
        ( 0, ['#','W','S','P','N','P','P','N','Z','#'], ),
    ]

def g2():
    return {
        'START': ['#1'],
        '#1': ['Z1','N1'],
        'Z1': ['Z2','W1'],
        'N1': ['N2','P2'],
        'Z2': ['S2','S3','W2'],
        'W1': ['N2',],
        'N2': ['P1','P3','S1'],
        'P2': ['S2','S3','W2'],
        'S2': ['S3',],
        'S3': ['W2',],
        'W2': ['#2',],
        'P1': ['P3',],
        'P3': ['S1',],
        'S1': ['Z2',],
        '#2': ['END',]
    }


def g2_train():
    return [
        ( 1, ['#','Z','Z','S','W','#'], ),
        ( 1, ['#','Z','Z','S','S','W','#'], ),
        ( 1, ['#','Z','W','P','W','#'], ),
        ( 1, ['#','Z','W','N','S','W','P','W','#'], ),
        ( 1, ['#','Z','W','P','S','S','W','#'], ),
        ( 1, ['#','Z','W','N','S','Z','S','S','W','#'], ),
        ( 1, ['#','Z','W','N','P','S','Z','S','W','#'], ),
        ( 1, ['#','Z','W','N','P','S','Z','W','#'], ),
        ( 1, ['#','N','P','S','S','W','#'], ),
        ( 1, ['#','N','P','W','#'], ),
        ( 1, ['#','N','N','P','S','Z','S','S','W','#'], ),
        ( 1, ['#','N','N','S','Z','W','#'], ),
        ( 1, ['#','N','N','P','S','Z','S','W','#'], ),
        ( 1, ['#','N','N','P','P','S','Z','W','#'], ),
        ( 1, ['#','N','N','P','P','S','W','P','W','#'], ),
        ( 1, ['#','N','N','S','W','P','W','#'], ),
        ( 1, ['#','N','N','S','W','P','S','W','#'], ),
        ( 1, ['#','N','N','S','W','P','S','S','W','#'], ),
    ]
