# Main file

import torch, torchvision
from torch.utils.data.dataset import Dataset
import numpy as np
from grammar import GrammarGen

from torch.utils.data import DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SequenceDataset( Dataset ):
    """
    Dataset for Sequences
    """

    def __init__( self, size, grammar=None ):
        """
        https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
        Args:
            size (int): amount of sequences generated
            grammar (dict): dictionary specifying the grammar
        """
        self.Grammar = GrammarGen( grammar )
        self.seqs = self.Grammar.generate( size )

    def __len__(self):
        return len( self.seqs )

    def __getitem__(self, idx):
        return ( 1, self.seqs[idx], )

def collate_batch( batch ):
    """
    https://pytorch.org/tutorials/beginner/text_sentiment_ngrams_tutorial.html
    Basically:
    1. create tensor for all labels in a batch
    2. smash all sequences together in one big tensor
    3. remember where which sequence starts in offsets
    """
    label_list, seq_list, offsets = [], [], [0]
    for (_label, _seq) in batch:
        label_list.append( _label )
        processed_seq = torch.tensor( _seq, dtype=torch.int16 )
        seq_list.append( processed_seq )
        offsets.append( processed_seq.size(0) )
    label_list = torch.tensor( label_list, dtype=torch.int16 )
    offsets = torch.tensor( offsets[:-1] ).cumsum( dim=0 )
    seq_list = torch.cat( seq_list )
    return label_list.to( device ), seq_list.to( device ), offsets.to( device )

def main():
    dataset = SequenceDataset( 21 )

    dataloader = DataLoader( dataset, batch_size=3, shuffle=False, collate_fn=collate_batch )

    for (a,b,c) in dataloader:
        print( a, b, c )


if __name__ == '__main__':
    main()