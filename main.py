# Main file

import torch, torchvision
from torch.utils import data
from torch.utils.data.dataset import Dataset
import numpy as np
from grammar import GrammarGen

from torch import nn
from torch import optim
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
        self.grammar = GrammarGen( grammar )
        self.seqs = self.grammar.generate( size )

    def __len__(self):
        return len( self.seqs )

    def __getitem__(self, idx):
        return ( 1, self.seqs[idx], )


class SequenceClassificationModel(nn.Module):

    def __init__(self, stimuli_dim, hidden_dim, emb_dim):
        super(SequenceClassificationModel, self).__init__()
        self.emb = nn.EmbeddingBag(stimuli_dim, emb_dim)
        #self.lstm = nn.LSTM( emb_dim, hidden_dim )
        self.lin = nn.Linear( emb_dim, 1 )
        self.act = nn.Sigmoid()

    def forward(self, seqs, offsets):
        embedded = self.emb( seqs, offsets )
        return  self.act(self.lin( embedded ) )


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
        processed_seq = torch.tensor( _seq, dtype=torch.int32 )
        seq_list.append( processed_seq )
        offsets.append( processed_seq.size(0) )
    label_list = torch.tensor( label_list, dtype=torch.float )
    offsets = torch.tensor( torch.tensor( offsets[:-1] ).cumsum( dim=0 ), dtype=torch.int32 )
    seq_list = torch.cat( seq_list )
    return label_list.to( device ), seq_list.to( device ), offsets.to( device )


def loss_batch(model, loss_func, labels, seqs, offsets, opt=None):
    labels = labels.unsqueeze(1)
    loss = loss_func(model(seqs, offsets), labels)

    if opt is not None:
        loss.backward()
        opt.step()
        opt.zero_grad()

    return loss.item(), len(labels)


def fit(epochs, model, loss_func, opt, train_dl):
    for epoch in range(epochs):
        model.train()
        for labels, seqs, offsets in train_dl:
            loss_batch(model, loss_func, labels, seqs, offsets, opt)
        print(epoch)

def get_model(stimuli_dim, hidden_dim, embedding_dim, lr):
    model = SequenceClassificationModel( stimuli_dim, hidden_dim, embedding_dim )
    return model, optim.SGD(model.parameters(), lr=lr)

def get_data(train_ds, valid_ds, bs):
    return (
        DataLoader( train_ds, batch_size=bs, shuffle=True, collate_fn=collate_batch ),
        DataLoader( valid_ds, batch_size=bs * 2 ),
    )


def main():
    bs = 3
    train_ds, valid_ds = SequenceDataset( 60 ), SequenceDataset( 12 )
    train_dl, valid_dl = get_data( train_ds, valid_ds, bs )

    lr = 0.1
    emsize = 5
    stimuli_dim = len( train_ds.grammar )

    model, opt = get_model( stimuli_dim, 0, emsize, lr )

    loss_func = nn.BCELoss()

    for labels, seqs, offsets in dataloader:
        with torch.no_grad():
            print( labels, seqs, offsets )
            print( model(seqs, offsets) )
            print( loss_batch(model, loss_func, labels, seqs, offsets) )
        break

    epochs = 3
    fit(epochs, model, loss_func, opt, dataloader)

    for labels, seqs, offsets in dataloader:
        with torch.no_grad():
            print( labels, seqs, offsets )
            print( model(seqs, offsets) )
            print( loss_batch(model, loss_func, labels, seqs, offsets) )
        break


if __name__ == '__main__':
    main()
