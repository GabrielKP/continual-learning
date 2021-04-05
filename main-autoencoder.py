# Main file

import torch

import numpy as np
from grammar import GrammarGen, SequenceDataset, get_data, get_trainstimuliSequence, get_teststimuliSequence

from torch import nn
from torch import optim

device = torch.device( 'cuda' if torch.cuda.is_available() else 'cpu' )


class Encoder(nn.Module):

    def __init__(self, input_dim, hidden_dim, embedding=True ):
        super(Encoder, self).__init__()

        # Vars
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.embedding_dim = input_dim

        # Layers
        self.embed = nn.Embedding( self.input_dim, self.embedding_dim )
        if not embedding:
            self.embed.weight.data = torch.eye( input_dim )
        self.lstm = nn.LSTM( self.embedding_dim, self.hidden_dim, batch_first=True )

        # Init Params
        # for param in self.lstm.parameters():
        #     nn.init.zeros_( param )

        # nn.init.zeros_( self.lin.weight )

    def forward(self, seqs):
        lengths = [ len( seq ) for seq in seqs ]
        padded_seqs = nn.utils.rnn.pad_sequence( seqs, batch_first=True )
        padded_embeds = self.embed( padded_seqs )
        padded_embeds_packed = nn.utils.rnn.pack_padded_sequence( padded_embeds, lengths, batch_first=True, enforce_sorted=False )
        _, (hidden, cell) = self.lstm( padded_embeds_packed )
        return  hidden, cell


def loss_batch(model, loss_func, labels, seqs, opt=None):
    labels = labels.unsqueeze(1)
    loss = loss_func( model( seqs ), labels )

    if opt is not None:
        loss.backward()
        opt.step()
        opt.zero_grad()

    return loss.item(), len( labels )


def fit(epochs, model, loss_func, opt, train_dl, valid_dl):
    for epoch in range(epochs):
        model.train()
        for labels, seqs in train_dl:
            loss_batch(model, loss_func, labels, seqs, opt)

        model.eval()
        with torch.no_grad():
            losses, nums = zip(
                *[loss_batch( model, loss_func, labels, seqs ) for labels, seqs in valid_dl]
            )
            val_loss = np.sum( np.multiply( losses, nums ) ) / np.sum( nums )

        print( epoch, val_loss )


def get_model(input_dim, hidden_dim, lr):
    model = Encoder( input_dim, hidden_dim )
    return model, optim.SGD( model.parameters(), lr=lr )


def main():
    bs = 3
    ggen = GrammarGen()
    seqs = ggen.stim2seqs( get_trainstimuliSequence() )
    teststimuliSequence = get_teststimuliSequence()
    test_ds = SequenceDataset( ggen.stim2seqs( teststimuliSequence ) )
    train_ds = SequenceDataset( seqs )
    train_dl, test_dl = get_data( train_ds, test_ds, bs )

    lr = 3
    hidden_dim = 5
    input_dim = len( ggen ) + 1

    model, opt = get_model( input_dim, hidden_dim, lr )

    loss_func = nn.BCELoss()

    epochs = 30
    fit( epochs, model, loss_func, opt, train_dl, test_dl )

if __name__ == '__main__':
    main()
