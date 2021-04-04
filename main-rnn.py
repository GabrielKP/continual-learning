# Main file

import torch, torchvision
from torch.utils import data
from torch.utils.data.dataset import Dataset
import numpy as np
from grammar import GrammarGen

from torch import nn
from torch import optim
from torch.utils.data import DataLoader

device = torch.device( 'cuda' if torch.cuda.is_available() else 'cpu' )


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


class SequenceClassificationModel(nn.Module):

    def __init__(self, input_dim, hidden_dim, embedding_dim):
        super(SequenceClassificationModel, self).__init__()
        self.embed = nn.Embedding( input_dim, embedding_dim )
        self.lstm = nn.LSTM( embedding_dim, hidden_dim, batch_first=True )
        self.lin = nn.Linear( hidden_dim, 1 )
        self.act = nn.Sigmoid()

        # for param in self.lstm.parameters():
        #     nn.init.zeros_( param )

        # nn.init.zeros_( self.lin.weight )

    def forward(self, seqs):
        lengths = [ len( seq ) for seq in seqs ]
        padded_seqs = nn.utils.rnn.pad_sequence( seqs, batch_first=True )
        padded_embeds = self.embed( padded_seqs )
        padded_embeds_packed = nn.utils.rnn.pack_padded_sequence( padded_embeds, lengths, batch_first=True, enforce_sorted=False )
        _, (hidden, _) = self.lstm( padded_embeds_packed )
        return  self.act( self.lin( hidden[-1] ) )


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
        processed_seq = torch.tensor( _seq, dtype=torch.int32 )
        seq_list.append( processed_seq )
    label_list = torch.tensor( label_list, dtype=torch.float )
    return label_list.to( device ), seq_list


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


def get_model(input_dim, hidden_dim, embedding_dim, lr):
    model = SequenceClassificationModel( input_dim, hidden_dim, embedding_dim )
    return model, optim.SGD( model.parameters(), lr=lr )


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

def main():
    bs = 3
    ggen = GrammarGen()
    seqs = ggen.stim2seqs( get_trainstimuliExtendedSequence() )
    train_ds = SequenceDataset( seqs )
    valid_ds = SequenceDataset( seqs )
    train_dl, valid_dl = get_data( train_ds, valid_ds, bs )

    lr = 3
    hidden_dim = 5
    input_dim = len( ggen ) + 1
    embedding_dim = 4

    model, opt = get_model( input_dim, hidden_dim, embedding_dim, lr )

    loss_func = nn.BCELoss()

    epochs = 10
    fit( epochs, model, loss_func, opt, train_dl, valid_dl )

    # Testsequence
    teststimuliSequence = get_teststimuliSequence()
    test_ds = SequenceDataset( ggen.stim2seqs( teststimuliSequence ) )
    test_dl = DataLoader( test_ds, batch_size=bs * 2, collate_fn=collate_batch )
    model.eval()
    with torch.no_grad():
        losses, nums = zip(
            *[loss_batch( model, loss_func, labels, seqs ) for labels, seqs in test_dl]
        )
        test_loss = np.sum( np.multiply(losses, nums) ) / np.sum( nums )
        for labels, seqs in test_dl:
            preds = model( seqs )
            for i in range( labels.size()[0] ):
                print( f'seq: {seqs[i].tolist()} model: {preds[i]} label: {labels[i]} ')

    print( "Total loss: ", test_loss )


if __name__ == '__main__':
    main()
