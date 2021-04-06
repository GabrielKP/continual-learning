# Main file

import random
import sys
import torch
import numpy as np
from torch.optim import optimizer
from torch.utils.data.dataloader import DataLoader

from grammar import GrammarGen, SequenceDataset, collate_batch, get_data, get_invalidStimuliSequence, get_trainstimuliSequence, get_teststimuliSequence, get_validStimuliSequence

from torch import nn
from torch import optim

device = torch.device( 'cuda' if torch.cuda.is_available() else 'cpu' )
PAD_TOKEN = 0   # ugly but works for now
CLIP = 0.5

class Encoder(nn.Module):

    def __init__(self, input_dim, hidden_dim, n_layers, embedding=True ):
        super(Encoder, self).__init__()

        # Vars
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.embedding_dim = input_dim
        self.n_layers = n_layers

        # Layers
        self.embed = nn.Embedding( self.input_dim, self.embedding_dim )
        if not embedding:
            self.embed.weight.data = torch.eye( input_dim )
        self.lstm = nn.LSTM( self.embedding_dim, self.hidden_dim, n_layers, batch_first=True )

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


class Decoder(nn.Module):

    def __init__(self, output_dim, hidden_dim, n_layers, embedding=True ):
        super(Decoder, self).__init__()

        # Vars
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.embedding_dim = output_dim
        self.n_layers = n_layers

        # Layers
        self.embed = nn.Embedding( self.output_dim, self.embedding_dim )
        if not embedding:
            self.embed.weight.data = torch.eye( self.embedding_dim )

        self.lstm = nn.LSTM( self.embedding_dim, self.hidden_dim, self.n_layers, batch_first=True )

        self.fc_out = nn.Linear( hidden_dim, output_dim )

        # Init Params
        # for param in self.lstm.parameters():
        #     nn.init.zeros_( param )

        # nn.init.zeros_( self.lin.weight )

    def forward(self, nInput, hidden, cell):

        nInput = nInput.unsqueeze(-1)

        embedded = self.embed( nInput )

        output, (hidden, cell) = self.lstm( embedded, ( hidden, cell ) )

        # print( output.size() )

        prediction = self.fc_out( output.squeeze(1) )

        # print( output.size() )

        return  prediction, hidden, cell


class AutoEncoder(nn.Module):
    def __init__(self, encoder, decoder):
        super(AutoEncoder, self).__init__()

        self.encoder = encoder
        self.decoder = decoder

    def forward(self, labels, seqs, teacher_forcing_ratio = 0.5 ):

        trgts = nn.utils.rnn.pad_sequence( seqs, batch_first=True, padding_value=PAD_TOKEN )

        batch_size = len( seqs )
        trgt_vocab_size = self.decoder.output_dim
        max_len = max( [len( seq ) for seq in seqs] )

        # Vector to store outputs
        outputs = torch.zeros( batch_size, max_len, trgt_vocab_size )

        # Encode
        hidden, cell = self.encoder( seqs )

        # First input to decoder is start sequence token
        nInput = torch.tensor( [ 1 ] * batch_size )

        # Let's go
        for t in range( 1, max_len ):

            # Decode stimulus
            output, hidden, cell = self.decoder( nInput, hidden, cell )

            # Save output
            outputs[:,t] = output

            # Teacher forcing
            if random.random() < teacher_forcing_ratio:
                nInput = trgts[:,t]
            else:
                nInput = output.argmax(-1)

        return outputs


def loss_batch(model, loss_func, labels, seqs, teacher_forcing_ratio=0.5, opt=None):
    # loss function gets padded sequences -> autoencoder
    labels = nn.utils.rnn.pad_sequence( seqs, batch_first=True, padding_value=PAD_TOKEN ).type( torch.long )

    # Get model output
    output = model( labels, seqs, teacher_forcing_ratio )

    # Cut of start sequence
    output = output[:,1:].reshape(-1, model.decoder.output_dim )
    labels = labels[:,1:].reshape(-1)

    # Compute loss
    loss = loss_func( output, labels )

    if opt is not None:
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
        opt.step()
        opt.zero_grad()

    return loss.item(), len( labels )


def fit(epochs, model, loss_func, opt, train_dl, valid_dl, teacher_forcing_ratio=0.5):
    best_val_loss = float('inf')
    for epoch in range(epochs):
        model.train()
        for labels, seqs in train_dl:
            loss_batch(model, loss_func, labels, seqs, teacher_forcing_ratio, opt)

        val_loss = evaluate(model, loss_func, valid_dl)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save( model.state_dict(), 'autoEncoder-3.pt' )

        print( epoch, val_loss )


def evaluate(model, loss_func, test_dl):
    model.eval()
    with torch.no_grad():
        losses, nums = zip(
            *[loss_batch( model, loss_func, labels, seqs, teacher_forcing_ratio=0 ) for labels, seqs in test_dl]
        )
        return np.sum( np.multiply( losses, nums ) ) / np.sum( nums )


def init_weights(m):
    for _, param in m.named_parameters():
        nn.init.uniform_(param.data, -0.08, 0.08)


def get_model(input_dim, hidden_dim, n_layers, lr):

    encoder = Encoder( input_dim, hidden_dim, n_layers )
    decoder = Decoder( input_dim, hidden_dim, n_layers )

    model = AutoEncoder( encoder, decoder )
    print( model.apply( init_weights ) )
    return model, optim.Adam( model.parameters(), lr=lr )


def visual_eval(model, test_dl):
    model.eval()
    with torch.no_grad():
        for labels, seqs in test_dl:
            output = model( labels, seqs, teacher_forcing_ratio=0 )
            predictions = output.argmax(-1)
            for i, seq in enumerate( seqs ):
                print( f'Truth: {seq.tolist()[1:-1]} - Pred: {predictions[i].tolist()[1:-1]}' )


def main():
    bs = 2
    # Grammar
    ggen = GrammarGen()
    # Train
    train_seqs = ggen.stim2seqs( get_trainstimuliSequence() )
    train_ds = SequenceDataset( train_seqs )
    train_dl = DataLoader( train_ds, batch_size=bs, shuffle=True, collate_fn=collate_batch )

    # Test - Correct
    test_seqs = ggen.stim2seqs( get_validStimuliSequence() )
    test_ds = SequenceDataset( test_seqs )
    test_dl = DataLoader( test_ds, batch_size=bs * 2, collate_fn=collate_batch )

    # Test - Incorrect
    test_incorrect_seqs = ggen.stim2seqs( get_invalidStimuliSequence() )
    test_incorrect_ds = SequenceDataset( test_incorrect_seqs )
    test_incorrect_dl = DataLoader( test_incorrect_ds, batch_size=bs * 2, collate_fn=collate_batch )

    # Misc parameters
    epochs = 400
    lr = 0.001
    teacher_forcing_ratio = 1
    hidden_dim = 4
    n_layers = 3
    start_from_scratch = False
    input_dim = len( ggen ) + 3 # need 3 tokens to symbolize start, end, and padding

    # Get Model
    model, opt = get_model( input_dim, hidden_dim, n_layers, lr )
    if  not start_from_scratch:
        model.load_state_dict( torch.load( 'autoEncoder-3.pt' ) )

    # Loss Function
    loss_func = nn.CrossEntropyLoss( ignore_index=PAD_TOKEN )

    # Train
    fit( epochs, model, loss_func, opt, train_dl, test_dl, teacher_forcing_ratio )

    # Load best model
    model.load_state_dict( torch.load( 'autoEncoder-3.pt' ) )

    # Test
    print( '\nTrain' )
    visual_eval( model, train_dl )
    print( evaluate( model, loss_func, train_dl ) )

    print( '\nTest - Correct' )
    visual_eval( model, test_dl )
    print( evaluate( model, loss_func, test_dl ) )

    print( '\nTest - Incorrect' )
    visual_eval( model, test_incorrect_dl )
    print( evaluate( model, loss_func, test_incorrect_dl ) )


if __name__ == '__main__':
    main()
