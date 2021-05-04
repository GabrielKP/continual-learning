# Main file

import random
import torch
import grammars as g
from torch.utils.data.dataloader import DataLoader
from grammar import GrammarGen, START_TOKEN, SequenceDataset, collate_batch
from torch import nn
from torch import optim
from training import fit, visual_eval, evaluate, plotHist
from losses import SequenceLoss


device = torch.device( 'cuda' if torch.cuda.is_available() else 'cpu' )
PAD_TOKEN = 0   # ugly but works for now
END_TOKEN = 2
CLIP = 0.5

class Encoder(nn.Module):

    def __init__(self, input_dim, hidden_dim, intermediate_dim, n_layers, dropout, embedding=True, bidirectional=False ):
        super(Encoder, self).__init__()

        # Vars
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.embedding_dim = input_dim
        self.n_layers = n_layers
        self.intermediate_dim = intermediate_dim
        self.bidirectional = bidirectional

        # Layers
        self.embed = nn.Embedding( self.input_dim, self.embedding_dim )
        if not embedding:
            self.embed.weight.data = torch.eye( input_dim )

        self.dropout = nn.Dropout( dropout )

        self.fc_one = nn.Linear( self.embedding_dim, self.intermediate_dim )

        self.ac_one = nn.ReLU()

        self.lstm = nn.LSTM( self.intermediate_dim, self.hidden_dim, n_layers, batch_first=True, bidirectional=self.bidirectional )


    def forward(self, seqs):

        # Handle sequences separately

        hiddens = []
        cells = []

        for seq in seqs:

            embed = self.dropout( self.embed( seq ) )

            intermediate = self.dropout( self.ac_one( self.fc_one( embed ) ) )

            _, (hidden, cell) = self.lstm( intermediate.unsqueeze(0) )

            hiddens.append( hidden.squeeze() )
            cells.append( cell.squeeze() )

        hiddens = torch.stack( hiddens )
        cells = torch.stack( cells )

        return  hiddens, cells


class Decoder(nn.Module):

    def __init__(self, output_dim, hidden_dim, intermediate_dim, n_layers, dropout, embedding=True, bidirectional=False ):
        super(Decoder, self).__init__()

        # Vars
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.embedding_dim = output_dim
        self.n_layers = n_layers
        self.intermediate_dim = intermediate_dim
        self.bidirectional = bidirectional

        # Layers
        self.embed = nn.Embedding( self.output_dim, self.embedding_dim )
        if not embedding:
            self.embed.weight.data = torch.eye( self.embedding_dim )

        self.lstm = nn.LSTM( self.embedding_dim, self.hidden_dim, self.n_layers, batch_first=True, bidirectional=True )

        self.fc_out = nn.Linear( intermediate_dim, output_dim )

        self.fc_one = nn.Linear( hidden_dim + hidden_dim * bidirectional, intermediate_dim )

        self.ac_one = nn.ReLU()

        self.dropout = nn.Dropout( dropout )


    def forward(self, nInput, hidden, cell):

        embed = self.dropout( self.embed( nInput ) )

        output, (hidden, cell) = self.lstm( embed.unsqueeze(0).unsqueeze(0), (hidden.unsqueeze(1), cell.unsqueeze(1)) )

        intermediate = self.dropout( self.ac_one( self.fc_one( output.squeeze() ) ) )

        output = self.fc_out( intermediate )

        return  output, hidden.squeeze(), cell.squeeze()


class AutoEncoder(nn.Module):
    def __init__(self, encoder, decoder):
        super(AutoEncoder, self).__init__()

        self.encoder = encoder
        self.decoder = decoder

    def forward(self, labels, seqs, teacher_forcing_ratio = 0.5 ):

        bs = len( seqs )

        # Vector to store outputs
        outputs = []

        # Encode
        hiddens, cells = self.encoder( seqs )

        # First input to decoder is start sequence token
        nInputs = torch.tensor( [ START_TOKEN ] * bs )

        # Decide once beforand for teacherforcing or not
        teacher_forcing = random.random() < teacher_forcing_ratio

        for b in range(bs):
            nInput = nInputs[b]
            hidden = hiddens[b]
            cell = cells[b]
            seq = seqs[b]

            seq_out = []

            for t in range( 1, len( seq ) ):

                # Decode stimulus
                output, hidden, cell = self.decoder( nInput, hidden, cell )

                # Save output
                seq_out.append( output )

                # Teacher forcing
                if teacher_forcing:
                    nInput = seq[t]
                else:
                    nInput = output.argmax(-1)

            seq_out = torch.stack( seq_out )

            outputs.append( seq_out )

        return outputs


def init_weights(m):
    for _, param in m.named_parameters():
        nn.init.uniform_(param.data, -0.08, 0.08)


def count_parameters(model):
    return sum( p.numel() for p in model.parameters() if p.requires_grad )


def get_model(input_dim, hidden_dim, intermediate_dim, n_layers, lr, dropout, use_embedding=True, bidirectional=False):

    encoder = Encoder( input_dim, hidden_dim, intermediate_dim, n_layers, dropout, use_embedding, bidirectional )
    decoder = Decoder( input_dim, hidden_dim, intermediate_dim, n_layers, dropout, use_embedding, bidirectional )

    model = AutoEncoder( encoder, decoder )
    print( model.apply( init_weights ) )
    print( f'The model has {count_parameters(model):,} trainable parameters' )
    return model, optim.AdamW( model.parameters(), lr=lr )


def applyOnParameters( model, conditions, apply_function ):
    """
    conditions is a tuple of tuples (condition):
    ( ( keyword1 AND keyword2 AND ... ) OR ( keyword3 AND ... ) OR ... )
    a condition is multiple keywords which need to be in the parameter name
    to freeze the parameter
    apply_function is the function applied on the parameter chosen
    """
    for name, param in model.named_parameters():
        # Check every condition
        for condition in conditions:
            # check every keyword
            allincluded = True
            for keyword in condition:
                if keyword not in name:
                    allincluded = False
                    break
            if allincluded:
                apply_function( param )


def reInitParameters( model, conditions ):
    """
    conditions is a tuple of tuples (condition):
    ( ( keyword1 AND keyword2 AND ... ) OR ( keyword3 AND ... ) OR ... )
    a condition is multiple keywords which need to be in the parameter name
    to freeze the parameter
    """
    def init( param ):
        nn.init.uniform_( param.data, -0.08, 0.08 )
    applyOnParameters( model, conditions, init )


def freezeParameters( model, conditions ):
    """
    conditions is a tuple of tuples (condition):
    ( ( keyword1 AND keyword2 AND ... ) OR ( keyword3 AND ... ) OR ... )
    a condition is multiple keywords which need to be in the parameter name
    to freeze the parameter
    """
    def freeze( param ):
        param.requires_grad = False
    applyOnParameters( model, conditions, freeze )


def unfreezeParameters( model, conditions ):
    """
    conditions is a tuple of tuples (condition):
    ( ( keyword1 AND keyword2 AND ... ) OR ( keyword3 AND ... ) OR ... )
    a condition is multiple keywords which need to be in the parameter name
    to freeze the parameter
    """
    def unfreeze( param ):
        param.requires_grad = True
    applyOnParameters( model, conditions, unfreeze )


def main():                     # Best values so far
    bs = 4                      # 4
    epochs = 200                # 800 / 2000 for 1 layer
    lr = 0.01                   # 0.1
    teacher_forcing_ratio = 0.5 # 0.5
    use_embedding = True        # True
    bidirectional = True        # True
    hidden_dim = 3              # 3
    intermediate_dim = 200      # 200
    n_layers = 1                # 1
    dropout = 0.5
    start_from_scratch = True
    grammaticality_bias = 0
    punishment = 1
    # 4.pt 200 5 3
    # 5.pt 100 5 3
    LOADNAME = '../models/last-training1.pt'
    SAVENAME = '../models/last-training1.pt'
    # Grammar
    ggen = GrammarGen()

    # Note: BATCH IS IN FIRST DIMENSION
    # Train
    train_seqs = ggen.stim2seqs( g.g0_train() )
    train_ds = SequenceDataset( train_seqs )
    train_dl = DataLoader( train_ds, batch_size=bs, shuffle=True, collate_fn=collate_batch )

    # Validation
    valid_seqs = ggen.generate( 8 )
    valid_ds = SequenceDataset( valid_seqs )
    valid_dl = DataLoader( valid_ds, batch_size=bs, collate_fn=collate_batch )

    # Test - Correct
    test_seqs = ggen.stim2seqs( g.g0_test_gr() )
    test_ds = SequenceDataset( test_seqs )
    test_dl = DataLoader( test_ds, batch_size=bs, collate_fn=collate_batch )

    # Test - Incorrect
    test_incorrect_seqs = ggen.stim2seqs( g.g0_test_ugr() )
    test_incorrect_ds = SequenceDataset( test_incorrect_seqs )
    test_incorrect_dl = DataLoader( test_incorrect_ds, batch_size=bs * 2, collate_fn=collate_batch )

    input_dim = len( ggen )

    # Get Model
    model, opt = get_model( input_dim, hidden_dim, intermediate_dim, n_layers, lr, dropout, use_embedding, bidirectional )
    if  not start_from_scratch:
        model.load_state_dict( torch.load( LOADNAME ) )

    # Loss Function
    # loss_func = nn.CrossEntropyLoss( ignore_index=PAD_TOKEN, reduction='sum' )
    loss_func = SequenceLoss( ggen, ignore_index=PAD_TOKEN, grammaticality_bias=grammaticality_bias, punishment=punishment )

    # Train
    hist_train, hist_valid = fit( epochs, model, loss_func, opt, train_dl, train_dl, teacher_forcing_ratio, SAVENAME )

    # Load best model
    model.load_state_dict( torch.load( SAVENAME ) )

    plotHist( ( hist_train, 'Train', ), ( hist_valid, 'Valid') )

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
