# Main file

import time
import random
import torch
import numpy as np
from torch.nn.modules import dropout
from torch.optim import optimizer
from torch.utils.data.dataloader import DataLoader

from grammar import GrammarGen, START_TOKEN, SequenceDataset, collate_batch, get_correctStimuliSequence, get_incorrectStimuliSequence, get_trainstimuliSequence, get_teststimuliSequence

from torch import nn
from torch import optim

device = torch.device( 'cuda' if torch.cuda.is_available() else 'cpu' )
PAD_TOKEN = 0   # ugly but works for now
END_TOKEN = 2
CLIP = 0.5

class Encoder(nn.Module):

    def __init__(self, input_dim, hidden_dim, intermediate_dim, n_layers, dropout, embedding=True ):
        super(Encoder, self).__init__()

        # Vars
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.embedding_dim = input_dim
        self.n_layers = n_layers
        self.intermediate_dim = intermediate_dim

        # Layers
        self.embed = nn.Embedding( self.input_dim, self.embedding_dim )
        if not embedding:
            self.embed.weight.data = torch.eye( input_dim )

        self.lstm = nn.LSTM( self.intermediate_dim, self.hidden_dim, n_layers, batch_first=True )

        self.fc_one = nn.Linear( self.embedding_dim, self.intermediate_dim )

        self.ac_one = nn.ReLU()

        self.dropout = nn.Dropout( dropout )

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

    def __init__(self, output_dim, hidden_dim, intermediate_dim, n_layers, dropout, embedding=True ):
        super(Decoder, self).__init__()

        # Vars
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.embedding_dim = output_dim
        self.n_layers = n_layers
        self.intermediate_dim = intermediate_dim

        # Layers
        self.embed = nn.Embedding( self.output_dim, self.embedding_dim )
        if not embedding:
            self.embed.weight.data = torch.eye( self.embedding_dim )

        self.lstm = nn.LSTM( self.embedding_dim, self.hidden_dim, self.n_layers, batch_first=True )

        self.fc_out = nn.Linear( intermediate_dim, output_dim )

        self.fc_one = nn.Linear( hidden_dim, intermediate_dim )

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


def get_model(input_dim, hidden_dim, intermediate_dim, n_layers, lr, dropout, use_embedding=True):

    encoder = Encoder( input_dim, hidden_dim, intermediate_dim, n_layers, dropout, use_embedding )
    decoder = Decoder( input_dim, hidden_dim, intermediate_dim, n_layers, dropout, use_embedding )

    model = AutoEncoder( encoder, decoder )
    print( model.apply( init_weights ) )
    print( f'The model has {count_parameters(model):,} trainable parameters' )
    return model, optim.AdamW( model.parameters(), lr=lr )


def loss_batch(model, loss_func, labels, seqs, teacher_forcing_ratio=0.5, opt=None):
    # loss function gets padded sequences -> autoencoder
    labels = seqs

    # Get model output
    output = model( labels, seqs, teacher_forcing_ratio )

    # Cut of start sequence & reshaping
    # output = output[:,1:].reshape(-1, model.decoder.output_dim )
    # labels = labels[:,1:].reshape(-1)

    # print( "------------------" )
    # print( output )
    # print( labels )

    # Compute loss
    loss = loss_func( output, labels )

    if opt is not None:
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
        opt.step()
        opt.zero_grad()

    return loss.item(), len( labels )


def train(model, train_dl, loss_func, opt, teacher_forcing_ratio):
    """ Trains 1 epoch of the model, returns loss for train set"""

    model.train()

    epoch_loss = 0
    epoch_num_seqs = 0

    for labels, seqs in train_dl:
        batch_loss, batch_num_seqs = loss_batch(model, loss_func, labels, seqs, teacher_forcing_ratio, opt)
        epoch_loss += batch_loss * batch_num_seqs
        epoch_num_seqs += batch_num_seqs

    return epoch_loss / epoch_num_seqs


def evaluate(model, loss_func, test_dl):
    model.eval()
    with torch.no_grad():
        losses, nums = zip(
            *[loss_batch( model, loss_func, labels, seqs, teacher_forcing_ratio=0 ) for labels, seqs in test_dl]
        )
        return np.sum( np.multiply( losses, nums ) ) / np.sum( nums )


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int( elapsed_time / 60 )
    elapsed_secs = elapsed_time - (elapsed_mins * 60)
    return elapsed_mins, elapsed_secs


def fit(epochs, model, loss_func, opt, train_dl, valid_dl, teacher_forcing_ratio=0.5, FILENAME='aa'):
    """ Fits model on train data, printing val and train loss"""

    best_val_loss = float('inf')

    for epoch in range(epochs):

        start_time = time.time()

        train_loss = train(model, train_dl, loss_func, opt, teacher_forcing_ratio)
        valid_loss = evaluate(model, loss_func, valid_dl)

        end_time = time.time()

        if valid_loss < best_val_loss:
            best_val_loss = valid_loss
            torch.save( model.state_dict(), FILENAME )

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        print(f'Epoch: {epoch+1:03} | Time: {epoch_mins}m {epoch_secs:.2}s')
        print(f'\tTrain Loss: {train_loss:.5f} |  Val. Loss: {valid_loss:.5f}')


def cutStartAndEndToken(seq):
    ret = []
    for stim in seq:
        if stim == END_TOKEN:
            break
        ret.append( stim )
    return ret


def visual_eval(model, test_dl):
    model.eval()
    with torch.no_grad():
        for labels, seqs in test_dl:
            output = model( labels, seqs, teacher_forcing_ratio=0 )
            # print( output )
            for b, seq in enumerate( seqs ):
                prediction = output[b].argmax(-1)
                trgtlist = seq.tolist()[1:-1]
                predlist = cutStartAndEndToken( prediction.tolist() )
                print( f'Same: {trgtlist == predlist} Truth: {trgtlist} - Pred: {predlist}' )


def softmax( x ):
    return x.exp() / x.exp().sum(-1).unsqueeze(-1)

class SequenceLoss():

    def __init__(self, grammarGen: GrammarGen, ignore_index=None, grammaticality_bias=0.5, punishment=1):
        self.ggen = grammarGen
        self.gbias = grammaticality_bias
        self.number_grammar = grammarGen.number_grammar
        self.punishment = punishment
        self.ignore_index = ignore_index

        self.CEloss = nn.CrossEntropyLoss( ignore_index=ignore_index )

        self.init_grammaticalityMatrix()

    def init_grammaticalityMatrix(self):

        # Grammaticality Matrix is a Matrix of ones, in which
        # only the entries are 0 which are in the grammar
        vocab_size = len( self.ggen )
        self.grammaticalityMatrix = torch.ones( ( vocab_size, vocab_size ) )

        for stimA, stimBs in self.number_grammar.items():
            for stimB in stimBs:
                self.grammaticalityMatrix[stimA, stimB] = 0

        print( self.grammaticalityMatrix )
        self.grammaticality_indices = self.grammaticalityMatrix == 1


    def __call__(self, outputs: torch.tensor, labels: torch.tensor):

        bs = len( outputs )

        loss = torch.zeros( bs )

        for b in range(bs):

            CEOutput = outputs[b]
            CELabel = labels[b][1:].type( torch.long )

            loss[b] = self.CEloss( CEOutput, CELabel )

        return loss.mean()

        # ## print( "BOSD", vocab_size )
        # # print( outputs )

        # # Convert to probabilities
        # outputs = softmax( outputs )

        # # print( outputs )
        # # print( labels )

        # loss = torch.zeros( ( bs, seqlength - 1 ) )

        # # test = torch.zeros( ( vocab_size ) )
        # # test[6] = 1.0
        # # test[7] = 1.0

        # # Judge grammaticality
        # for batch in range( bs ):

        #     seq = outputs[batch]

        #     # seq = torch.tensor(
        #     #     [[0, 1, 0, 0, 0, 0, 0, 0],
        #     #     [0, 0, 0, 1, 0, 0, 0, 0],
        #     #     [1.6311e-06, 9.4581e-07, 2.9531e-05, 7.9313e-07, 4.9984e-01, 1.8668e-04, 9.9475e-07, 4.9994e-01],
        #     #     [1.6311e-06, 9.4579e-07, 2.9530e-05, 7.9310e-07, 4.9984e-01, 1.8668e-04, 9.9473e-07, 4.9994e-01],
        #     #     [1.6311e-06, 9.4579e-07, 2.9530e-05, 7.9310e-07, 4.9984e-01, 1.8668e-04, 9.9473e-07, 4.9994e-01]])
        #     # print( "seq" )
        #     # print( seq )
        #     # Compare stimuli pairwise
        #     for i in range( seqlength - 1 ):

        #         # loss[batch, i] = (seq[i] * test).sum()
        #         # continue

        #         # if sequence ended, then ignore
        #         if labels[batch,i] == self.ignore_index:
        #             break

        #         prev =  torch.tensor( seq[i].unsqueeze(0).transpose( 0, 1 ).tolist() )

        #         transitionMatrix = torch.matmul( prev, seq[i + 1].unsqueeze(0) )

        #         #print( transitionMatrix )

        #         errorMatrix = transitionMatrix[self.grammaticality_indices]

        #         # print( errorMatrix )
        #         # print( errorMatrix.sum() )

        #         loss[batch, i] = ( errorMatrix.sum() * self.punishment ).pow(2)

        return self.CEloss( CEOutputs, CELabels )
        return loss.mean() * self.gbias + self.CEloss( CEOutputs, CELabels ) * ( 1 - self.gbias )


def main():
    bs = 3
    # Grammar
    ggen = GrammarGen()

    # Note: BATCH IS IN FIRST DIMENSION
    # Train
    train_seqs = ggen.stim2seqs( get_trainstimuliSequence() )
    train_ds = SequenceDataset( train_seqs )
    train_dl = DataLoader( train_ds, batch_size=bs, shuffle=True, collate_fn=collate_batch )

    # Validation
    valid_seqs = ggen.generate( 20 )
    valid_ds = SequenceDataset( valid_seqs )
    valid_dl = DataLoader( valid_ds, batch_size=bs, collate_fn=collate_batch )

    # Test - Correct
    test_seqs = ggen.stim2seqs( get_correctStimuliSequence() )
    test_ds = SequenceDataset( test_seqs )
    test_dl = DataLoader( test_ds, batch_size=bs, collate_fn=collate_batch )

    # Test - Incorrect
    test_incorrect_seqs = ggen.stim2seqs( get_incorrectStimuliSequence() )
    test_incorrect_ds = SequenceDataset( test_incorrect_seqs )
    test_incorrect_dl = DataLoader( test_incorrect_ds, batch_size=bs * 2, collate_fn=collate_batch )

    # Misc parameters
    # dropout?
    epochs = 5000
    lr = 0.0001
    teacher_forcing_ratio = 0.5
    use_embedding = True
    hidden_dim = 5
    intermediate_dim = 200
    n_layers = 4
    dropout = 0.5
    start_from_scratch = True
    input_dim = len( ggen )
    # 4.pt 200 3
    # 5.pt 100 3 5
    FILENAME = 'autoEncoder-5.pt'
    # torch.autograd.set_detect_anomaly(True)

    # Get Model
    model, opt = get_model( input_dim, hidden_dim, intermediate_dim, n_layers, lr, dropout, use_embedding )
    if  not start_from_scratch:
        model.load_state_dict( torch.load( FILENAME ) )

    # Loss Function
    # loss_func = nn.CrossEntropyLoss( ignore_index=PAD_TOKEN, reduction='sum' )
    loss_func = SequenceLoss( ggen, ignore_index=PAD_TOKEN, grammaticality_bias=0, punishment=1 )

    # Train
    fit( epochs, model, loss_func, opt, train_dl, valid_dl, teacher_forcing_ratio, FILENAME )

    # Load best model
    model.load_state_dict( torch.load( FILENAME ) )

    # for labels, seqs in train_dl:
    #     labels = nn.utils.rnn.pad_sequence( seqs, batch_first=True, padding_value=PAD_TOKEN ).type( torch.long )

    #     output = model( labels, seqs )

    #     print( "output\n", output )
    #     print( output.argmax(-1) )
    #     print( loss_func( output[:,1:], labels[:,1:] ) )
    #     break
    # return

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


def othermain():

    ggen = GrammarGen()
    loss = SequenceLoss( ggen, ignore_index=PAD_TOKEN, grammaticality_bias=1 )


if __name__ == '__main__':
    main()
