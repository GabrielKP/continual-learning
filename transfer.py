
from matplotlib.pyplot import step
import grammars as g
import torch
from grammar import GrammarGen
from autoencoder import get_model, freezeParameters, unfreezeParameters, reInitParameters
from training import fit, plotHist, plotMultipleHist
from losses import SequenceLoss, allOrNoneloss
from torch import optim

def main():                     # Best values so far
    bs = 4                      # 4
    epochs = 1500                # 800 / 2000 for 1 layer
    lr = 0.01                   # 0.1
    teacher_forcing_ratio = 0.5 # 0.5
    use_embedding = True        # True
    bidirectional = True        # True
    hidden_dim = 4              # 3
    intermediate_dim = 500      # 200
    n_layers = 1                # 1
    dropout = 0.5
    start_from_scratch = True
    grammaticality_bias = 0
    stepsize = 10
    punishment = 1
    conditions = ( ( 'fc_out', ), ( 'fc_one', ), ( 'embed', ), )

    LOADNAME = 'models/lt-g1.pt'
    SAVENAME = 'models/lt-g1.pt'

    # Grammar
    ggen = GrammarGen( g.g1() )


    # seqs = ggen.seqs2stim( ggen.generate( 5 ) )

    # print( [ ''.join( seq ) for seq in seqs ] )

    g1_train, g1_test_gr, g1_test_ugr, g1_size = g.g1_dls( bs )

    input_dim = g1_size

    # Get Model
    model, opt = get_model( input_dim, hidden_dim, intermediate_dim, n_layers, lr, dropout, use_embedding, bidirectional )
    # Load pretrained models weights
    if  not start_from_scratch:
        model.load_state_dict( torch.load( LOADNAME ) )

    # Loss Function
    loss_func = SequenceLoss( ggen, grammaticality_bias=grammaticality_bias, punishment=punishment )

    # Check_dls
    check_dls1 = [ ( g1_train, loss_func, ), ( g1_train, allOrNoneloss, ), ( g1_test_gr, loss_func, ), ( g1_test_gr, allOrNoneloss, ), ( g1_test_ugr, loss_func, ), ( g1_test_ugr, allOrNoneloss, ) ]
    # check_dls2 = [ ( train_shift_dl, loss_func, ), ( train_shift_dl, allOrNoneloss, ), ( test_shift_dl, loss_func, ), ( test_shift_dl, allOrNoneloss, ), ( test_incorrect_shift_dl, loss_func, ), ( test_incorrect_shift_dl, allOrNoneloss, ) ]

    # Train
    _, hist_valid1, hist_check_dls1 = fit( epochs, model, loss_func, opt, g1_train, g1_train, teacher_forcing_ratio, SAVENAME, check_dls1, stepsize )

    # Load best model
    model.load_state_dict( torch.load( SAVENAME ) )

    sublabels = ( "Test", "Training-Correct", "Training-Incorrect")
    ylims = ( [0,len( g.g1_train() )], [0,len( g.g1_test_gr() )], [0,len( g.g1_test_ugr() )] )
    plotMultipleHist( ( hist_check_dls1, ), ("Normal",), stepsize, sublabels, ylims )
    return

    # Reinit Parameters
    reInitParameters( model, conditions )

    # New optimizer (if not because of internal states it keeps updating old params)
    opt = optim.AdamW( filter(lambda p: p.requires_grad, model.parameters()), lr=lr )

    # Train on shifted
    _, hist_valid2, hist_check_dls2 = fit( epochs, model, loss_func, opt, g1_train, g1_train, teacher_forcing_ratio, SAVENAME, check_dls1 )

    labels = ( "Normal", "Shifted", )
    sublabels = ( "Test", "Training-Correct", "Training-Incorrect")
    stepsize = 5
    plotMultipleHist( ( hist_check_dls1, hist_check_dls2 ), labels, stepsize, sublabels )

    return


if __name__ == '__main__':
    main()
