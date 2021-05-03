# File to load and test models

from losses import allOrNoneloss
import torch
import grammars as g
from autoencoder import SequenceLoss, evaluate, get_model, visual_eval
from grammar import GrammarGen, DataLoader, SequenceDataset, collate_batch

def main():

    LOADNAME = 'models/lt-g1.pt'
    bs = 4
    lr = 0.0001                         # Learning rate
    use_embedding = True                # Embedding Yes/No
    bidirectional = True                # bidirectional lstm layer Yes/o
    hidden_dim = 3                      # Lstm Neurons
    intermediate_dim = 200              # Intermediate Layer Neurons
    n_layers = 1                        # Lstm Layers
    dropout = 0.5
    grammaticality_bias = 0
    punishment = 0
    loss_func = SequenceLoss( GrammarGen( g.g1() ), grammaticality_bias=grammaticality_bias, punishment=punishment )

    g1_train, g1_test_gr, g1_test_ugr, g1_size = g.g1_dls( bs )

    input_dim = g1_size
    ### Load Model
    model, _ = get_model( input_dim, hidden_dim, intermediate_dim, n_layers, lr, dropout, use_embedding, bidirectional )
    model.load_state_dict( torch.load( LOADNAME ) )

    ### Test
    print( '\nTrain' )
    visual_eval( model, g1_train )
    print( evaluate( model, loss_func, g1_test_gr ) )

    print( '\nTest - Grammatical' )
    visual_eval( model, g1_test_gr )
    print( evaluate( model, loss_func, g1_test_gr ) )

    print( '\nTest - Ungrammatical' )
    visual_eval( model, g1_test_ugr )
    print( evaluate( model, loss_func, g1_test_ugr ) )
    print( evaluate( model, allOrNoneloss, g1_test_ugr ) )


if __name__ == "__main__":
    main()