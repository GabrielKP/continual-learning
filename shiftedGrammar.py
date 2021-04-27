
from grammar import *
from autoencoder import *

def main():                     # Best values so far
    bs = 4                      # 4
    epochs = 300                # 800 / 2000 for 1 layer
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

    LOADNAME = 'models/last-training_shifted.pt'
    SAVENAME = 'models/last-training_shifted.pt'

    # Grammar
    ggen = GrammarGen()

    # Note: BATCH IS IN FIRST DIMENSION
    # Train
    train_seqs = ggen.stim2seqs( get_trainstimuliSequence() )
    train_dl = get_dl( bs, train_seqs )

    # Train shifted
    train_shift_seqs, shifted_length = shiftStimuli( ggen, train_seqs )
    train_shift_dl = get_dl( bs, train_shift_seqs )

    # Test - Correct
    test_seqs = ggen.stim2seqs( get_correctStimuliSequence() )
    test_dl = get_dl( bs * 2, test_seqs, False )

    # Test shifted - Correct
    test_shift_seqs, _ = shiftStimuli( ggen, test_seqs )
    test_shift_dl = get_dl( bs * 2, test_shift_seqs, False )

    # Test - Incorrect
    test_incorrect_seqs = ggen.stim2seqs( get_incorrectStimuliSequence() )
    test_incorrect_dl = get_dl( bs * 2, test_incorrect_seqs, False )

    # Test shifted - Incorrect
    test_incorrect_shift_seqs, _ = shiftStimuli( ggen, test_incorrect_seqs )
    test_shift_dl = get_dl( bs * 2, test_incorrect_shift_seqs, False )

    input_dim = shifted_length

    # Get Model
    model, opt = get_model( input_dim, hidden_dim, intermediate_dim, n_layers, lr, dropout, use_embedding, bidirectional )
    if  not start_from_scratch:
        model.load_state_dict( torch.load( LOADNAME ) )

    # Loss Function
    loss_func = SequenceLoss( ggen, ignore_index=PAD_TOKEN, grammaticality_bias=grammaticality_bias, punishment=punishment )

    # Train
    _, hist_valid1, hist_check_dls = fit( epochs, model, loss_func, opt, train_dl, train_dl, teacher_forcing_ratio, SAVENAME, test_dl )

    # Load best model
    model.load_state_dict( torch.load( SAVENAME ) )

    # Train on shifted
    _, hist_valid2, hist_test2 = fit( epochs, model, loss_func, opt, train_shift_dl, train_shift_dl, teacher_forcing_ratio, SAVENAME, test_shift_dl )

    # plotHist( ( hist_valid1, 'Normal', ), ( hist_valid2, 'Shifted', ) )
    # plotHist( ( hist_test1, 'Normal', ), ( hist_test2, 'Shifted', ) )

    return
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
