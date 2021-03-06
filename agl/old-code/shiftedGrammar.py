
import grammars as g

from grammar import *
from ae_lstm import get_model, freezeParameters, unfreezeParameters, reInitParameters
from training import fit, one_hot, plotMultipleHist
from losses import SequenceLoss, allOrNoneloss
from torch import optim


def main():                     # Best values so far
    bs = 4                      # 4
    epochs = 1000                # 800 / 2000 for 1 layer
    lr = 0.01                   # 0.1
    teacher_forcing_ratio = 0.5  # 0.5
    use_embedding = True        # True
    bidirectional = True        # True
    hidden_dim = 3              # 3
    intermediate_dim = 200      # 200
    n_layers = 1                # 1
    dropout = 0.5
    start_from_scratch = True
    grammaticality_bias = 0
    punishment = 1
    conditions = (('fc_out', ), ('fc_one', ), ('embed', ), )

    LOADNAME = '../models/last-training_shifted.pt'
    SAVENAME = '../models/last-training_shifted.pt'

    # Grammar
    ggen = GrammarGen(g.g1())
    ggen2 = GrammarGen(g.g3())

    print( ggen.cores, ggen.stim2nbr )
    return

    train = ggen.stim2seqs( g.g1_train() )
    print( train[0] )
    print( one_hot( train[0], len(ggen) ) )

    return
    # Note: BATCH IS IN FIRST DIMENSION
    # Train
    train_seqs = ggen.stim2seqs(get_trainstimuliSequence())
    train_dl = get_dl(bs, train_seqs)

    # Train shifted
    train_shift_seqs, shifted_length = shiftStimuli(ggen, train_seqs)
    train_shift_dl = get_dl(bs, train_shift_seqs)

    # Test - Correct
    test_seqs = ggen.stim2seqs(get_correctStimuliSequence())
    test_dl = get_dl(bs * 2, test_seqs, False)

    # Test shifted - Correct
    test_shift_seqs, _ = shiftStimuli(ggen, test_seqs)
    test_shift_dl = get_dl(bs * 2, test_shift_seqs, False)

    # Test - Incorrect
    test_incorrect_seqs = ggen.stim2seqs(get_incorrectStimuliSequence())
    test_incorrect_dl = get_dl(bs * 2, test_incorrect_seqs, False)

    # Test shifted - Incorrect
    test_incorrect_shift_seqs, _ = shiftStimuli(ggen, test_incorrect_seqs)
    test_incorrect_shift_dl = get_dl(bs * 2, test_incorrect_shift_seqs, False)

    input_dim = len(ggen)

    # Get Model
    model, opt = get_model(input_dim, hidden_dim, intermediate_dim,
                           n_layers, lr, dropout, use_embedding, bidirectional)
    if not start_from_scratch:
        model.load_state_dict(torch.load(LOADNAME))

    # Loss Function
    loss_func = SequenceLoss(ggen, ignore_index=PAD_TOKEN,
                             grammaticality_bias=grammaticality_bias, punishment=punishment)

    # Check_dls
    check_dls1 = [(train_dl, loss_func, ), (train_dl, allOrNoneloss, ), (test_dl, loss_func, ),
                  (test_dl, allOrNoneloss, ), (test_incorrect_dl, loss_func, ), (test_incorrect_dl, allOrNoneloss, )]
    check_dls2 = [(train_shift_dl, loss_func, ), (train_shift_dl, allOrNoneloss, ), (test_shift_dl, loss_func, ),
                  (test_shift_dl, allOrNoneloss, ), (test_incorrect_shift_dl, loss_func, ), (test_incorrect_shift_dl, allOrNoneloss, )]

    # Train
    _, hist_valid1, hist_check_dls1 = fit(
        epochs, model, loss_func, opt, train_dl, train_dl, teacher_forcing_ratio, SAVENAME, check_dls1)

    # Load best model
    model.load_state_dict(torch.load(SAVENAME))

    # for name, param in model.named_parameters():
    #     if param.requires_grad == True:
    #         print( name )

    # print( "..\n\n")

    # Freeze Parameters
    reInitParameters(model, conditions)
    # unfreezeParameters( model, conditions )

    # for name, param in model.named_parameters():
    #     if param.requires_grad == True:
    #         print( name )

    # New optimizer (if not because of internal states it keeps updating old params)
    opt = optim.AdamW(filter(lambda p: p.requires_grad,
                      model.parameters()), lr=lr)

    # Train on shifted
    _, hist_valid2, hist_check_dls2 = fit(
        epochs, model, loss_func, opt, train_dl, train_dl, teacher_forcing_ratio, SAVENAME, check_dls1)

    labels = ("Normal", "Shifted", )
    sublabels = ("Test", "Training-Correct", "Training-Incorrect")
    stepsize = 5
    plotMultipleHist((hist_check_dls1, hist_check_dls2),
                     labels, stepsize, sublabels)

    return


if __name__ == '__main__':
    main()
