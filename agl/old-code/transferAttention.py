
from matplotlib.pyplot import step
import grammars as g
import torch
from grammar import GrammarGen
from ae_attention import displayAttention, get_model, freezeParameters, unfreezeParameters, reInitParameters
from training import fit, plotHist, plotMultipleHist
from losses import SequenceLoss, allOrNoneloss, dienesLoss
from torch import optim


def main():                     # Best values so far
    bs = 1                      # 4
    epochs = 1500                # 800 / 2000 for 1 layer
    lr = 0.0001                   # 0.1
    teacher_forcing_ratio = 0  # 0.5
    use_embedding = True        # True
    bidirectional = True        # True
    hidden_dim = 64              # 3
    intermediate_dim = 128      # 200
    n_layers = 1                # 1
    dropout = 0.5
    start_from_scratch = True
    grammaticality_bias = 0
    stepsize = 10
    punishment = 1
    conditions = (('fc_out', ), ('fc_one', ), ('embed', ), )

    prefix = 'g1-attn-'
    title = f'{prefix}AE-{int(bidirectional)}-{hidden_dim}-{n_layers}-{intermediate_dim}-lr{lr}'
    LOADNAME = 'models/autosave/' + title + '.pt'
    SAVENAME = 'models/autosave/' + title + '.pt'
    figpath = 'plots/autosave/'  + title + '.png'

    # Grammar
    ggen = GrammarGen(g.g1())

    g1_train, g1_test_gr, g1_test_ugr, g1_size = g.g1_dls(bs)

    input_dim = g1_size

    # Get Model
    model, opt = get_model(input_dim, hidden_dim, intermediate_dim,
                           n_layers, lr, dropout, use_embedding, bidirectional)
    # Load pretrained models weights
    if not start_from_scratch:
        model.load_state_dict(torch.load(LOADNAME))

    opt = optim.Adam(filter(lambda p: p.requires_grad,
                     model.parameters()), lr=lr)

    # Loss Function
    loss_func = SequenceLoss(
        ggen, grammaticality_bias=grammaticality_bias, punishment=punishment)

    # Check_dls
    check_dls1 = [(g1_train, loss_func, ), (g1_train, allOrNoneloss, ), (g1_test_gr, loss_func, ),
                  (g1_test_gr, allOrNoneloss, ), (g1_test_ugr, loss_func, ), (g1_test_ugr, allOrNoneloss, )]
    # check_dls2 = [ ( train_shift_dl, loss_func, ), ( train_shift_dl, allOrNoneloss, ), ( test_shift_dl, loss_func, ), ( test_shift_dl, allOrNoneloss, ), ( test_incorrect_shift_dl, loss_func, ), ( test_incorrect_shift_dl, allOrNoneloss, ) ]

    # Train
    _, hist_valid1, hist_check_dls1 = fit(
        epochs, model, loss_func, opt, g1_train, g1_train, teacher_forcing_ratio, SAVENAME, check_dls1, stepsize)

    # Load best model
    model.load_state_dict(torch.load(SAVENAME))

    sublabels = ("Training", "Test-Correct", "Test-Incorrect")
    ylims = ([0, len(g.g1_train())], [0, len(g.g1_test_gr())],
             [0, len(g.g1_test_ugr_balanced())])
    plotMultipleHist((hist_check_dls1, ), ("Normal",),
                     stepsize, sublabels, ylims, title, figpath)

    for labels, seqs in g1_train:
        for seq in seqs:
            displayAttention(model, seq)
    return

    # Reinit Parameters
    reInitParameters(model, conditions)

    # New optimizer (if not because of internal states it keeps updating old params)
    opt = optim.AdamW(filter(lambda p: p.requires_grad,
                      model.parameters()), lr=lr)

    # Train on shifted
    _, hist_valid2, hist_check_dls2 = fit(
        epochs, model, loss_func, opt, g1_train, g1_train, teacher_forcing_ratio, SAVENAME, check_dls1)

    labels = ("Normal", "Shifted", )
    sublabels = ("Test", "Training-Correct", "Training-Incorrect")
    stepsize = 5
    plotMultipleHist((hist_check_dls1, hist_check_dls2),
                     labels, stepsize, sublabels)

    return


if __name__ == '__main__':
    main()
