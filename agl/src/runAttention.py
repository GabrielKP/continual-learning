# File to load and test models

from ae_attention import displayAttention
from losses import allOrNoneloss, SequenceLoss
import torch
import grammars as g
from ae_attention import get_model
from grammar import GrammarGen
from training import dienes_eval, evaluate, visual_eval


def main():

    LOADNAME = 'models/autosave/g1-gru-AE-1-5-1-500-lr0.001.pt'
    bs = 4
    lr = 0.0005                         # Learning rate
    use_embedding = True                # Embedding Yes/No
    bidirectional = True                # bidirectional lstm layer Yes/o
    hidden_dim = 5                      # Lstm Neurons
    intermediate_dim = 128
    n_layers = 1                        # Lstm Layers
    dropout = 0.5
    grammaticality_bias = 0
    punishment = 0
    ggen = GrammarGen(g.g1())
    loss_func = SequenceLoss(
        ggen, grammaticality_bias=grammaticality_bias, punishment=punishment)
    prefix = 'g1-attn-'
    title = f'{prefix}AE-{int(bidirectional)}-{hidden_dim}-{n_layers}-{intermediate_dim}-lr{lr}'
    LOADNAME = 'models/autosave/' + title + '.pt'

    g1_train, g1_test_gr, g1_test_ugr, g1_size = g.g1_dls(bs)

    input_dim = g1_size
    # Load Model
    model, _ = get_model(input_dim, hidden_dim, intermediate_dim,
                         n_layers, lr, dropout, use_embedding, bidirectional)
    model.load_state_dict(torch.load(LOADNAME))

    # Test
    print('\nTrain')
    print(visual_eval(model, g1_train, ggen))
    print(evaluate(model, loss_func, g1_test_gr))

    print('\nTest - Grammatical')
    print(visual_eval(model, g1_test_gr, ggen))
    print(evaluate(model, loss_func, g1_test_gr))
    print(evaluate(model, allOrNoneloss, g1_test_gr))

    print('\nTest - Ungrammatical')
    visual_eval(model, g1_test_ugr, ggen)
    print(evaluate(model, loss_func, g1_test_ugr))
    print(evaluate(model, allOrNoneloss, g1_test_ugr))

    for labels, seqs in g1_train:
        for seq in seqs:
            displayAttention(model, seq)
    return
    k=1
    T=-0.122
    # lstm: k=1, T=-0.164: 76% accuracy 'models/autosave/g1AE-1-5-1-500-lr0.001.pt'
    # gru: k=1, T=-0.122:  64% accuracy 'models/autosave/g1-gru-AE-1-5-1-500-lr0.001.pt'
    dienes_eval(model, g1_test_gr, ggen, k, T)
    dienes_eval(model, g1_test_ugr, ggen, k, T)


if __name__ == "__main__":
    main()
