# File to load and test models

from losses import allOrNoneloss, lossBasedAllorNone
import torch
import grammars as g
from ae_gru import SequenceLoss, evaluate, get_model, visual_eval
from grammar import GrammarGen, DataLoader, SequenceDataset, collate_batch
from training import dienes_eval, generic_visual_eval


def main():

    LOADNAME = 'models/autosave/g1-gru-AE-1-5-1-600-lr0.001.pt'
    bs = 4
    lr = 0.0001                         # Learning rate
    use_embedding = True                # Embedding Yes/No
    bidirectional = True                # bidirectional lstm layer Yes/o
    hidden_dim = 5                      # Lstm Neurons
    intermediate_dim = 600              # Intermediate Layer Neurons
    n_layers = 1                        # Lstm Layers
    dropout = 0.5
    grammaticality_bias = 0
    punishment = 0
    ggen = GrammarGen(g.g1())
    loss_func = SequenceLoss(
        ggen, grammaticality_bias=grammaticality_bias, punishment=punishment)

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
    loss = evaluate(model, loss_func, g1_test_gr)
    print(loss)
    print(evaluate(model, allOrNoneloss, g1_test_gr))

    lossBased_lossfunc = lossBasedAllorNone( loss * 2 )

    generic_visual_eval(model, lossBased_lossfunc, g1_test_gr)

    print('\nTest - Ungrammatical')
    visual_eval(model, g1_test_ugr, ggen)
    print(evaluate(model, loss_func, g1_test_ugr))
    print(evaluate(model, allOrNoneloss, g1_test_ugr))

    generic_visual_eval(model, lossBased_lossfunc, g1_test_ugr)

    k=1
    T=-0.164
    # k=1, T=-0.164: 76% accuracy
    dienes_eval(model, g1_test_gr, ggen, k, T)
    dienes_eval(model, g1_test_ugr, ggen, k, T)


if __name__ == "__main__":
    main()
