# Simulate the experiment

from autoencoder import SequenceLoss, evaluate, get_model, visual_eval
from grammar import *


def main():
    LOADNAME = 'models/aEv1-100-5-3.pt'
    bs = 3
    epochs = 5000
    lr = 0.0001                         # Learning rate
    teacher_forcing_ratio = 0.5         # How much teacher_forcing
    use_embedding = True                # Embedding Yes/No
    hidden_dim = 5                      # Lstm Neurons
    intermediate_dim = 100              # Intermediate Layer Neurons
    n_layers = 3                        # Lstm Layers
    dropout = 0.5
    ggen = GrammarGen()                 # Grammar
    input_dim = len(ggen)
    grammaticality_bias = 0
    punishment = 0
    loss_func = SequenceLoss(
        ggen, grammaticality_bias=grammaticality_bias, punishment=punishment)

    # Note: BATCH IS IN FIRST DIMENSION
    # Train
    train_seqs = ggen.stim2seqs(get_trainstimuliSequence())
    train_ds = SequenceDataset(train_seqs)
    train_dl = DataLoader(train_ds, batch_size=bs,
                          shuffle=True, collate_fn=collate_batch)

    # Validation
    valid_seqs = ggen.generate(8)
    valid_ds = SequenceDataset(valid_seqs)
    valid_dl = DataLoader(valid_ds, batch_size=bs, collate_fn=collate_batch)

    # Test - Correct
    test_seqs = ggen.stim2seqs(get_correctStimuliSequence())
    test_ds = SequenceDataset(test_seqs)
    test_dl = DataLoader(test_ds, batch_size=bs, collate_fn=collate_batch)

    # Test - Incorrect
    test_incorrect_seqs = ggen.stim2seqs(get_incorrectStimuliSequence())
    test_incorrect_ds = SequenceDataset(test_incorrect_seqs)
    test_incorrect_dl = DataLoader(
        test_incorrect_ds, batch_size=bs * 2, collate_fn=collate_batch)

    # Load Model
    model, _ = get_model(input_dim, hidden_dim, intermediate_dim,
                         n_layers, lr, dropout, use_embedding)
    model.load_state_dict(torch.load(LOADNAME))


if __name__ == '__main__':
    main()
