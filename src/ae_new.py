# Main file

import random
import torch
from torch.nn.modules.sparse import Embedding
import grammars as g
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader
from grammar import GrammarGen, START_TOKEN, SequenceDataset, collate_batch
from torch import nn
from torch import optim
from training import fit, visual_eval, evaluate, plotHist
from losses import SequenceLoss


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
PAD_TOKEN = 0   # ugly but works for now
END_TOKEN = 2
CLIP = 0.5


class EncoderDecoder(nn.Module):

    def __init__(self, input_dim, hidden_dim, embedding_dim, intermediate_dim, n_layers, dropout, bidirectional=False):
        super(EncoderDecoder, self).__init__()

        # Vars
        self.input_dim = input_dim
        self.embedding_dim = embedding_dim
        self.output_dim = input_dim
        self.encoder_dim = intermediate_dim
        self.decoder_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.hidden_dim = hidden_dim

        self.n_layers = n_layers
        self.bidirectional = bidirectional

        # Layers
        self.embed = nn.Embedding(self.input_dim, self.embedding_dim)

        self.encoder = nn.Linear(self.embedding_dim, self.encoder_dim)

        self.rnn = nn.GRU(self.encoder_dim, self.hidden_dim,
                          n_layers, batch_first=True, bidirectional=self.bidirectional)

        self.decoder = nn.Linear(self.decoder_dim, self.decoder_dim)

        self.out = nn.Linear(self.decoder_dim, self.output_dim)

        self.dropout = nn.Dropout(dropout)

        self.activation = nn.ReLU()

    def forward(self, input, hidden):

        embedded = self.dropout(self.embed(input))

        # embedded = [embedded_dim]

        encoded = self.dropout(self.activation(self.encoder(embedded)))

        # embedded = [encoded_dim]

        encoded = encoded.unsqueeze(0).unsqueeze(0)

        # encoded = [1,1,encoded_dim]

        if hidden is None:
            compressed, hidden = self.rnn(encoded)
        else:
            compressed, hidden = self.rnn(encoded, hidden)

        # compressed = [1, 1, hidden_dim * directions]
        # hidden = [n_layers * directions, 1, hidden_dim]

        decoded = self.dropout(self.activation(self.decoder(compressed)))

        # decoded = [1, 1, decoded_dim]

        output = self.out(decoded).squeeze()

        # output = [output_dim]

        return output, hidden


class AutoEncoder(nn.Module):
    """
    Element for Element prediction:
    Beginning:
    prediction[0] = seq[0] = START
    hidden[0] = 0

    For t = 1..len(seq):
    prediction[t] = Decoder(RNN(Encoder(seq[t-1]), hidden[t-1]))
    """

    def __init__(self, endecoder):
        super(AutoEncoder, self).__init__()

        self.endecoder = endecoder

    def forward(self, labels, seqs, teacher_forcing_ratio=0.5):

        # seqs = [(bs), seq_len]
        # autoencoder <- don't need labels for teacher_forcing
        bs = len(seqs)
        vocab_size = self.endecoder.output_dim

        # Vector to store outputs
        outputs = []

        # First input to decoder is start sequence token
        inputs = torch.tensor([START_TOKEN] * bs)

        # inputs = [bs]

        for b in range(bs):
            input = inputs[b]
            seq = seqs[b]
            hidden = None
            seq_len = len(seq)

            # input = [1]
            # seq = [seq_len]

            seq_out = torch.zeros((seq_len - 1, vocab_size))

            for t in range(seq_len - 1):

                # Get prediction
                output, hidden = self.endecoder(input, hidden)

                # output = [vocab_size]
                # hidden = [hidden_dim]

                # Save output
                seq_out[t] = output

                # Teacher forcing
                input = seq[t]

            outputs.append(seq_out)

        return outputs


def init_weights(m):
    for _, param in m.named_parameters():
        nn.init.uniform_(param.data, -0.08, 0.08)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model(input_dim, embedding_dim, hidden_dim, intermediate_dim, n_layers, dropout, lr, bidirectional=False):

    endecoder = EncoderDecoder(input_dim, hidden_dim, embedding_dim,
                               intermediate_dim, n_layers, dropout, bidirectional)

    model = AutoEncoder(endecoder)
    print(model.apply(init_weights))
    print(f'The model has {count_parameters(model):,} trainable parameters')
    return model, optim.AdamW(model.parameters(), lr=lr)


def applyOnParameters(model, conditions, apply_function):
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
                apply_function(param)


def reInitParameters(model, conditions):
    """
    conditions is a tuple of tuples (condition):
    ( ( keyword1 AND keyword2 AND ... ) OR ( keyword3 AND ... ) OR ... )
    a condition is multiple keywords which need to be in the parameter name
    to freeze the parameter
    """
    def init(param):
        nn.init.uniform_(param.data, -0.08, 0.08)
    applyOnParameters(model, conditions, init)


def freezeParameters(model, conditions):
    """
    conditions is a tuple of tuples (condition):
    ( ( keyword1 AND keyword2 AND ... ) OR ( keyword3 AND ... ) OR ... )
    a condition is multiple keywords which need to be in the parameter name
    to freeze the parameter
    """
    def freeze(param):
        param.requires_grad = False
    applyOnParameters(model, conditions, freeze)


def unfreezeParameters(model, conditions):
    """
    conditions is a tuple of tuples (condition):
    ( ( keyword1 AND keyword2 AND ... ) OR ( keyword3 AND ... ) OR ... )
    a condition is multiple keywords which need to be in the parameter name
    to freeze the parameter
    """
    def unfreeze(param):
        param.requires_grad = True
    applyOnParameters(model, conditions, unfreeze)
