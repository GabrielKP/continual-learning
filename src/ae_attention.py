# Main file

import random
import torch
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


class Encoder(nn.Module):

    def __init__(self, input_dim, hidden_dim, intermediate_dim, n_layers, dropout, embedding=True, bidirectional=False):
        super(Encoder, self).__init__()

        # Vars
        self.hidden_dim = hidden_dim
        self.input_dim = input_dim
        self.embedding_dim = input_dim
        self.n_layers = n_layers
        self.intermediate_dim = intermediate_dim
        self.bidirectional = bidirectional

        # Layers
        self.embed = nn.Embedding(self.input_dim, self.embedding_dim)
        if not embedding:
            self.embed.weight.data = torch.eye(input_dim)

        self.dropout = nn.Dropout(dropout)

        self.fc_one = nn.Linear(self.embedding_dim, self.intermediate_dim)

        self.ac_one = nn.ReLU()

        self.gru = nn.GRU(self.intermediate_dim, self.hidden_dim,
                          n_layers, batch_first=True, bidirectional=self.bidirectional)

        # Merge bidirectional last hidden output to one since decoder not bidirectional
        self.fc_hidden = nn.Linear(self.hidden_dim * 2, self.hidden_dim)

    def forward(self, seqs):

        # Handle sequences separately

        outputs = []
        hiddens = []

        for seq in seqs:

            embed = self.dropout(self.embed(seq))

            intermediate = self.dropout(self.ac_one(self.fc_one(embed)))

            output, hidden = self.gru(intermediate.unsqueeze(0))

            outputs.append(output.squeeze())

            hidden = hidden.squeeze()
            hiddens.append(torch.tanh(self.fc_hidden(
                torch.cat((hidden[-2, :], hidden[-1, :]), dim=0))))

        hiddens = torch.stack(hiddens)

        # hiddens = [bs, hidden_dim]
        # outputs = [(bs), seq_len, hidden_dim * 2]

        return outputs, hiddens


class Attention(nn.Module):

    def __init__(self, hidden_dim):
        super(Attention, self).__init__()

        # Takes encoder_hidden_dim * 2 (because of bidirectional) + decoder_hidden_dim
        # to decoder_hidden_dim
        self.attn = nn.Linear(hidden_dim * 3, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, hidden, encoder_output):

        seq_len = encoder_output.shape[0]

        # hidden = [hidden_dim]
        # encoder_output = [seq_len, hidden_dim * 2]

        hidden = hidden.repeat(seq_len, 1)

        # hidden = [seq_len, hidden_dim]

        energy = torch.tanh(
            self.attn(torch.cat((hidden, encoder_output), dim=1)))

        # energy = [seq_len, hidden_dim]

        attention = self.v(energy).squeeze()

        # attention = [seq_len]

        return F.softmax(attention, dim=0)


class Decoder(nn.Module):

    def __init__(self, output_dim, hidden_dim, intermediate_dim, n_layers, dropout, attention, embedding=True):
        super(Decoder, self).__init__()

        # Vars
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.embedding_dim = output_dim
        self.n_layers = n_layers
        self.intermediate_dim = intermediate_dim

        self.attention = attention

        # Layers
        self.embed = nn.Embedding(self.output_dim, self.embedding_dim)
        if not embedding:
            self.embed.weight.data = torch.eye(self.embedding_dim)

        self.gru = nn.GRU(self.hidden_dim * 2 + self.embedding_dim, self.hidden_dim,
                          self.n_layers, batch_first=True)

        # hidden_dim * 2 (from attention weighted encoder output) + hidden_dim (from gru layer)
        # + embed_dim (from embedded input)
        self.fc_one = nn.Linear(self.hidden_dim * 3 +
                                self.embedding_dim, self.intermediate_dim)

        self.fc_out = nn.Linear(self.intermediate_dim, self.output_dim)

        self.ac_one = nn.ReLU()

        self.dropout = nn.Dropout(dropout)

    def forward(self, nInput, hidden, encoder_output):

        # nInput = 1
        # hidden = [hidden_dim]
        # encoder_output = [seq_len, hidden_dim * 2]

        embed = self.dropout(self.embed(nInput))

        # embed = [embedding_dim]

        embed = embed.unsqueeze(0)

        # embed = [1, embedding_dim]

        a = self.attention(hidden, encoder_output)

        # a = [seq_len]

        a = a.unsqueeze(0)

        # a = [1, seq_len]

        weighted = torch.matmul(a, encoder_output)

        # weighted = [1, hidden_dim * 2]

        rnn_input = torch.cat((embed, weighted), dim=1)

        # Predicting one word: seq_len = 1, need to unsqueeze to have one batch
        # rnn_input = [1, embedding_dim + hidden_dim * 2]

        output, hidden = self.gru(rnn_input.unsqueeze(
            0), hidden.unsqueeze(0).unsqueeze(0))

        # output = [1, 1, hidden_dim]
        # hidden = [1, 1, hidden_dim]

        # intermediate = self.dropout(self.ac_one(self.fc_one(output.squeeze())))

        output = output.squeeze(0)
        hidden = hidden.squeeze(0)

        # output = [1, hidden_dim]
        # hidden = [1, hidden_dim]

        intermediate = self.ac_one(self.fc_one(torch.cat((output, weighted, embed), dim=1)))

        # intermediate = [1, intermediate_dim]

        prediction = self.fc_out(intermediate)

        # prediction = [1, output_dim]

        return prediction, hidden.squeeze()


class AutoEncoder(nn.Module):
    def __init__(self, encoder, decoder):
        super(AutoEncoder, self).__init__()

        self.encoder = encoder
        self.decoder = decoder

    def forward(self, labels, seqs, teacher_forcing_ratio=0.5):

        # seqs = [(bs), seq_len]
        # autoencoder <- don't need labels for teacher_forcing
        bs = len(seqs)
        vocab_size = self.decoder.output_dim

        # Vector to store outputs
        outputs = []

        # Encoder run
        encoder_outputs, hiddens = self.encoder(seqs)

        # encoder_outputs = [(bs), seq_len, hidden_dim * 2]
        # hiddens = [bs, hidden_dim]

        # First input to decoder is start sequence token
        nInputs = torch.tensor([START_TOKEN] * bs)

        # nInputs = [bs]

        # Decide teacherforcing for entire batch
        teacher_forcing = random.random() < teacher_forcing_ratio

        for b in range(bs):
            nInput = nInputs[b]
            hidden = hiddens[b]
            encoder_output = encoder_outputs[b]
            seq = seqs[b]

            # nInput = [1]
            # hidden = [hidden_dim]
            # encoder_output = [seq_len, hidden_dim * 2]
            # seq = [seq_len]

            seq_len = len(seq)
            seq_out = torch.zeros((seq_len - 1, vocab_size))

            for t in range(1, seq_len):

                # Get prediction
                output, hidden = self.decoder(nInput, hidden, encoder_output)

                # Save output
                seq_out[t-1] = output

                # Teacher forcing
                nInput = seq[t] if teacher_forcing else output.argmax(
                    -1).squeeze()

            outputs.append(seq_out)

        return outputs


def init_weights(m):
    for _, param in m.named_parameters():
        nn.init.uniform_(param.data, -0.08, 0.08)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_model(input_dim, hidden_dim, intermediate_dim, n_layers, lr, dropout, use_embedding=True, bidirectional=False):

    attention = Attention(hidden_dim)
    encoder = Encoder(input_dim, hidden_dim, intermediate_dim,
                      n_layers, dropout, use_embedding, bidirectional)
    decoder = Decoder(input_dim, hidden_dim, intermediate_dim,
                      n_layers, dropout, attention, use_embedding)

    model = AutoEncoder(encoder, decoder)
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
