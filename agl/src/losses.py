# Losses

import torch
from grammar import GrammarGen
from torch import nn

END_TOKEN = 2


def softmax(x):
    return x.exp() / x.exp().sum(-1).unsqueeze(-1)


class SequenceLoss():

    def __init__(self, grammarGen: GrammarGen, ignore_index=-1, grammaticality_bias=0.5, punishment=1):
        self.ggen = grammarGen
        self.gbias = grammaticality_bias
        self.number_grammar = grammarGen.number_grammar
        self.punishment = punishment
        self.ignore_index = ignore_index

        self.CEloss = nn.CrossEntropyLoss(ignore_index=ignore_index)

        self.init_grammaticalityMatrix()

    def init_grammaticalityMatrix(self):

        # Grammaticality Matrix is a Matrix of ones, in which
        # only the entries are 0 which are in the grammar
        vocab_size = len(self.ggen)
        self.grammaticalityMatrix = torch.ones((vocab_size, vocab_size))

        for stimA, stimBs in self.number_grammar.items():
            for stimB in stimBs:
                self.grammaticalityMatrix[stimA, stimB] = 0

        self.grammaticality_indices = self.grammaticalityMatrix == 1

    def __call__(self, outputs: torch.tensor, labels: torch.tensor):

        bs = len(outputs)

        CEloss = torch.zeros(bs)
        GRloss = torch.zeros(bs)

        for b in range(bs):

            # Cross Entropy loss
            CEOutput = outputs[b]
            # Cut off starting token and conver to long
            CELabel = labels[b][1:].type(torch.long)

            CEloss[b] = self.CEloss(CEOutput, CELabel)
            if self.gbias == 0:
                continue

            # Grammaticality
            output = softmax(outputs[b])
            label = labels[b][1:]

            seq_length = output.size(0)

            seq_loss = torch.empty(seq_length)

            # Judge pairwise grammaticality of prediction A -> prediction B in stimulus sequence
            for i in range(seq_length - 1):

                if label[i] == self.ignore_index:
                    continue

                # Get rid of the gradient (no_grad did not work :( )
                predA = torch.tensor(output[i].unsqueeze(
                    0).transpose(0, 1).tolist())
                predB = output[i + 1].unsqueeze(0)
                transitionMatrix = torch.matmul(predA, predB)

                errorvalues = transitionMatrix[self.grammaticality_indices]

                seq_loss[i] = (errorvalues.sum() * self.punishment).pow(2)

            GRloss[b] = seq_loss.sum()

        if self.gbias == 0:
            return CEloss.mean()

        return GRloss.mean() * self.gbias + CEloss.mean() * (1 - self.gbias)


def cutStartAndEndToken(seq):
    ret = []
    for stim in seq:
        if stim == END_TOKEN:
            break
        ret.append(stim)
    return ret


def allOrNoneloss(output, labels):
    ret = []
    for b, seq in enumerate(labels):
        prediction = output[b].argmax(-1)
        trgtlist = seq.tolist()[1:-1]
        predlist = cutStartAndEndToken(prediction.tolist())
        ret.append(not predlist == trgtlist)
    return torch.tensor(sum(ret))


def sigmoid(x, k, T):
    return 1 / (1 + (-k*x-T).exp())


def one_hot(seqs, vocabsize):
    eye = torch.eye(vocabsize)
    return [eye[seq.type(torch.long)] for seq in seqs]


def dienesLoss(outputs, labels, k=1, T=0):

    bs = len(outputs)

    dLoss = torch.zeros(bs)
    cos = nn.CosineSimilarity(0)
    inputs = one_hot(labels, 8)

    for b, input in enumerate(inputs):
        flat_out = outputs[b].flatten()
        flat_inp = input[1:].flatten()

        dLoss[b] = sigmoid(cos(flat_out, flat_inp), k, T)

    return (torch.ones(bs) - dLoss).mean()


class lossBasedAllorNone():

    def __init__(self, threshhold, ignore_index=-1):

        self.CEloss = nn.CrossEntropyLoss(ignore_index=ignore_index)
        self.threshhold = threshhold


    def __call__(self, outputs, labels):

        bs = len(outputs)

        CEloss = torch.zeros(bs)

        for b in range(bs):

            # Cross Entropy loss
            CEOutput = outputs[b]
            # Cut off starting token and conver to long
            CELabel = labels[b][1:].type(torch.long)

            CEloss[b] = self.CEloss(CEOutput, CELabel)

        # Convert treshold to 1
        below = CEloss < self.threshhold
        CEloss[below] = 1
        CEloss[~below] = 0

        return CEloss.sum()
