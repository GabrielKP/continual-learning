# Training functions

from losses import allOrNoneloss, cutStartAndEndToken
import torch
import time
import numpy as np
import matplotlib.pyplot as plt

CLIP = 1


def loss_batch(model, loss_func, labels, seqs, teacher_forcing_ratio=0.5, opt=None):
    # loss function gets padded sequences -> autoencoder
    labels = seqs

    # Get model output
    output = model(labels, seqs, teacher_forcing_ratio)

    # Compute loss
    loss = loss_func(output, labels)

    if opt is not None:
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), CLIP)
        opt.step()
        opt.zero_grad()

    return loss.item(), len(labels)


def train(model, train_dl, loss_func, opt, teacher_forcing_ratio):
    """ Trains 1 epoch of the model, returns loss for train set"""

    model.train()

    epoch_loss = 0
    epoch_num_seqs = 0

    for labels, seqs in train_dl:
        batch_loss, batch_num_seqs = loss_batch(
            model, loss_func, labels, seqs, teacher_forcing_ratio, opt)
        epoch_loss += batch_loss * batch_num_seqs
        epoch_num_seqs += batch_num_seqs

    return epoch_loss / epoch_num_seqs


def evaluate(model, loss_func, test_dl):
    model.eval()
    with torch.no_grad():
        losses, nums = zip(
            *[loss_batch(model, loss_func, labels, seqs, teacher_forcing_ratio=0) for labels, seqs in test_dl]
        )
        if loss_func == allOrNoneloss:
            return np.sum(losses)
        return np.sum(np.multiply(losses, nums)) / np.sum(nums)


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = elapsed_time - (elapsed_mins * 60)
    return elapsed_mins, elapsed_secs


def fit(epochs, model, loss_func, opt, train_dl, valid_dl, teacher_forcing_ratio=0.5, FILENAME='aa', check_dls=None, stepsize=5):
    """
    Fits model on train data, printing val and train loss

    check_dls : list of tuples: (dataloader, loss_function)
        when given as argument, in every epoch every dataloader
        will be evaluted with its loss function.
        If given fit returns additional list with tensors containing
        evaluation results for every epoch
    stepsize : int
        how often check_dls will be evaluated
    """

    best_val_loss = float('inf')
    hist_valid = torch.empty(epochs)
    hist_train = torch.empty(epochs)
    if check_dls is not None:
        hist_check = []
        for _ in range(len(check_dls)):
            hist_check.append(torch.empty(epochs))

    for epoch in range(epochs):

        start_time = time.time()

        train_loss = train(model, train_dl, loss_func,
                           opt, teacher_forcing_ratio)
        valid_loss = evaluate(model, loss_func, valid_dl)

        end_time = time.time()

        if valid_loss < best_val_loss:
            best_val_loss = valid_loss
            torch.save(model.state_dict(), FILENAME)

        epoch_mins, epoch_secs = epoch_time(start_time, end_time)

        hist_train[epoch] = train_loss
        hist_valid[epoch] = valid_loss
        print(f'Epoch: {epoch+1:03} | Time: {epoch_mins}m {epoch_secs:.2}s')
        print(f'\tTrain Loss: {train_loss:.5f} |  Val. Loss: {valid_loss:.5f}')

        if check_dls is not None:
            if epoch % stepsize == 0:
                for i, (dl, dl_loss_func) in enumerate(check_dls):
                    hist_check[i][epoch] = evaluate(model, dl_loss_func, dl)

    if check_dls is None:
        return hist_train, hist_valid
    return hist_train, hist_valid, hist_check


def plotHist(*hist_tuples, stepsize=5):
    """
    example: plotHist( ( hist_valid1, 'Normal', ), ( hist_valid2, 'Shifted', ) )
    """
    labels = []
    #colors = ['blue', 'or']
    for history, label in hist_tuples:
        xvals = range(0, history.size(0), stepsize)
        plt.plot(xvals, history[xvals])
        labels.append(label)
    plt.legend(labels)
    plt.show()


def plotMultipleHist(hist_tensors, labels, stepsize=5, sublabels=None, ylims=None, title=None, path=None):
    """
    hist_tensors expected in following form:
    ( [label1_plotdata1, label1_plotdata2, ...], [ label2_plotdata1, label2_plotdata2, ... ], ...)
    """
    assert len(hist_tensors) == len(
        labels), "labels and different plots do not match"

    n_figs = len(hist_tensors[0])
    n_difflines = len(hist_tensors)

    fig = plt.figure()
    for x in range(n_figs):
        ax = fig.add_subplot(n_figs // 2, 2, x + 1)
        for m in range(n_difflines):
            xvals = range(0, hist_tensors[m][x].size(0), stepsize)
            ax.plot(xvals, hist_tensors[m][x][xvals])

        if ylims is not None and (x + 1) % 2 == 0:
            a, b = ylims[x//2]
            yticks = list(range(a, b, (b - a)//3)) + [b]
            # make sure last ytick does not overlap
            if yticks[-1] - yticks[-2] < 0.1 * b:
                yticks.pop(-2)
            ax.set_ylim((a, b))
            ax.set_yticks(yticks)

        if sublabels is not None and x % 2 == 0:
            ax.set_title(sublabels[x//2])

    ax.legend(labels)
    if title is not None:
        plt.suptitle(title)
    fig.tight_layout()
    if path is not None:
        plt.savefig(path)
    plt.show()


def visual_eval(model, test_dl, ggen=None):
    ret = []
    i = 0
    model.eval()
    with torch.no_grad():
        for labels, seqs in test_dl:
            output = model(labels, seqs, teacher_forcing_ratio=0)

            for b, seq in enumerate(seqs):
                prediction = output[b].argmax(-1)
                trgtlist = seq.tolist()[1:-1]
                predlist = cutStartAndEndToken(prediction.tolist())
                same = trgtlist == predlist
                if ggen is None:
                    print(f'Same:{same:2} Truth: {trgtlist} - Pred: {predlist}')
                else:
                    gramm = ggen.isGrammatical([predlist])[0]
                    trgtlist = ggen.seqs2stim([trgtlist])[0]
                    predlist = ggen.seqs2stim([predlist])[0]
                    print(f'Same:{same:2} Gramm:{gramm:2} Truth: {trgtlist} - Pred: {predlist}')
                if not same:
                    ret.append(i)
                i += 1
    return ret
