# Specific instances of grammars, their train and testsets
# gr = grammatical, ugr = ungrammatical
import grammar as gr


def g0_dls(bs=4):
    """
    Returns tuple of DataLoaders and amount of
    stimuli in g0: (train, test_gr, test_ugr, len( g0 ) )
    """

    ggen = gr.GrammarGen(g0())

    train_seqs = ggen.stim2seqs(g0_train())
    train_ds = gr.SequenceDataset(train_seqs)
    train_dl = gr.DataLoader(train_ds, batch_size=bs,
                             shuffle=True, collate_fn=gr.collate_batch)

    # test gr
    test_gr_seqs = ggen.stim2seqs(g0_test_gr())
    test_gr_ds = gr.SequenceDataset(test_gr_seqs)
    test_gr_dl = gr.DataLoader(
        test_gr_ds, batch_size=bs, collate_fn=gr.collate_batch)

    # test ugr
    test_ugr_seqs = ggen.stim2seqs(g0_test_ugr())
    test_ugr_ds = gr.SequenceDataset(test_ugr_seqs)
    test_ugr_dl = gr.DataLoader(
        test_ugr_ds, batch_size=bs * 2, collate_fn=gr.collate_batch)

    return train_dl, test_gr_dl, test_ugr_dl, len(ggen)


def g1_dls(bs=4):
    """
    Returns tuple of DataLoaders and the amount of
    stimuli in g1: (train, test_gr, test_ugr, len( g1 ) )
    """

    ggen = gr.GrammarGen(g1())

    train_seqs = ggen.stim2seqs(g1_train())
    train_ds = gr.SequenceDataset(train_seqs)
    train_dl = gr.DataLoader(train_ds, batch_size=bs,
                             shuffle=True, collate_fn=gr.collate_batch)

    # test gr
    test_gr_seqs = ggen.stim2seqs(g1_test_gr())
    test_gr_ds = gr.SequenceDataset(test_gr_seqs)
    test_gr_dl = gr.DataLoader(
        test_gr_ds, batch_size=bs, collate_fn=gr.collate_batch)

    # test ugr
    test_ugr_seqs = ggen.stim2seqs(g1_test_ugr_balanced())
    test_ugr_ds = gr.SequenceDataset(test_ugr_seqs)
    test_ugr_dl = gr.DataLoader(
        test_ugr_ds, batch_size=bs * 2, collate_fn=gr.collate_batch)

    return train_dl, test_gr_dl, test_ugr_dl, len(ggen)


def g0():
    """Returns grammar structure for g0"""
    return {
        'START': ['A'],
        'A': ['D', 'C1'],
        'C1': ['G1', 'F'],
        'G1': ['F'],
        'D': ['C1'],
        'F': ['C2', 'END'],
        'C2': ['END', 'G2'],
        'G2': ['END']
    }


def g0_train():
    """Returns training set with Sequences for g0"""
    return [
        ['A', 'C', 'F'],
        ['A', 'C', 'F', 'C', 'G'],
        ['A', 'C', 'G', 'F'],
        ['A', 'C', 'G', 'F', 'C', 'G'],
        ['A', 'D', 'C', 'F'],
        ['A', 'D', 'C', 'F', 'C'],
        ['A', 'D', 'C', 'F', 'C', 'G'],
        ['A', 'D', 'C', 'G', 'F', 'C', 'G'],
    ]


def g0_test_gr():
    """Returns grammatical test Sequences for g0"""
    return [
        ['A', 'C', 'F', 'C', 'G'],
        ['A', 'D', 'C', 'F', 'C'],
        ['A', 'C', 'G', 'F', 'C'],
        ['A', 'D', 'C', 'G', 'F'],
    ]


def g0_test_ugr():
    """Returns ungrammatical test Sequences for g0"""
    return [
        ['A', 'D', 'C', 'F', 'G'],
        ['A', 'D', 'F', 'C', 'G'],
        ['A', 'D', 'G', 'C', 'F'],
        ['A', 'D', 'G', 'F', 'C'],
        ['A', 'G', 'C', 'F', 'G'],
        ['A', 'G', 'F', 'G', 'C'],
        ['A', 'G', 'D', 'C', 'F'],
        ['A', 'G', 'F', 'D', 'C'],
    ]


def g1():
    """Returns grammar structure for g1"""
    return {
        # 'START': ['#1'],
        # '#1': ['W1','N1'],
        'START': ['W1', 'N1'],
        'W1': ['S1', 'S2', 'W2'],
        'N1': ['P1', 'P2', 'N2'],
        'S1': ['S2', ],
        'S2': ['W2', ],
        'W2': ['S3', 'Z'],
        'P1': ['P2', ],
        'P2': ['N2', ],
        'N2': ['P3', 'Z'],
        'S3': ['P1', 'P2', 'N2'],
        # 'Z': ['#2',],
        'Z': ['END', ],
        'P3': ['S3', 'Z'],
        # '#2': ['END']
    }


def g1_train():
    """Returns train Sequences for g1"""
    return [
        ['W', 'S', 'W', 'Z', ],
        ['W', 'S', 'S', 'W', 'Z', ],
        ['W', 'S', 'W', 'S', 'N', 'P', 'Z', ],
        ['W', 'S', 'W', 'S', 'P', 'N', 'Z', ],
        ['W', 'W', 'S', 'P', 'N', 'P', 'Z', ],
        ['W', 'W', 'S', 'N', 'P', 'S', 'N', 'Z', ],
        ['W', 'W', 'S', 'P', 'N', 'Z', ],
        ['W', 'W', 'S', 'N', 'Z', ],
        ['W', 'S', 'S', 'W', 'S', 'N', 'Z', ],
        ['W', 'S', 'W', 'S', 'N', 'Z', ],
        ['N', 'P', 'P', 'N', 'Z', ],
        ['N', 'N', 'Z', ],
        ['N', 'P', 'P', 'N', 'P', 'Z', ],
        ['N', 'N', 'P', 'S', 'P', 'N', 'P', 'Z', ],
        ['N', 'P', 'N', 'P', 'S', 'N', 'P', 'Z', ],
        ['N', 'N', 'P', 'S', 'P', 'P', 'N', 'Z', ],
        ['N', 'P', 'P', 'N', 'P', 'S', 'N', 'Z', ],
        ['N', 'N', 'P', 'S', 'N', 'Z', ],
    ]


def g1_test_gr():
    """Returns grammatical test Sequences for g1"""
    return [
        ['W', 'W', 'Z', ],
        ['W', 'W', 'S', 'N', 'P', 'Z', ],
        ['W', 'S', 'S', 'W', 'S', 'N', 'P', 'Z', ],
        ['W', 'W', 'S', 'P', 'P', 'N', 'P', 'Z', ],
        ['W', 'W', 'S', 'P', 'N', 'Z', ],
        ['W', 'S', 'W', 'S', 'N', 'Z', ],
        ['W', 'S', 'W', 'S', 'N', 'P', 'Z', ],
        ['W', 'S', 'S', 'W', 'S', 'P', 'N', 'Z', ],
        ['W', 'W', 'S', 'P', 'P', 'N', 'Z', ],
        ['W', 'S', 'W', 'S', 'P', 'P', 'N', 'Z', ],
        ['N', 'P', 'N', 'Z', ],
        ['N', 'N', 'P', 'S', 'N', 'P', 'Z', ],
        ['N', 'N', 'P', 'Z', ],
        ['N', 'P', 'N', 'P', 'S', 'N', 'Z', ],
        ['N', 'N', 'P', 'S', 'P', 'N', 'Z', ],
        ['N', 'P', 'N', 'P', 'S', 'P', 'N', 'Z', ],
        ['W', 'W', 'S', 'P', 'P', 'N', 'P', 'Z', ],
    ]

def g1_test_ugr_balanced():
    """Returns same amount of ungrammatical test Sequences for g1 as g1_test_gr"""
    return [
        ['W', 'S', 'Z', ],
        ['W', 'W', 'S', 'P', 'W', 'S', 'N', 'Z', ],
        ['W', 'S', 'W', 'P', 'P', 'N', 'Z', ],
        ['W', 'S', 'W', 'S', 'Z', ],
        ['W', 'N', 'P', 'S', 'P', 'P', 'N', 'Z', ],
        ['N', 'P', 'N', 'S', 'P', 'N', 'Z', ],
        ['N', 'N', 'Z', 'S', 'P', 'S', 'N', 'Z', ],
        ['N', 'W', 'S', 'W', 'Z', ],
        ['N', 'P', 'N', 'W', 'Z', ],  # NPP, end, NPL start
        ['W', 'S', 'S', 'N', 'Z', ],
        ['W', 'S', 'P', 'S', 'P', 'N', 'Z', ],
        ['W', 'W', 'S', 'P', 'P', 'Z', ],
        ['W', 'S', 'S', 'N', 'P', 'N', 'P', 'Z', ],
        ['N', 'P', 'P', 'S', 'N', 'Z', ],
        ['N', 'P', 'S', 'S', 'P', 'S', 'N', 'Z', ],
        ['N', 'N', 'P', 'S', 'N', 'N', 'P', 'Z', ],
        ['N', 'P', 'N', 'P', 'N', 'Z', ],
    ]


def g1_test_ugr():
    """Returns ungrammatical test Sequences for g1"""
    return [
        ['W', 'Z', 'W', 'Z', ],
        ['W', 'S', 'W', 'N', 'P', 'Z', ],
        ['W', 'W', 'S', 'N', 'W', 'Z', ],
        ['W', 'S', 'W', 'N', 'P', 'S', 'N', 'Z', ],
        ['N', 'Z', 'P', 'P', 'N', 'Z', ],
        ['N', 'N', 'S', 'P', 'N', 'P', 'Z', ],
        ['N', 'S', 'P', 'P', 'N', 'Z', ],
        ['N', 'N', 'W', 'S', 'P', 'P', 'N', 'Z', ],
        ['W', 'S', 'Z', ],
        ['W', 'W', 'S', 'P', 'W', 'S', 'N', 'Z', ],
        ['W', 'S', 'W', 'P', 'P', 'N', 'Z', ],
        ['W', 'S', 'W', 'S', 'Z', ],
        ['W', 'N', 'P', 'S', 'P', 'P', 'N', 'Z', ],
        ['N', 'P', 'N', 'S', 'P', 'N', 'Z', ],
        ['N', 'N', 'Z', 'S', 'P', 'S', 'N', 'Z', ],
        ['N', 'W', 'S', 'W', 'Z', ],
        ['N', 'P', 'N', 'W', 'Z', ],  # NPP, end, NPL start
        ['W', 'S', 'S', 'N', 'Z', ],
        ['W', 'S', 'P', 'S', 'P', 'N', 'Z', ],
        ['W', 'W', 'S', 'P', 'P', 'Z', ],
        ['W', 'S', 'S', 'N', 'P', 'N', 'P', 'Z', ],
        ['N', 'P', 'P', 'S', 'N', 'Z', ],
        ['N', 'P', 'S', 'S', 'P', 'S', 'N', 'Z', ],
        ['N', 'N', 'P', 'S', 'N', 'N', 'P', 'Z', ],
        ['N', 'P', 'N', 'P', 'N', 'Z', ],
        ['N', 'P', 'P', 'Z', ],
        ['N', 'P', 'S', 'P', 'Z', ],
        ['N', 'P', 'N', 'P', 'S', 'S', 'W', 'Z', ],
        ['N', 'P', 'S', 'P', 'P', 'N', 'Z', ],
        ['W', 'W', 'S', 'N', 'N', 'Z', ],
        ['W', 'S', 'W', 'S', 'S', 'N', 'Z', ],
        ['W', 'S', 'N', 'Z', ],
        ['W', 'S', 'P', 'P', 'N', 'Z', ],
        ['W', 'S', 'P', 'N', 'P', 'P', 'N', 'Z', ],
    ]


def g1_train_x():
    """Returns train Sequences for g1"""
    return [
        ['#', 'W', 'S', 'W', 'Z', '#'],
        ['#', 'W', 'S', 'S', 'W', 'Z', '#'],
        ['#', 'W', 'S', 'W', 'S', 'N', 'P', 'Z', '#'],
        ['#', 'W', 'S', 'W', 'S', 'P', 'N', 'Z', '#'],
        ['#', 'W', 'W', 'S', 'P', 'N', 'P', 'Z', '#'],
        ['#', 'W', 'W', 'S', 'N', 'P', 'S', 'N', 'Z', '#'],
        ['#', 'W', 'W', 'S', 'P', 'N', 'Z', '#'],
        ['#', 'W', 'W', 'S', 'N', 'Z', '#'],
        ['#', 'W', 'S', 'S', 'W', 'S', 'N', 'Z', '#'],
        ['#', 'W', 'S', 'W', 'S', 'N', 'Z', '#'],
        ['#', 'N', 'P', 'P', 'N', 'Z', '#'],
        ['#', 'N', 'N', 'Z', '#'],
        ['#', 'N', 'P', 'P', 'N', 'P', 'Z', '#'],
        ['#', 'N', 'N', 'P', 'S', 'P', 'N', 'P', 'Z', '#'],
        ['#', 'N', 'P', 'N', 'P', 'S', 'N', 'P', 'Z', '#'],
        ['#', 'N', 'N', 'P', 'S', 'P', 'P', 'N', 'Z', '#'],
        ['#', 'N', 'P', 'P', 'N', 'P', 'S', 'N', 'Z', '#'],
        ['#', 'N', 'N', 'P', 'S', 'N', 'Z', '#'],
    ]


def g1_test_gr_x():
    """Returns grammatical test Sequences for g1"""
    return [
         ['#', 'W', 'W', 'Z', '#'],
         ['#', 'W', 'W', 'S', 'N', 'P', 'Z', '#'],
         ['#', 'W', 'S', 'S', 'W', 'S', 'N', 'P', 'Z', '#'],
         ['#', 'W', 'W', 'S', 'P', 'P', 'N', 'P', 'Z', '#'],
         ['#', 'W', 'W', 'S', 'P', 'N', 'Z', '#'],
         ['#', 'W', 'S', 'W', 'S', 'N', 'Z', '#'],
         ['#', 'W', 'S', 'W', 'S', 'N', 'P', 'Z', '#'],
         ['#', 'W', 'S', 'S', 'W', 'S', 'P', 'N', 'Z', '#'],
         ['#', 'W', 'W', 'S', 'P', 'P', 'N', 'Z', '#'],
         ['#', 'W', 'S', 'W', 'S', 'P', 'P', 'N', 'Z', '#'],
         ['#', 'N', 'P', 'N', 'Z', '#'],
         ['#', 'N', 'N', 'P', 'S', 'N', 'P', 'Z', '#'],
         ['#', 'N', 'N', 'P', 'Z', '#'],
         ['#', 'N', 'P', 'N', 'P', 'S', 'N', 'Z', '#'],
         ['#', 'N', 'N', 'P', 'S', 'P', 'N', 'Z', '#'],
         ['#', 'N', 'P', 'N', 'P', 'S', 'P', 'N', 'Z', '#'],
         ['#', 'W', 'W', 'S', 'P', 'P', 'N', 'P', 'Z', '#'],
    ]


def g1_test_ugr_x():
    """Returns ungrammatical test Sequences for g1"""
    return [
        ['#', 'W', 'Z', 'W', 'Z', '#'],
        ['#', 'W', 'S', 'W', 'N', 'P', 'Z', '#'],
        ['#', 'W', 'W', 'S', 'N', 'W', 'Z', '#'],
        ['#', 'W', 'S', 'W', 'N', 'P', 'S', 'N', 'Z', '#'],
        ['#', 'N', 'Z', 'P', 'P', 'N', 'Z', '#'],
        ['#', 'N', 'N', 'S', 'P', 'N', 'P', 'Z', '#'],
        ['#', 'N', 'S', 'P', 'P', 'N', 'Z', '#'],
        ['#', 'N', 'N', 'W', 'S', 'P', 'P', 'N', 'Z', '#'],
        ['#', 'W', 'S', 'Z', '#'],
        ['#', 'W', 'W', 'S', 'P', 'W', 'S', 'N', 'Z', '#'],
        ['#', 'W', 'S', 'W', 'P', 'P', 'N', 'Z', '#'],
        ['#', 'W', 'S', 'W', 'S', 'Z', '#'],
        ['#', 'W', 'N', 'P', 'S', 'P', 'P', 'N', 'Z', '#'],
        ['#', 'N', 'P', 'N', 'S', 'P', 'N', 'Z', '#'],
        ['#', 'N', 'N', 'Z', 'S', 'P', 'S', 'N', 'Z', '#'],
        ['#', 'N', 'W', 'S', 'W', 'Z', '#'],
        ['#', 'N', 'P', 'N', 'W', 'Z', '#'],  # NPP, end, NPL start
        ['#', 'W', 'S', 'S', 'N', 'Z', '#'],
        ['#', 'W', 'S', 'P', 'S', 'P', 'N', 'Z', '#'],
        ['#', 'W', 'W', 'S', 'P', 'P', 'Z', '#'],
        ['#', 'W', 'S', 'S', 'N', 'P', 'N', 'P', 'Z', '#'],
        ['#', 'N', 'P', 'P', 'S', 'N', 'Z', '#'],
        ['#', 'N', 'P', 'S', 'S', 'P', 'S', 'N', 'Z', '#'],
        ['#', 'N', 'N', 'P', 'S', 'N', 'N', 'P', 'Z', '#'],
        ['#', 'N', 'P', 'N', 'P', 'N', 'Z', '#'],
        ['#', 'N', 'P', 'P', 'Z', '#'],
        ['#', 'N', 'P', 'S', 'P', 'Z', '#'],
        ['#', 'N', 'P', 'N', 'P', 'S', 'S', 'W', 'Z', '#'],
        ['#', 'N', 'P', 'S', 'P', 'P', 'N', 'Z', '#'],
        ['#', 'W', 'W', 'S', 'N', 'N', 'Z', '#'],
        ['#', 'W', 'S', 'W', 'S', 'S', 'N', 'Z', '#'],
        ['#', 'W', 'S', 'N', 'Z', '#'],
        ['#', 'W', 'S', 'P', 'P', 'N', 'Z', '#'],
        ['#', 'W', 'S', 'P', 'N', 'P', 'P', 'N', 'Z', '#'],
    ]


def g2():
    """Returns grammar structure for g2"""
    return {
        'START': ['#1'],
        '#1': ['Z1', 'N1'],
        'Z1': ['Z2', 'W1'],
        'N1': ['N2', 'P2'],
        'Z2': ['S2', 'S3', 'W2'],
        'W1': ['N2', ],
        'N2': ['P1', 'P3', 'S1'],
        'P2': ['S2', 'S3', 'W2'],
        'S2': ['S3', ],
        'S3': ['W2', ],
        'W2': ['#2', ],
        'P1': ['P3', ],
        'P3': ['S1', ],
        'S1': ['Z2', ],
        '#2': ['END', ]
    }


def g2_train():
    """Returns train Sequences for g2"""
    return [
        ['#', 'Z', 'Z', 'S', 'W', '#'],
        ['#', 'Z', 'Z', 'S', 'S', 'W', '#'],
        ['#', 'Z', 'W', 'P', 'W', '#'],
        ['#', 'Z', 'W', 'N', 'S', 'W', 'P', 'W', '#'],
        ['#', 'Z', 'W', 'P', 'S', 'S', 'W', '#'],
        ['#', 'Z', 'W', 'N', 'S', 'Z', 'S', 'S', 'W', '#'],
        ['#', 'Z', 'W', 'N', 'P', 'S', 'Z', 'S', 'W', '#'],
        ['#', 'Z', 'W', 'N', 'P', 'S', 'Z', 'W', '#'],
        ['#', 'N', 'P', 'S', 'S', 'W', '#'],
        ['#', 'N', 'P', 'W', '#'],
        ['#', 'N', 'N', 'P', 'S', 'Z', 'S', 'S', 'W', '#'],
        ['#', 'N', 'N', 'S', 'Z', 'W', '#'],
        ['#', 'N', 'N', 'P', 'S', 'Z', 'S', 'W', '#'],
        ['#', 'N', 'N', 'P', 'P', 'S', 'Z', 'W', '#'],
        ['#', 'N', 'N', 'P', 'P', 'S', 'W', 'P', 'W', '#'],
        ['#', 'N', 'N', 'S', 'W', 'P', 'W', '#'],
        ['#', 'N', 'N', 'S', 'W', 'P', 'S', 'W', '#'],
        ['#', 'N', 'N', 'S', 'W', 'P', 'S', 'S', 'W', '#'],
    ]


def g3():
    """Returns grammar structure for g3"""
    return {
        'START': ['A'],
        'A': ['D', 'F'],
        'D': ['F', 'C1'],
        'F': ['C2', 'END'],
        'C1': ['F', 'G1'],
        'C2': ['G2', 'END'],
        'G1': ['F'],
        'G2': ['END'],
    }


def g3_train():
    return [
        ['A', 'D', 'C', 'G', 'F'],
        ['A', 'D', 'C', 'F', 'C'],
        ['A', 'D', 'F'],
        ['A', 'D', 'F', 'C'],
        ['A', 'D', 'C', 'G', 'F', 'C'],
        ['A', 'F', 'C', 'G'],
        ['A', 'D', 'F', 'C', 'G'],
        ['A', 'D', 'C', 'F'],
    ]


def g3_test_gr():
    return [
        ['A', 'F'],
        ['A', 'D', 'C', 'G', 'F'],
        ['A', 'D', 'C', 'F', 'C'],
        ['A', 'D', 'C', 'F', 'C', 'G'],
    ]


def g3_test_ugr():
    return [
        ['A', 'D', 'C', 'G'],
        ['A', 'C', 'D', 'F', 'C'],
        ['A', 'G', 'F'],
        ['A', 'D', 'G', 'F'],
        ['A', 'D', 'G', 'F', 'C'],
        ['A', 'C', 'F', 'G'],
        ['A', 'D', 'F', 'G'],
        ['A', 'F', 'C', 'F'],
        ['A', 'C', 'G', 'F', 'C', 'G'],
        ['A', 'F', 'G'],
    ]
