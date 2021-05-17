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
        (1, ['A', 'C', 'F'], ),
        (1, ['A', 'C', 'F', 'C', 'G'], ),
        (1, ['A', 'C', 'G', 'F'], ),
        (1, ['A', 'C', 'G', 'F', 'C', 'G'], ),
        (1, ['A', 'D', 'C', 'F'], ),
        (1, ['A', 'D', 'C', 'F', 'C'], ),
        (1, ['A', 'D', 'C', 'F', 'C', 'G'], ),
        (1, ['A', 'D', 'C', 'G', 'F', 'C', 'G'], ),
    ]


def g0_train_extended():
    """Returns training set with grammatical and ungrammatical Sequences for g0"""
    return [
        (1, ['A', 'C', 'F'], ),
        (1, ['A', 'C', 'F', 'C', 'G'], ),
        (1, ['A', 'C', 'G', 'F'], ),
        (1, ['A', 'C', 'G', 'F', 'C', 'G'], ),
        (1, ['A', 'D', 'C', 'F'], ),
        (1, ['A', 'D', 'C', 'F', 'C'], ),
        (1, ['A', 'D', 'C', 'F', 'C', 'G'], ),
        (1, ['A', 'D', 'C', 'G', 'F', 'C', 'G'], ),
        (0, ['A', 'C', 'F', 'G']),
        (0, ['A', 'D', 'G', 'F', 'C']),
        (0, ['A', 'C', 'C', 'G']),
        (0, ['A', 'G', 'F', 'C', 'G']),
        (0, ['A', 'D', 'C', 'F', 'G']),
        (0, ['A', 'D', 'C', 'G', 'F', 'G', 'C'], ),
    ]


def g0_test_gr():
    """Returns grammatical test Sequences for g0"""
    return [
        (1, ['A', 'C', 'F', 'C', 'G'], ),
        (1, ['A', 'D', 'C', 'F', 'C'], ),
        (1, ['A', 'C', 'G', 'F', 'C'], ),
        (1, ['A', 'D', 'C', 'G', 'F'], ),
    ]


def g0_test_ugr():
    """Returns ungrammatical test Sequences for g0"""
    return [
        (0, ['A', 'D', 'C', 'F', 'G'], ),
        (0, ['A', 'D', 'F', 'C', 'G'], ),
        (0, ['A', 'D', 'G', 'C', 'F'], ),
        (0, ['A', 'D', 'G', 'F', 'C'], ),
        (0, ['A', 'G', 'C', 'F', 'G'], ),
        (0, ['A', 'G', 'F', 'G', 'C'], ),
        (0, ['A', 'G', 'D', 'C', 'F'], ),
        (0, ['A', 'G', 'F', 'D', 'C'], ),
    ]


def g0_test_all():
    """Returns grammatical and ungrammatical test Sequences"""
    return [
        (1, ['A', 'C', 'F', 'C', 'G'], ),
        (1, ['A', 'D', 'C', 'F', 'G'], ),
        (1, ['A', 'C', 'G', 'F', 'C'], ),
        (1, ['A', 'D', 'C', 'G', 'F'], ),
        (0, ['A', 'D', 'C', 'F', 'G'], ),
        (0, ['A', 'D', 'F', 'C', 'G'], ),
        (0, ['A', 'D', 'G', 'C', 'F'], ),
        (0, ['A', 'D', 'G', 'F', 'C'], ),
        (0, ['A', 'G', 'C', 'F', 'G'], ),
        (0, ['A', 'G', 'F', 'G', 'C'], ),
        (0, ['A', 'G', 'D', 'C', 'F'], ),
        (0, ['A', 'G', 'F', 'D', 'C'], ),
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
        (1, ['W', 'S', 'W', 'Z', ], ),
        (1, ['W', 'S', 'S', 'W', 'Z', ], ),
        (1, ['W', 'S', 'W', 'S', 'N', 'P', 'Z', ], ),
        (1, ['W', 'S', 'W', 'S', 'P', 'N', 'Z', ], ),
        (1, ['W', 'W', 'S', 'P', 'N', 'P', 'Z', ], ),
        (1, ['W', 'W', 'S', 'N', 'P', 'S', 'N', 'Z', ], ),
        (1, ['W', 'W', 'S', 'P', 'N', 'Z', ], ),
        (1, ['W', 'W', 'S', 'N', 'Z', ], ),
        (1, ['W', 'S', 'S', 'W', 'S', 'N', 'Z', ], ),
        (1, ['W', 'S', 'W', 'S', 'N', 'Z', ], ),
        (1, ['N', 'P', 'P', 'N', 'Z', ], ),
        (1, ['N', 'N', 'Z', ], ),
        (1, ['N', 'P', 'P', 'N', 'P', 'Z', ], ),
        (1, ['N', 'N', 'P', 'S', 'P', 'N', 'P', 'Z', ], ),
        (1, ['N', 'P', 'N', 'P', 'S', 'N', 'P', 'Z', ], ),
        (1, ['N', 'N', 'P', 'S', 'P', 'P', 'N', 'Z', ], ),
        (1, ['N', 'P', 'P', 'N', 'P', 'S', 'N', 'Z', ], ),
        (1, ['N', 'N', 'P', 'S', 'N', 'Z', ], ),
    ]


def g1_test_gr():
    """Returns grammatical test Sequences for g1"""
    return [
        (1, ['W', 'W', 'Z', ], ),
        (1, ['W', 'W', 'S', 'N', 'P', 'Z', ], ),
        (1, ['W', 'S', 'S', 'W', 'S', 'N', 'P', 'Z', ], ),
        (1, ['W', 'W', 'S', 'P', 'P', 'N', 'P', 'Z', ], ),
        (1, ['W', 'W', 'S', 'P', 'N', 'Z', ], ),
        (1, ['W', 'S', 'W', 'S', 'N', 'Z', ], ),
        (1, ['W', 'S', 'W', 'S', 'N', 'P', 'Z', ], ),
        (1, ['W', 'S', 'S', 'W', 'S', 'P', 'N', 'Z', ], ),
        (1, ['W', 'W', 'S', 'P', 'P', 'N', 'Z', ], ),
        (1, ['W', 'S', 'W', 'S', 'P', 'P', 'N', 'Z', ], ),
        (1, ['N', 'P', 'N', 'Z', ], ),
        (1, ['N', 'N', 'P', 'S', 'N', 'P', 'Z', ], ),
        (1, ['N', 'N', 'P', 'Z', ], ),
        (1, ['N', 'P', 'N', 'P', 'S', 'N', 'Z', ], ),
        (1, ['N', 'N', 'P', 'S', 'P', 'N', 'Z', ], ),
        (1, ['N', 'P', 'N', 'P', 'S', 'P', 'N', 'Z', ], ),
        (1, ['W', 'W', 'S', 'P', 'P', 'N', 'P', 'Z', ], ),
    ]


def g1_test_ugr_balanced():
    """Returns same amount of ungrammatical test Sequences for g1 as g1_test_gr"""
    return [
        (0, ['W', 'S', 'Z', ], ),
        (0, ['W', 'W', 'S', 'P', 'W', 'S', 'N', 'Z', ], ),
        (0, ['W', 'S', 'W', 'P', 'P', 'N', 'Z', ], ),
        (0, ['W', 'S', 'W', 'S', 'Z', ], ),
        (0, ['W', 'N', 'P', 'S', 'P', 'P', 'N', 'Z', ], ),
        (0, ['N', 'P', 'N', 'S', 'P', 'N', 'Z', ], ),
        (0, ['N', 'N', 'Z', 'S', 'P', 'S', 'N', 'Z', ], ),
        (0, ['N', 'W', 'S', 'W', 'Z', ], ),
        (0, ['N', 'P', 'N', 'W', 'Z', ], ),  # NPP, end, NPL start
        (0, ['W', 'S', 'S', 'N', 'Z', ], ),
        (0, ['W', 'S', 'P', 'S', 'P', 'N', 'Z', ], ),
        (0, ['W', 'W', 'S', 'P', 'P', 'Z', ], ),
        (0, ['W', 'S', 'S', 'N', 'P', 'N', 'P', 'Z', ], ),
        (0, ['N', 'P', 'P', 'S', 'N', 'Z', ], ),
        (0, ['N', 'P', 'S', 'S', 'P', 'S', 'N', 'Z', ], ),
        (0, ['N', 'N', 'P', 'S', 'N', 'N', 'P', 'Z', ], ),
        (0, ['N', 'P', 'N', 'P', 'N', 'Z', ], ),
    ]


def g1_test_ugr():
    """Returns ungrammatical test Sequences for g1"""
    return [
        (0, ['W', 'Z', 'W', 'Z', ], ),
        (0, ['W', 'S', 'W', 'N', 'P', 'Z', ], ),
        (0, ['W', 'W', 'S', 'N', 'W', 'Z', ], ),
        (0, ['W', 'S', 'W', 'N', 'P', 'S', 'N', 'Z', ], ),
        (0, ['N', 'Z', 'P', 'P', 'N', 'Z', ], ),
        (0, ['N', 'N', 'S', 'P', 'N', 'P', 'Z', ], ),
        (0, ['N', 'S', 'P', 'P', 'N', 'Z', ], ),
        (0, ['N', 'N', 'W', 'S', 'P', 'P', 'N', 'Z', ], ),
        (0, ['W', 'S', 'Z', ], ),
        (0, ['W', 'W', 'S', 'P', 'W', 'S', 'N', 'Z', ], ),
        (0, ['W', 'S', 'W', 'P', 'P', 'N', 'Z', ], ),
        (0, ['W', 'S', 'W', 'S', 'Z', ], ),
        (0, ['W', 'N', 'P', 'S', 'P', 'P', 'N', 'Z', ], ),
        (0, ['N', 'P', 'N', 'S', 'P', 'N', 'Z', ], ),
        (0, ['N', 'N', 'Z', 'S', 'P', 'S', 'N', 'Z', ], ),
        (0, ['N', 'W', 'S', 'W', 'Z', ], ),
        (0, ['N', 'P', 'N', 'W', 'Z', ], ),  # NPP, end, NPL start
        (0, ['W', 'S', 'S', 'N', 'Z', ], ),
        (0, ['W', 'S', 'P', 'S', 'P', 'N', 'Z', ], ),
        (0, ['W', 'W', 'S', 'P', 'P', 'Z', ], ),
        (0, ['W', 'S', 'S', 'N', 'P', 'N', 'P', 'Z', ], ),
        (0, ['N', 'P', 'P', 'S', 'N', 'Z', ], ),
        (0, ['N', 'P', 'S', 'S', 'P', 'S', 'N', 'Z', ], ),
        (0, ['N', 'N', 'P', 'S', 'N', 'N', 'P', 'Z', ], ),
        (0, ['N', 'P', 'N', 'P', 'N', 'Z', ], ),
        (0, ['N', 'P', 'P', 'Z', ], ),
        (0, ['N', 'P', 'S', 'P', 'Z', ], ),
        (0, ['N', 'P', 'N', 'P', 'S', 'S', 'W', 'Z', ], ),
        (0, ['N', 'P', 'S', 'P', 'P', 'N', 'Z', ], ),
        (0, ['W', 'W', 'S', 'N', 'N', 'Z', ], ),
        (0, ['W', 'S', 'W', 'S', 'S', 'N', 'Z', ], ),
        (0, ['W', 'S', 'N', 'Z', ], ),
        (0, ['W', 'S', 'P', 'P', 'N', 'Z', ], ),
        (0, ['W', 'S', 'P', 'N', 'P', 'P', 'N', 'Z', ], ),
    ]


def g1_train_x():
    """Returns train Sequences for g1"""
    return [
        (1, ['#', 'W', 'S', 'W', 'Z', '#'], ),
        (1, ['#', 'W', 'S', 'S', 'W', 'Z', '#'], ),
        (1, ['#', 'W', 'S', 'W', 'S', 'N', 'P', 'Z', '#'], ),
        (1, ['#', 'W', 'S', 'W', 'S', 'P', 'N', 'Z', '#'], ),
        (1, ['#', 'W', 'W', 'S', 'P', 'N', 'P', 'Z', '#'], ),
        (1, ['#', 'W', 'W', 'S', 'N', 'P', 'S', 'N', 'Z', '#'], ),
        (1, ['#', 'W', 'W', 'S', 'P', 'N', 'Z', '#'], ),
        (1, ['#', 'W', 'W', 'S', 'N', 'Z', '#'], ),
        (1, ['#', 'W', 'S', 'S', 'W', 'S', 'N', 'Z', '#'], ),
        (1, ['#', 'W', 'S', 'W', 'S', 'N', 'Z', '#'], ),
        (1, ['#', 'N', 'P', 'P', 'N', 'Z', '#'], ),
        (1, ['#', 'N', 'N', 'Z', '#'], ),
        (1, ['#', 'N', 'P', 'P', 'N', 'P', 'Z', '#'], ),
        (1, ['#', 'N', 'N', 'P', 'S', 'P', 'N', 'P', 'Z', '#'], ),
        (1, ['#', 'N', 'P', 'N', 'P', 'S', 'N', 'P', 'Z', '#'], ),
        (1, ['#', 'N', 'N', 'P', 'S', 'P', 'P', 'N', 'Z', '#'], ),
        (1, ['#', 'N', 'P', 'P', 'N', 'P', 'S', 'N', 'Z', '#'], ),
        (1, ['#', 'N', 'N', 'P', 'S', 'N', 'Z', '#'], ),
    ]


def g1_test_gr_x():
    """Returns grammatical test Sequences for g1"""
    return [
        (1, ['#', 'W', 'W', 'Z', '#'], ),
        (1, ['#', 'W', 'W', 'S', 'N', 'P', 'Z', '#'], ),
        (1, ['#', 'W', 'S', 'S', 'W', 'S', 'N', 'P', 'Z', '#'], ),
        (1, ['#', 'W', 'W', 'S', 'P', 'P', 'N', 'P', 'Z', '#'], ),
        (1, ['#', 'W', 'W', 'S', 'P', 'N', 'Z', '#'], ),
        (1, ['#', 'W', 'S', 'W', 'S', 'N', 'Z', '#'], ),
        (1, ['#', 'W', 'S', 'W', 'S', 'N', 'P', 'Z', '#'], ),
        (1, ['#', 'W', 'S', 'S', 'W', 'S', 'P', 'N', 'Z', '#'], ),
        (1, ['#', 'W', 'W', 'S', 'P', 'P', 'N', 'Z', '#'], ),
        (1, ['#', 'W', 'S', 'W', 'S', 'P', 'P', 'N', 'Z', '#'], ),
        (1, ['#', 'N', 'P', 'N', 'Z', '#'], ),
        (1, ['#', 'N', 'N', 'P', 'S', 'N', 'P', 'Z', '#'], ),
        (1, ['#', 'N', 'N', 'P', 'Z', '#'], ),
        (1, ['#', 'N', 'P', 'N', 'P', 'S', 'N', 'Z', '#'], ),
        (1, ['#', 'N', 'N', 'P', 'S', 'P', 'N', 'Z', '#'], ),
        (1, ['#', 'N', 'P', 'N', 'P', 'S', 'P', 'N', 'Z', '#'], ),
        (1, ['#', 'W', 'W', 'S', 'P', 'P', 'N', 'P', 'Z', '#'], ),
    ]


def g1_test_ugr_x():
    """Returns ungrammatical test Sequences for g1"""
    return [
        (0, ['#', 'W', 'Z', 'W', 'Z', '#'], ),
        (0, ['#', 'W', 'S', 'W', 'N', 'P', 'Z', '#'], ),
        (0, ['#', 'W', 'W', 'S', 'N', 'W', 'Z', '#'], ),
        (0, ['#', 'W', 'S', 'W', 'N', 'P', 'S', 'N', 'Z', '#'], ),
        (0, ['#', 'N', 'Z', 'P', 'P', 'N', 'Z', '#'], ),
        (0, ['#', 'N', 'N', 'S', 'P', 'N', 'P', 'Z', '#'], ),
        (0, ['#', 'N', 'S', 'P', 'P', 'N', 'Z', '#'], ),
        (0, ['#', 'N', 'N', 'W', 'S', 'P', 'P', 'N', 'Z', '#'], ),
        (0, ['#', 'W', 'S', 'Z', '#'], ),
        (0, ['#', 'W', 'W', 'S', 'P', 'W', 'S', 'N', 'Z', '#'], ),
        (0, ['#', 'W', 'S', 'W', 'P', 'P', 'N', 'Z', '#'], ),
        (0, ['#', 'W', 'S', 'W', 'S', 'Z', '#'], ),
        (0, ['#', 'W', 'N', 'P', 'S', 'P', 'P', 'N', 'Z', '#'], ),
        (0, ['#', 'N', 'P', 'N', 'S', 'P', 'N', 'Z', '#'], ),
        (0, ['#', 'N', 'N', 'Z', 'S', 'P', 'S', 'N', 'Z', '#'], ),
        (0, ['#', 'N', 'W', 'S', 'W', 'Z', '#'], ),
        (0, ['#', 'N', 'P', 'N', 'W', 'Z', '#'], ),  # NPP, end, NPL start
        (0, ['#', 'W', 'S', 'S', 'N', 'Z', '#'], ),
        (0, ['#', 'W', 'S', 'P', 'S', 'P', 'N', 'Z', '#'], ),
        (0, ['#', 'W', 'W', 'S', 'P', 'P', 'Z', '#'], ),
        (0, ['#', 'W', 'S', 'S', 'N', 'P', 'N', 'P', 'Z', '#'], ),
        (0, ['#', 'N', 'P', 'P', 'S', 'N', 'Z', '#'], ),
        (0, ['#', 'N', 'P', 'S', 'S', 'P', 'S', 'N', 'Z', '#'], ),
        (0, ['#', 'N', 'N', 'P', 'S', 'N', 'N', 'P', 'Z', '#'], ),
        (0, ['#', 'N', 'P', 'N', 'P', 'N', 'Z', '#'], ),
        (0, ['#', 'N', 'P', 'P', 'Z', '#'], ),
        (0, ['#', 'N', 'P', 'S', 'P', 'Z', '#'], ),
        (0, ['#', 'N', 'P', 'N', 'P', 'S', 'S', 'W', 'Z', '#'], ),
        (0, ['#', 'N', 'P', 'S', 'P', 'P', 'N', 'Z', '#'], ),
        (0, ['#', 'W', 'W', 'S', 'N', 'N', 'Z', '#'], ),
        (0, ['#', 'W', 'S', 'W', 'S', 'S', 'N', 'Z', '#'], ),
        (0, ['#', 'W', 'S', 'N', 'Z', '#'], ),
        (0, ['#', 'W', 'S', 'P', 'P', 'N', 'Z', '#'], ),
        (0, ['#', 'W', 'S', 'P', 'N', 'P', 'P', 'N', 'Z', '#'], ),
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
        (1, ['#', 'Z', 'Z', 'S', 'W', '#'], ),
        (1, ['#', 'Z', 'Z', 'S', 'S', 'W', '#'], ),
        (1, ['#', 'Z', 'W', 'P', 'W', '#'], ),
        (1, ['#', 'Z', 'W', 'N', 'S', 'W', 'P', 'W', '#'], ),
        (1, ['#', 'Z', 'W', 'P', 'S', 'S', 'W', '#'], ),
        (1, ['#', 'Z', 'W', 'N', 'S', 'Z', 'S', 'S', 'W', '#'], ),
        (1, ['#', 'Z', 'W', 'N', 'P', 'S', 'Z', 'S', 'W', '#'], ),
        (1, ['#', 'Z', 'W', 'N', 'P', 'S', 'Z', 'W', '#'], ),
        (1, ['#', 'N', 'P', 'S', 'S', 'W', '#'], ),
        (1, ['#', 'N', 'P', 'W', '#'], ),
        (1, ['#', 'N', 'N', 'P', 'S', 'Z', 'S', 'S', 'W', '#'], ),
        (1, ['#', 'N', 'N', 'S', 'Z', 'W', '#'], ),
        (1, ['#', 'N', 'N', 'P', 'S', 'Z', 'S', 'W', '#'], ),
        (1, ['#', 'N', 'N', 'P', 'P', 'S', 'Z', 'W', '#'], ),
        (1, ['#', 'N', 'N', 'P', 'P', 'S', 'W', 'P', 'W', '#'], ),
        (1, ['#', 'N', 'N', 'S', 'W', 'P', 'W', '#'], ),
        (1, ['#', 'N', 'N', 'S', 'W', 'P', 'S', 'W', '#'], ),
        (1, ['#', 'N', 'N', 'S', 'W', 'P', 'S', 'S', 'W', '#'], ),
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
        (1, ['A', 'D', 'C', 'G', 'F'], ),
        (1, ['A', 'D', 'C', 'F', 'C'], ),
        (1, ['A', 'D', 'F'], ),
        (1, ['A', 'D', 'F', 'C'], ),
        (1, ['A', 'D', 'C', 'G', 'F', 'C'], ),
        (1, ['A', 'F', 'C', 'G'], ),
        (1, ['A', 'D', 'F', 'C', 'G'], ),
        (1, ['A', 'D', 'C', 'F'], ),
    ]


def g3_test_gr():
    return [
        (1, ['A', 'F'], ),
        (1, ['A', 'D', 'C', 'G', 'F'], ),
        (1, ['A', 'D', 'C', 'F', 'C'], ),
        (1, ['A', 'D', 'C', 'F', 'C', 'G'], ),
    ]


def g3_test_ugr():
    return [
        (1, ['A', 'D', 'C', 'G'], ),
        (1, ['A', 'C', 'D', 'F', 'C'], ),
        (1, ['A', 'G', 'F'], ),
        (1, ['A', 'D', 'G', 'F'], ),
        (1, ['A', 'D', 'G', 'F', 'C'], ),
        (1, ['A', 'C', 'F', 'G'], ),
        (1, ['A', 'D', 'F', 'G'], ),
        (1, ['A', 'F', 'C', 'F'], ),
        (1, ['A', 'C', 'G', 'F', 'C', 'G']),
        (1, ['A', 'F', 'G'], ),
    ]
