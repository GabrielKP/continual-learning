# Continual-Learning

The repository has two important parts:
1. cifar10x10.ipynb
2. agl

## cifar10x10.ipynb

The notebook contains an algorithmic basis to test continual learning algorithms.
It implements Finetune, (more to come).

## AGL Folder

This part was inspired by the thought of repeating the Milne et al AGL
experiment for humans and compare it to simulations on the computer. Since
training is only happening on one class, direct classification cannot happen
based on training of labels, thus the computer model consists of an RNN which
aims to reconstruct the learned sequences. For that a non-state-of-the-art
Sequence2Sequence model is used. It is important the model does not generalize
too easily (as transformers, I have tried) because it will not be able to
distinguish ungrammatical and grammatical sequences - which often only differ
by edit distance 1. Continual learning for the model on the computer is realized
by different methods: the mixture of expert model DynaMoE from
[this](https://doi.org/10.1073/pnas.2009591117) paper is implemented with some
adaptions for the supervised setting. The model makes use of a mixture of
experts: multiple experts are individually trained and a gating network decides
which expert to use. Although the original model recognizes
when a new expert needs to initialized, after it does that the old expert
becomes untrainable. Furthermore it makes use of some sort of "learning oracle":
the model assumes that only the same task is fed to it until convergence.

Thus, the model is further developed to automatically assign its incoming
training examples to the different experts depending on the classification
error. Furthermore, original task data is replayed to the gating network
to prevent the gating from catastrophically forgetting which experts to assign
inputs to.

Another model is built: the ensembler. Instead of choosing an expert, the
ensembler multiplies the experts outputs and adds them. Thus, intermediate
outputs between experts outputs are possible. The model has an essential flaw:
due to self-reinforcing effects when training with backprop the model always
converges to one expert doing all the heavy lifting with all other models
being basically useless.

The model is improved by a replay mechanism and a custom training mechanism, in
which the error is not entirely backpropagated through the gating to the
experts, but a specific expert is chosen based on the classification error to
be trained. This prevents the expert overpreference problem from above.
Furthermore the gating is equipped with replay of old task data to not forget
old inputs.
### Artificial Grammar Learning Experiment Milne et al 2018

Familiarisation -> Test -> Refamilarisation -> Test ...

Familiarisation:
Monkey:
- 20 times each exposure sequence in random order
Human:
- 6 times each exposure sequence in random order

Test:
Monkey:
- Many trials. Many Many trials.
- Response: Analysis looking duration.
Human:
- 32 test sequences; incorrect sequence twice, correct sequence four times;
- Response: Forced choice key press.

Refamiliarisation:
Monkey:
- 8 times each exposure
Human:
- 4 times each exposure sequence

Exposure sequences:
```
['A','C','F'],
['A','C','F','C','G'],
['A','C','G','F'],
['A','C','G','F','C','G'],
['A','D','C','F'],
['A','D','C','F','C'],
['A','D','C','F','C','G'],
['A','D','C','G','F','C','G'],
```
Test sequences:
```
correct: ['A','C','F','C','G'],
correct: ['A','D','C','F','G'],
correct: ['A','C','G','F','C'],
correct: ['A','D','C','G','F'],
incorrect: ['A','D','C','F','G'],
incorrect: ['A','D','F','C','G'],
incorrect: ['A','D','G','C','F'],
incorrect: ['A','D','G','F','C'],
incorrect: ['A','G','C','F','G'],
incorrect: ['A','G','F','G','C'],
incorrect: ['A','G','D','C','F'],
incorrect: ['A','G','F','D','C'],
```
## Grammars

Following section contains visualizations of alternate grammars, in the
end we decided to use the original grammar from Milne et al.

### Grammatical

Original from A.S. Reuber, 1969:

Grammar 1:
<img src="agl/data/grammar-1.png" alt="grammar" width="400"/>

Grammar 2:
<img src="agl/data/grammar-2.png" alt="grammar" width="400"/>

- Every token on average the same information
- Both Grammars exactly 43 different paths from start to finish with length <= 8


Converted from Gomez & Schwaneveldt, 1994

Grammar 1:
<img src="data/grammar-3.png" alt="grammar" width="400"/>

Grammar 2:
<img src="data/grammar-4.png" alt="grammar" width="400"/>

### (Un)grammatical sequence constraints

Vokey-Brooks-1992:
- 3 to 7 letter seqs
- different seqs at least edit distance 2
- balanced seq length, balanced usage of transitions
- ug seqs differed in only one position (edit distance 1)
- Accuracy in Recognition Task 72%

Lotz-Kinder-2006:
- same as Vokey-Brooks
- non-transfer accuracy: 56% (still significant) transfer: 54%

Gomez-Schvaneveldt-1994:
- Letter repetitions limited to two
- No ungrammaticalities at beginning or end of seq
- Ungrammaticalities: insertion of 'illegal pair' into string, insertion of 'legal pair' into wrong location
- do not use hits but sensitivity as measure..


## Adhoc

### Old Grammar

5 Stimuli: A C D G F

```
   D    G    C -> G
 /  \ /  \ /  \  /
A -> C -> F -> END
```

Based on:
```
S -> AP + CP + FP
AP -> A + (D)
CP -> C + (G)
FP -> F + (CP)
```
Predictable


Example for unpredictable (Saffran 2008):
```
S -> AP + BP
AP -> {(A) + (D)}
BP -> CP + F
CP -> {(C) + (G)}

{} == xor
```

### Grammar 3 (ad hoc)

5 Stimuli: A C D G F

Basis:
```
S -> AP + FP
AP -> A + (DP)
DP -> D + (CP)
CP -> C + (G)
FP -> F + (CP)
```

## Model architecture

### Encoder
 Input -> Embedding -> Dropout -> fc_one -> ReLU -> dropout -> LSTM -> Hidden

### Decoder (single token decoder)
                            Hidden -> \
 PrevToken -> Embedding -> Dropout -> LSTM -> fc_one -> ReLU -> dropout -> fc_out -> Output/Hidden
