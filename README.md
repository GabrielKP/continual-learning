# Continual-Learning

## Artificial Grammar Learning Experiment Milne et al 2018

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
