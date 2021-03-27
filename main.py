# Main file

import torch, torchvision
import numpy as np
from grammar import GrammarGen


def main():
    G = GrammarGen()
    lambda seq: [ {0:'A',1:'C',2:'D',3:'F',4:'G'}]
    seqs = G.generate(10)
    print( G.out2read( seqs ) )

if __name__ == '__main__':
    main()