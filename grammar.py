# Grammar

import random

class GrammarGen():
    """
    Generates Grammar sequences from grammars
    Grammars are dictionaries:
    - always have START
    - all paths lead eventually to END
    - Entries starting with the same letter
      have same output
    """

    def __init__(self, grammar=None ):
        if grammar is None:
            self.grammar = {
                'START': ['A'],
                'A': ['D','C1'],
                'C1': ['G1','F'],
                'G1': ['F'],
                'D': ['C1'],
                'F': ['C2', 'END'],
                'C2': ['END', 'G2'],
                'G2': ['END']
                }
        else:
            self.grammar = grammar
        self._initOutput()

    def _initOutput(self):
        """
        Creates Mapping from Stimulus to Output
        """
        stimulusToOutput = dict()
        cores = dict() # keep track of same output letters
        i = 1
        for stimulus in self.grammar:
            if stimulus == 'START':
                continue
            core = stimulus[0]
            if core not in cores:
                cores[core] = i
                i += 1
            stimulusToOutput[stimulus] = cores[core]
        self.stim2out = stimulusToOutput
        self.cores = cores

    def __len__(self):
        return len(self.cores)

    def stim2seqs(self, stimuliSequences):
        seqs = []
        for label, stimulusSequence in stimuliSequences:
            seqs.append( ( label, [ self.cores[stimulus] for stimulus in stimulusSequence ] ) )
        return seqs

    def generate(self, n):
        ret = []
        for _ in range(n):
            token = []
            current = 'START'
            while current != 'END':
                # Append current
                if current != 'START':
                    token.append( self.stim2out[current] )
                # Choose next
                r = random.randint( 0, len( self.grammar[current] ) - 1 )
                current = self.grammar[current][r]
            ret.append( ( 1, token, ) )
        return ret

    def seqs2stim(self, seqs):
        o2r = { v: k for k, v in self.cores.items() }
        return [ [ o2r[stim] for stim in seq ] for _,seq in seqs ]

    def generateUngrammatical(self, n):
        pass
