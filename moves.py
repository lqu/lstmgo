#!/usr/bin/env python

class MoveEncoder(object):
    def __init__(self, filename):
        with open(filename, 'r') as f:
            self.moves = tuple(f)
        
        self.c2i = dict((c, i) for i, c in enumerate(self.moves))
        self.i2c = dict((i, c) for i, c in enumerate(self.moves))
 
    def encode(self, c):
        """treat all komi values the same way"""
        if c[0:4] == 'komi':
            return self.c2i['komi\n']
        else:
            return self.c2i[c]

    def decode(self, i):
        return self.i2c[i]

    def size(self):
        return len(self.moves)
