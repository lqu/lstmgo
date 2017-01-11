# Teaching LSTM to play Go

Inspired by the [LSTM text generation](https://github.com/fchollet/keras/blob/master/examples/lstm_text_generation.py), I thought it would be interesting to have LSTM generate a Go game. The current state of a Go board can be viewed as a sequence of moves that leads to this state from an empty board. The sequences can be viewed as a **language**, whose alphabet consists of all the possible moves. 

The goal is to learn a function that maps a sequence of moves to a next move, or to a probability distribution of the next move. This is analogous to [AlphaGo](https://en.wikipedia.org/wiki/AlphaGo)'s supervised learning (SL) policy network. It is designed to predict human expert moves, instead of winning a game. So there is no reinforcement learning or Monte-Carlo Tree Search (MCTS) included here.

Probably, this sequence perspective is not the most efficient way to capture the current state of a Go game, but it shows what the LSTM can learn without any prior domain knowledge such as the rule of the game or the 2D shape of the board.

## Game representatinos
### SGF format
Here is a game in SGF format. See it in [eidogo](http://eidogo.com/#yuedAS1F).

`(;GM[1]FF[4]CA[UTF-8]SZ[19];B[qd];W[dc];B[pq];W[oc];B[cp];W[po];B[pe];W[np])`

### GTP format
It is first translated into GTP format, which is a sequence of moves. This can be done by `sgf2gtp` from [pachi](https://github.com/openai/pachi-py/blob/master/pachi_py/pachi/tools/sgf2gtp.py).

```
boardsize 19
clear_board
play B r16
play W d17
play B q3
play W p17
play B c4
play W q5
play B q15
play W o4
```

### Integar representation
The GTP format is then translated into a list of integars in range [0, 735). This will help generate the training data.

`[0, 1, 323, 438, 309, 647, 62, 673, 303, 633]`

For a quick view of the mapping, run command `nl -v0 moves.gtp`. The total 735 moves are from the following.

```
1. boardsize      1     (size = 19)
2. clear_board    1
3. komi           1     (different numbers treated the same)
4. handicap       8     (from 2 to 9)
5. black moves    361
6. white moves    361
7. black pass     1
8. white pass     1
```

### Training data
The training data are then generated from the above list. Padding is added to make the same length. The labels will be encoded to one hot categorical form.

```
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]                ==>> [1]
[0, 0, 0, 0, 0, 0, 0, 0, 0, 1]                ==>> [323]
[0, 0, 0, 0, 0, 0, 0, 0, 1, 323]              ==>> [438]
[0, 0, 0, 0, 0, 0, 0, 1, 323, 438]            ==>> [309]
[0, 0, 0, 0, 0, 0, 1, 323, 438, 309]          ==>> [647]
[0, 0, 0, 0, 0, 1, 323, 438, 309, 647]        ==>> [62]
[0, 0, 0, 0, 1, 323, 438, 309, 647, 62]       ==>> [673]
[0, 0, 0, 1, 323, 438, 309, 647, 62, 673]     ==>> [303]
[0, 0, 1, 323, 438, 309, 647, 62, 673, 303]   ==>> [633]
```

## Initial Result
Without any tuning of parameters, the LSTM learns the following.

1. alternate black and white stones
2. in handicap games, place black stones before the frist white move 
3. basic pattern and openings

In this [game](http://eidogo.com/#29VJNhrLX), every move is generated from the the sequence of previous moves, starting with 'play B r16'. It even plays a *large avalanche* joseki. But move 32 makes no sense, and move 38 (play W c8) is illegal.

`(;GM[1]FF[4]CA[UTF-8]SZ[19];B[qd];W[dp];B[pq];W[dc];B[de];W[ce];B[dd];W[cd];B[ec];W[cf];B[df];W[dg];B[cc];W[db];B[bc];W[cb];B[cg];W[ch];B[bb];W[eb];B[bg];W[bf];B[bh];W[bd];B[fc];W[dh];B[ci];W[di];B[cj];W[dj];B[ck];W[bk];B[cl];W[bl];B[bm];W[dk];B[dl])`
