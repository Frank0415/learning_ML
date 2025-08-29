import torch
import torch.nn as nn
from torch.nn import functional as F
import random
import numpy as np

batch_size, num_steps = 32, 35
def load_time_machine():
    with open('./timemachine.txt', 'r') as f:
        lines = f.readlines()
    return [line.strip().lower() for line in lines]

def tokenize(lines, token='char'):
    if token == 'char':
        return [list(line) for line in lines]
    elif token == 'word':
        return [line.split() for line in lines]
    else:
        raise ValueError('Unknown token type: ' + token)

class Vocab:
    def __init__(self, tokens, min_freq=0, reserved_tokens=None):
        if reserved_tokens is None:
            reserved_tokens = []
        counter = {}
        for line in tokens:
            for token in line:
                counter[token] = counter.get(token, 0) + 1
        self.token_freqs = sorted(counter.items(), key=lambda x: x[1], reverse=True)
        self.idx_to_token = ['<unk>'] + reserved_tokens
        self.token_to_idx = {token: idx for idx, token in enumerate(self.idx_to_token)}
        for token, freq in self.token_freqs:
            if freq < min_freq:
                continue
            if token not in self.token_to_idx:
                self.idx_to_token.append(token)
                self.token_to_idx[token] = len(self.idx_to_token) - 1

    def __len__(self):
        return len(self.idx_to_token)

    def __getitem__(self, tokens):
        if not isinstance(tokens, (list, tuple)):
            return self.token_to_idx.get(tokens, self.token_to_idx['<unk>'])
        return [self.__getitem__(token) for token in tokens]

    def to_tokens(self, indices):
        if not isinstance(indices, (list, tuple)):
            return self.idx_to_token[indices]
        return [self.idx_to_token[index] for index in indices]

def seq_data_iter_random(corpus, batch_size, num_steps):
    # Offset for random sampling
    corpus = corpus[random.randint(0, num_steps - 1):]
    num_subseqs = (len(corpus) - 1) // num_steps
    initial_indices = list(range(0, num_subseqs * num_steps, num_steps))
    random.shuffle(initial_indices)

    def data(pos):
        return corpus[pos: pos + num_steps]

    num_batches = num_subseqs // batch_size
    for i in range(0, batch_size * num_batches, batch_size):
        batch_indices = initial_indices[i: i + batch_size]
        X = [data(j) for j in batch_indices]
        Y = [data(j + 1) for j in batch_indices]
        yield torch.tensor(X, dtype=torch.long), torch.tensor(Y, dtype=torch.long)

def load_corpus_time_machine():
    lines = load_time_machine()
    tokens = tokenize(lines, 'char')
    vocab = Vocab(tokens)
    corpus = [vocab[token] for line in tokens for token in line]
    return corpus, vocab

corpus, vocab = load_corpus_time_machine()

# Test correctness
for X, Y in seq_data_iter_random(corpus, batch_size, num_steps):
    print('X shape:', X.shape)
    print('Y shape:', Y.shape)
    print('First X:', X[0])
    print('First Y:', Y[0])
    break