import torch
import data_utils

class SkipGram(torch.nn.Module):
    def __init__(
        self,
        device,
        vocab_size,
        embedding_dim
    ):
        super(SkipGram, self).__init__()
        self.device = device
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim

        # embedding layer
        # a lookup table mapping each token/word to a vector of size embedding_dim, so has dimensions B x emb
        # these are the weights being learned for each word representation
        # rows are vocab size so each row represents a word in the vocab
        # cols are emb_dim, which represent features for each word 
        # generally don't want embedding vector to be larger than size of data bc then it'll just learn directly

        # relevant parameters are: (num_embeddings, embedding_dim, padding_idx=None)
        # num_embeddings is size of the dictionary of embeddings
        # embedding_dim is the size of each embedding/feature vector that represents a word
        self.input_embedding = torch.nn.Embedding(self.vocab_size, self.embedding_dim, padding_idx=0)

        # self.output_embedding = torch.nn.Embedding(batch_size, v, padding_idx=0)

        # initializing embeddings with uniform distribution 
        self.input_embedding.weight.data.uniform_(-1,1)

        # https://towardsdatascience.com/implementing-word2vec-in-pytorch-skip-gram-model-e6bae040d2fb
        # https://www.kaggle.com/code/karthur10/skip-gram-implementation-with-pytorch-step-by-step/notebook
        # https://towardsdatascience.com/word2vec-skip-gram-model-part-1-intuition-78614e4d6e0b
        # http://mccormickml.com/2016/04/19/word2vec-tutorial-the-skip-gram-model/
        # https://www.geeksforgeeks.org/implement-your-own-word2vecskip-gram-model-in-python/
        # https://github.com/n0obcoder/Skip-Gram-Model-PyTorch/blob/master/model.py

        # According to an article above, there are two weight matrices
        # 1. between inputs and hidden layer. This is used to represent the word vectors
        # 2. between the hidden layer and outputs. This is basically only used for training and error calculation
  
        # W1 = Variable(torch.randn(embedding_dims, vocabulary_size).float(), requires_grad=True)
        # W2 = Variable(torch.randn(vocabulary_size, embedding_dims).float(), requires_grad=True)
        # W1 = Variable(torch.randn(vocab_size, embedding_dims, device=device).uniform_(-initrange, initrange).float(), requires_grad=True) # shape V*H
        # W2 = Variable(torch.randn(embedding_dims, vocab_size, device=device).uniform_(-initrange, initrange).float(), requires_grad=True) #shape H*V

        # one fully connected layer
        self.fc = torch.nn.Linear(embedding_dim, self.vocab_size)

    def forward(self, words): #, contexts):
        # words is the input, and it's a list of tokenized words
        # each row represents a word (the input)
        # words has dimensions batch_size x 1   // should I also transform this into batch_size x vocab_dim? would take more memory...
        # I think it has to be transformed to batch_size x vocab_dim so that the first weight matrix is vocab_dim x emb_dim because those are the word vectors
        # it looks like this: 
        # [985,
        #  16,
        #  1684,
        #  ...
        #  3421]
        # 
        # context is the true context
        # context has dimenions batch_size x |vocab_size|
        # each row represents the context, where the index that has 1's correspond to the token representing a word found in the context of the input
        # it looks like this:
        # [[0, 1, 0, 0, ..., 1, ..., 1, ...],
        # [0, 0, 0, 0, ..., 1, ..., 1, ..., 1, ...],
        # [0, 0, 0, 0, ..., 1, ..., 1, ..., 1, ...].
        # ...
        # ]

        # First, transform words from a B x 1 vector to B x |V|, where the rows are 1 hot encoded representations of the word 
        # words_transformed = data_utils.token_to_multihot(words, self.vocab_size)

        # Create embeddings for words 

        # retreive the embedding/feature vectors for the input words
        # should be of size B x emb_dim
        word_embs = self.input_embedding(words)

        # pass it through the fully connected linear layer, which has size emb_dim x |vocab|
        # result is B x |vocab|, where each row is a multihot encoding of the predicted words    
        out = self.fc(word_embs)

        return out