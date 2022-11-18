# IMPLEMENT YOUR MODEL CLASS HERE
import numpy as np
import torch.nn as nn
import torch
import random

class Encoder(nn.Module):
    """
    Encode a sequence of tokens. Run the input sequence
    through any recurrent model and output a hidden representation.
    TODO: edit the forward pass arguments to suit your needs
    """
    def __init__(
        self,
        device,
        vocab_size,
        embedding_dim,
        hidden_dim
    ):
        super(Encoder, self).__init__()
        self.device = device
        self.vocab_size = vocab_size
        self.encoder_embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim

        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim, num_layers=1, batch_first=True)

    def forward(self, instruction):
        # instruction is batch_size x len_cutoff
        # print("instruction size: ")
        # print(instruction.size())
        # get vector representation of the words 
        embeds = self.embedding(instruction)
        # print("after embedding: ")
        # print(embeds.size())
        # embeds is batch_size x len_cutoff x emb_dim
        # an array of sequences of lenth len_cutoff with an emb_dim size vector to represent each word

        # hidden contains all of the hidden layers
        # h_n is the last hidden layer, ie the context vector 

        # output is batch_size x seq_len x emb_dim
        # h_n is 1 x batch_size x emb_dim
        # c_n is 1 x batch_size x emb_dim
        output, (h_n, c_n) = self.lstm(embeds)    

        return h_n, c_n


class Decoder(nn.Module):
    """
    Conditional recurrent decoder. Iteratively generates the next
    token given the context vector from the encoder and ground truth
    labels using teacher forcing.
    TODO: edit the forward pass arguments to suit your needs
    """
       
    def __init__(
        self, 
        # check if device does anything...
        device,
        actions_size,
        targets_size,
        embedding_dim,
        hidden_dim,

    ):

        super(Decoder, self).__init__()
        self.device = device
        self.hidden_dim = hidden_dim
        self.actions_size = actions_size
        self.targets_size = targets_size
        self.embedding_dim = embedding_dim

        # self.label_emb = nn.Embedding(2*batch_size, embedding_dim)
        # add 3 to take into account bos, eos, and padding
        self.actions_emb = nn.Embedding(actions_size, embedding_dim)
        self.targets_emb = nn.Embedding(targets_size, embedding_dim)
        self.LSTM = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim, num_layers=1, batch_first=True)
        self.action_fc = nn.Linear(hidden_dim, actions_size)
        self.target_fc = nn.Linear(hidden_dim, targets_size)

    def forward(self, action, target, h_n, c_n):
        # print("h_n size")
        # print(h_n.size())
        # print("c_n size")
        # print(c_n.size())
        # print("action size")
        # print(action.size())
        # print("target size")
        # print(target.size())

        # labels = torch.from_numpy(np.concatenate((action, target)))

        # print("labels size")
        # print(labels.size())
        # labels_emb = self.label_emb(labels)
        # print(action.size())
        # print(action)
        # print(self.actions_size)

        action_emb = self.actions_emb(action)
        target_emb = self.targets_emb(target)
        # print("action emb size")
        # print(action_emb.size())
        # print("target emb size")
        # print(target_emb.size())

        # labels_emb is batch_size x 2 x emb_dim
        labels_emb = torch.cat((action_emb, target_emb), dim=1)
        # check that adding the two is concat, not vector addition
        # print(type(action_emb))
        # print(action_emb.size())
        # print(action_emb)
        # print("labels emb size")
        # print(labels_emb.size())

        # Expected hidden[0] size (1, 5, 7), got [1, 3, 7]
        # h_n and c_n are  batch_size x 1 x emb_dim
        output, (h_n, c_n) = self.LSTM(labels_emb, (h_n, c_n)) # plus either hidden, or (hidden, cell))
        
        # might pass in output.squeeze(0) (aka all the layers, not just last)
        # print("output size")
        # print(output.size())
        
        pred_action = self.action_fc(output[:, 0, :])
        pred_target = self.target_fc(output[:, 1, :])

        # print("pred action")
        # print(pred_action.size())
        # print("pred target")
        # print(pred_target.size())

        # pred_action is of dimensions batch_size x actions_size
        # pred_target is of dimensions batch_size x targets_size
        return pred_action, pred_target, h_n, c_n
        

# have to change forward function...how do we do the loop without the number of actions to predict ahead of time?
class EncoderDecoder(nn.Module):
    """
    Wrapper class over the Encoder and Decoder.
    TODO: edit the forward pass arguments to suit your needs
    """

    def __init__(self, device, encoder, decoder, actions_size, targets_size, batch_size, teacher_forcing_prob):
        super(EncoderDecoder, self).__init__()

        self.device = device
        self.encoder = encoder
        self.decoder = decoder
        self.actions_size = actions_size
        self.targets_size = targets_size
        self.batch_size = batch_size
        self.teacher_forcing_prob = float(teacher_forcing_prob)
        # set dim=1 to softmax along each row 
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, instructions, labels, training):
        # instructions is batch_size x len_cutoff
        h_n, c_n = self.encoder(instructions.to(self.device))

        # tensor to hold all predicted actions and targets

        # dimensions batch_size x longest_len_episode x actions/target size
        # labels.size(1)/2 because it's twice as long as it really is
        all_pred_actions = torch.zeros((self.batch_size, int(labels.size(1)/2), self.actions_size))
        all_pred_targets = torch.zeros((self.batch_size, int(labels.size(1)/2), self.targets_size))
        
        # only have to slide along the columns. Batching will mean that it steps through all of the columns at the same time,
        # so it'll do column 1 for all rows at the same time, col 2 for all rows at the same time, etc. 
        # There will be lots of padding at the end but the loss function knows not to touch them


        # get number_of_labels_in_each_episode predictions back 
        # since we made all of them the same length we go past to padding sometimes
        for idx in range(0, int(labels.size(1)), 2):

            # only use true labels for teacher forcing if training
            if idx == 0 or (training and random.random() < self.teacher_forcing_prob):

                # pred_action is of dimension batch_size x action or batch_size x target size
                pred_action, pred_target, h_n, c_n = self.decoder(labels[:, idx:idx+1].to(self.device), 
                      labels[:, idx+1:idx+2].to(self.device), 
                      h_n.to(self.device),
                      c_n.to(self.device))

                # print()
                # print(labels[:, idx:idx+1].size())
                # print(labels[:, idx:idx+1])
                # print(pred_action)
                # print(torch.reshape(torch.argmax(pred_action, axis=1), (self.batch_size, 1)))
                # pred_action = pred_action[:, torch.argmax(pred_action, axis=1)]
                # print(pred_action.size())
                # print(pred_action)
                # print(labels[:, idx+1:idx+2].size())
                # print(pred_target.size())
                # print(pred_target)
                # print()             

            else:
                # pred_action is of dimension batch_size x action or batch_size x target size
                pred_action, pred_target, h_n, c_n = self.decoder(pred_action.to(self.device),
                        pred_target.to(self.device),
                        h_n.to(self.device), 
                        c_n.to(self.device))

            # greedy decoding
            if not training:

                # why do we do this? shouldn't it be the same without this 
                # turn the result into a probabilitiy distribution
                pred_action = self.logsoftmax(pred_action.to(self.device))
                # get the indices of the action with the highest probability 
                pred_action_indexes = torch.argmax(pred_action, axis=1)
                
                # create a batch_size x actions_size array of 0's
                pred_action = torch.zeros((self.batch_size, self.actions_size))
                # one hot encode the index with the highest probability
                pred_action[:, pred_action_indexes] = 1

                pred_target = self.logsoftmax(pred_target)
                pred_target_indexes = torch.argmax(pred_target, axis=1)
                pred_target = torch.zeros((self.batch_size, self.targets_size))
                pred_target[:, pred_target_indexes] = 1

                # reshape:
                # before: batch_size x action_size or batch_size x target_size, where the second dimension is the one hot encoding of the prediction
                # after: batch_size x 1, where the second dimension is the prediction 
                # pred_action = torch.reshape(torch.argmax(pred_action, axis=1), (self.batch_size, 1))
                # pred_target = torch.reshape(torch.argmax(pred_target, axis=1), (self.batch_size, 1))
                # pred_action = torch.reshape(torch.argmax(pred_action, axis=1), (self.batch_size, 1))
                # pred_target = torch.reshape(torch.argmax(pred_target, axis=1), (self.batch_size, 1))

            # pred_action should be a batch_size x actions_size matrix.
            # each row is a 1 hot encoded vector if not training
            # print("number 1")
            # print(type(all_pred_actions))
            # print("number 2")
            # print(type(pred_action))
                    
            # since we're stepping by 2 the true "index" is divided by 2
                # dimensions batch_size x longest_len_episode x actions/target size
                # pred_action is of dimension batch_size x action or batch_size x target size

            all_pred_actions[:, int(idx/2), :] = pred_action
            all_pred_targets[:, int(idx/2), :] = pred_target

            # reshape so when pred_action and pred_target are used in the next timestep they're in the correct format:
            # before: batch_size x action_size or batch_size x target_size, where the second dimension is the one hot encoding of the prediction
            # after: batch_size x 1, where the second dimension is the prediction 
            pred_action = torch.reshape(torch.argmax(pred_action, axis=1), (self.batch_size, 1))
            pred_target = torch.reshape(torch.argmax(pred_target, axis=1), (self.batch_size, 1))

        # convert numpy to tensor
        # all_pred_actions = all_pred_actions
        # all_pred_targets = all_pred_targets
        # all_pred_actions = torch.from_numpy(all_pred_actions)
        # all_pred_targets = torch.from_numpy(all_pred_targets)
        return all_pred_actions, all_pred_targets