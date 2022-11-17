import tqdm
import torch
import argparse
from sklearn.metrics import accuracy_score
from torch.utils.data import TensorDataset, DataLoader
import json
import numpy as np
import pandas as pd
from model import EncoderDecoder, Encoder, Decoder
import torch.nn.functional as functional


from utils import (
    get_device,
    preprocess_string,
    build_tokenizer_table,
    build_output_tables,
    prefix_match
)

def encode_data(data, vocab_to_index, len_cutoff, actions_to_index, targets_to_index):

    encoded_instructions_seqs = np.zeros((len(data), len_cutoff), dtype=np.int32)
    
    longest_episode_len = 0
    for episode in data:
        if len(episode) > longest_episode_len:
            longest_episode_len = len(episode)

    encoded_labels_seqs = np.zeros((len(data), (2*longest_episode_len)+4), dtype=np.int32)


    # loop through the data 
    i = 0
    for episode in data:
        # first index should be the start token
        j = 0
        k = 0
        encoded_instructions_seqs[i][j] = vocab_to_index["<start>"]
        j += 1
        encoded_labels_seqs[i][k] = actions_to_index["<bos>"]
        # if encoded_labels_seqs[i][k] > 10:
        #     print("1")
        #     print(encoded_labels_seqs[i][k])
        #     return
        k += 1
        encoded_labels_seqs[i][k] = targets_to_index["<bos>"]
        # if encoded_labels_seqs[i][k] > 10:
        #     print("2")
        #     print(encoded_labels_seqs[i][k])
        #     return
        k += 1
        
        # encode all of the instructions in an episode and concatenate them together
        for training_instance in episode:
            instruction = training_instance[0]
            label = training_instance[1]

            instruction = preprocess_string(instruction)

            for word in instruction.split(" "):
                if len(word) > 0:
                    encoded_instructions_seqs[i][j] = vocab_to_index[word] if word in vocab_to_index else vocab_to_index["<unk>"]
                    j += 1

                    # break early if necessary
                    if j == len_cutoff-1:
                        break
                
            encoded_labels_seqs[i][k] = actions_to_index[label[0]]
            # if encoded_labels_seqs[i][k] > 10:
            #     print("3")
            #     print(encoded_labels_seqs[i][k])
            #     return
            k += 1
            encoded_labels_seqs[i][k] = targets_to_index[label[1]]
            # if encoded_labels_seqs[i][k] > 10:
            #     print("4")
            #     print(encoded_labels_seqs[i][k])
            #     return
            k += 1

        # encode the end of the sentence
        encoded_instructions_seqs[i][j] = vocab_to_index["<end>"]
        j += 1
        encoded_labels_seqs[i][k] = actions_to_index["<eos>"]
        # if encoded_labels_seqs[i][k] > 10:
        #     print("5")
        #     print(encoded_labels_seqs[i][k])
        #     return
        k += 1
        encoded_labels_seqs[i][k] =  targets_to_index["<eos>"]
        # if encoded_labels_seqs[i][k] > 10:
        #     print("6")
        #     print(encoded_labels_seqs[i][k])
        #     return
        k += 1
        
        i += 1

    # for row in range(len(encoded_labels_seqs)):
    #     print(np.max(encoded_labels_seqs[row]))
    # transform to an even array so it can be casted to a np array
    encoded_labels_seqs = np.array(encoded_labels_seqs, dtype=np.int32)
    return encoded_instructions_seqs, encoded_labels_seqs


def extract_train_val_splits(input_data):
    # open file and get raw data
    raw_data = open(input_data)
    loaded_data = json.load(raw_data)
    raw_data.close()
    return loaded_data["train"], loaded_data["valid_seen"]


def setup_dataloader(args):
    """
    return:
        - train_loader: torch.utils.data.Dataloader
        - val_loader: torch.utils.data.Dataloader
    """
    # ================== TODO: CODE HERE ================== #
    # Task: Load the training data from provided json file.
    # Perform some preprocessing to tokenize the natural
    # language instructions and labels. Split the data into
    # train set and validataion set and create respective
    # dataloaders.

    # Hint: use the helper functions provided in utils.py
    # ===================================================== #

    # Read in the training and validation data
    train_data, val_data = extract_train_val_splits(args.in_data_fn)

    # Tokenize the training set
    # Don't tokenize all data or validation data will leak into training!!
    vocab_to_index, index_to_vocab, len_cutoff = build_tokenizer_table(train_data)

    actions_to_index, index_to_actions, targets_to_index, index_to_targets = build_output_tables(train_data)

    # Encode the training set inputs/outputs.
    train_np_x, train_np_y = encode_data(train_data, vocab_to_index, len_cutoff, actions_to_index, targets_to_index)

    # train_y_weight = np.array([1. / (sum([train_np_y[jdx] == idx for jdx in range(len(train_np_y))]) / len(train_np_y)) for idx in range(len(books_to_index))], dtype=np.float32)
    # convert train data to tensors
    train_dataset = TensorDataset(torch.from_numpy(train_np_x), torch.from_numpy(train_np_y))

    # Encode validation set inputs/outputs
    val_np_x, val_np_y = encode_data(val_data, vocab_to_index, len_cutoff, actions_to_index, targets_to_index)

    # convert val data to tensors
    val_dataset = TensorDataset(torch.from_numpy(val_np_x), torch.from_numpy(val_np_y))

    # Get TFIDF weights from training data.
    # tfidf_ws = get_tfidf_weights(cpb, vocab_to_index, books_to_index)

    # Create dataloaders
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size)
    val_loader = DataLoader(val_dataset, shuffle=True, batch_size=args.batch_size)

    # Create maps for future reference
    maps = {
        "vocab_to_index": vocab_to_index, 
        "index_to_vocab": index_to_vocab,
        "actions_to_index": actions_to_index,
        "index_to_actions": index_to_actions,
        "targets_to_index": targets_to_index,
        "index_to_targets": index_to_targets
        }

    return train_loader, val_loader, maps, len_cutoff


def setup_model(args, vocab_size, actions_size, targets_size, device):
    """
    return:
        - model: YourOwnModelClass
    """
    # ================== TODO: CODE HERE ================== #
    # Task: Initialize your model. Your model should be an
    # an encoder-decoder architecture that encoders the
    # input sentence into a context vector. The decoder should
    # take as input this context vector and autoregressively
    # decode the target sentence. You can define a max length
    # parameter to stop decoding after a certain length.

    # For some additional guidance, you can separate your model
    # into an encoder class and a decoder class.
    # The encoder class forward pass will simply run the input
    # sequence through some recurrent model.
    # The decoder class you will need to implement a teacher
    # forcing mechanism in the forward pass such that instead
    # of feeding the model prediction into the recurrent model,
    # you will give the embedding of the target token.
    # ===================================================== #

    encoder = Encoder(
        device=device, 
        vocab_size=vocab_size,
        embedding_dim=args.emb_dim,
        hidden_dim=args.emb_dim
    )

    decoder = Decoder(
        device=device,
        actions_size=actions_size,
        targets_size=targets_size,
        embedding_dim=args.emb_dim,
        hidden_dim=args.emb_dim,
    )

    model = EncoderDecoder(
        device = device,
        encoder = encoder,
        decoder = decoder,
        actions_size = actions_size,
        targets_size = targets_size,
        batch_size=args.batch_size
    )
    
    return model


def setup_optimizer(args, model):
    """
    return:
        - criterion: loss_fn
        - optimizer: torch.optim
    """
    # ================== TODO: CODE HERE ================== #
    # Task: Initialize the loss function for action predictions
    # and target predictions. Also initialize your optimizer.
    # ===================================================== #
    # action_criterion = torch.nn.BCEWithLogitsLoss()
    # target_criterion = torch.nn.BCEWithLogitsLoss()

    action_criterion = torch.nn.CrossEntropyLoss(ignore_index=0)
    target_criterion = torch.nn.CrossEntropyLoss(ignore_index=0)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate)
    #optimizer in our case always cross entropy, but have to define two different ones 
    #with current setup to check prediction against 
    return action_criterion, target_criterion, optimizer



def train_epoch(
    args,
    model,
    loader,
    optimizer,
    action_criterion,
    target_criterion,
    device,
    training=True,
):
    """
    # TODO: implement function for greedy decoding.
    # This function should input the instruction sentence
    # and autoregressively predict the target label by selecting
    # the token with the highest probability at each step.
    # Note this is slightly different from the forward pass of
    # your decoder because you want to pick the token
    # with the highest probability instead of using the
    # teacher-forced token.

    # e.g. Input: "Walk straight, turn left to the counter. Put the knife on the table."
    # Output: [(GoToLocation, diningtable), (PutObject, diningtable)]
    # Also write some code to compute the accuracy of your
    # predictions against the ground truth.
    """

    epoch_loss = 0.0
    epoch_acc = 0.0

    # iterate over each batch in the dataloader
    # NOTE: you may have additional outputs from the loader __getitem__, you can modify this
    counter = 0
    for (inputs, labels) in loader:
        if counter == int(len(loader)/2):
            print("halfway through this epoch!")
        elif counter == int(len(loader)/4):
            print("a quarter through this epoch")
        elif counter == int(3*len(loader)/4):
            print("3/4ths of the way through")
        counter += 1
        # put model inputs to device
        inputs, labels = inputs.to(device), labels.to(device)

        # calculate the loss and train accuracy and perform backprop
        # NOTE: feel free to change the parameters to the model forward pass here + outputs
        pred_actions, pred_targets = model(inputs, labels, training=training)

        # print('here')
        # print(pred_actions.size())
        # print(pred_actions)
        # print(labels.shape)
        # print(labels[:, 2:-2:2].shape)
        # print(labels[:, 2:-2:2])

        # shape of labels is [5, 19]
        # we want [5, 19, 8+3]
        # print(labels[:, 2:-2:2, None].size())
        true_actions = labels[:, ::2]
        true_targets = labels[:, 1::2]
        # true_actions = labels[:, 2:-2:2]
        # print(true_actions.size())
        # print(true_actions)

        # true_actions = torch.unsqueeze(true_actions, dim=2)
        # true_targets = torch.unsqueeze(true_targets, dim=2)
        # print(true_actions[0, :, :])
        # print(true_actions[:, 0, :])
        # print(true_actions[:,:,0])

        # print(true_actions)
        # print(true_actions.size())

        # print(true_actions.max(dim=2).values)
        # torch.set_printoptions(edgeitems=25)
        # print(true_actions.max(dim=2).values.to(torch.int64))

        # I spent four hours figuring out how to properly do this matrix transformation i think i'm going insane 
        # true_actions = functional.one_hot(true_actions.max(dim=2).values.to(torch.int64), 11)
        # true_targets = functional.one_hot(true_targets.max(dim=2).values.to(torch.int64), 83)
        


        # print(true_actions.size())
        # print(true_actions)
        # print(true_actions[0, 0, :])


        # print(true_actions.size())
        # print(true_actions)
        # print(true_actions[0,0,:])
        # true_tagets = functional.one_hot(pred_targets.argmax(dim=1), 83)
        # labels consists of batch_size x longest_len_episode,
        # where longest_len_episode = [start_action, start_target, action_0, target_0, action_1, target_1,...,action_n, target_n, end_action, end_target]
        # need to slice list appropriately to get the labels
        # action labels should by every other index start from index 2 up to but not including the 2nd last element in the list
        # target labels should be every other index strting from index 3 up to but not includin ghte last element in the list

        pred_actions = torch.flatten(pred_actions, end_dim=1)
        true_actions = torch.flatten(true_actions).type(torch.LongTensor)
        pred_targets = torch.flatten(pred_targets, end_dim=1)
        true_targets = torch.flatten(true_actions).type(torch.LongTensor)
        # print(pred_actions.size())
        # print(pred_actions)
        # print(true_actions.size())
        # print(true_actions)
        # print()
        # print(pred_targets.size())
        # print(true_targets.size())

        action_loss = action_criterion(pred_actions, true_actions)
        target_loss = target_criterion(pred_targets, true_targets)

        # step optimizer and compute gradients during training
        if training:
            optimizer.zero_grad()
            # sum up all the values of policy_losses and value_losses
            loss = action_loss + target_loss
            loss.backward()
            optimizer.step()
        """
        # TODO: implement code to compute some other metrics between your predicted sequence
        # of (action, target) labels vs the ground truth sequence. We already provide 
        # exact match and prefix exact match. You can also try to compute longest common subsequence.
        # Feel free to change the input to these functions.
        """
        # TODO: add code to log these metrics
        if not training:
            exact_match = 0 # TODO
            prefix_em = prefix_match(pred_actions, pred_targets, labels)
            # acc = 0.0

            # logging
            epoch_loss += (action_loss+target_loss)/2
            epoch_acc += prefix_em

        epoch_loss /= len(loader)
        epoch_acc /= len(loader)

    return epoch_loss, epoch_acc


def validate(args, model, loader, optimizer, action_criterion, target_criterion, device):
    # set model to eval mode
    model.eval()

    # don't compute gradients
    with torch.no_grad():
        val_loss, val_acc = train_epoch(
            args,
            model,
            loader,
            optimizer,
            action_criterion,
            target_criterion,
            device,
            training=False,
        )

    return val_loss, val_acc


def train(args, model, loaders, optimizer, action_criterion, target_criterion, device):
    # Train model for a fixed number of epochs
    # In each epoch we compute loss on each sample in our dataset and update the model
    # weights via backpropagation

    counter = 0
    for epoch in tqdm.tqdm(range(args.num_epochs)):
        # train single epoch
        # returns loss for action and target prediction and accuracy
        print("on epoch: " + str(counter))
        counter += 1
        train_loss, train_acc = train_epoch(
            args,
            model,
            loaders["train"],
            optimizer,
            action_criterion,
            target_criterion,
            device,
        )

        # some logging
        print(f"train loss : {train_loss}")

        # run validation every so often
        # during eval, we run a forward pass through the model and compute
        # loss and accuracy but we don't update the model weights
        if epoch % args.val_every == 0:
            val_loss, val_acc = validate(
                args,
                model,
                loaders["val"],
                optimizer,
                action_criterion,
                target_criterion,
                device,
            )

            print(f"val loss : {val_loss} | val acc: {val_acc}")

    # ================== TODO: CODE HERE ================== #
    # Task: Implement some code to keep track of the model training and
    # evaluation loss. Use the matplotlib library to plot
    # 3 figures for 1) training loss, 2) validation loss, 3) validation accuracy
    # ===================================================== #


def main(args):
    device = get_device(args.force_cpu)

    # get dataloaders
    train_loader, val_loader, maps, len_cutoff = setup_dataloader(args)
    loaders = {"train": train_loader, "val": val_loader}

    # print(maps["actions_to_index"])

    # build model
    # subtract 3 because we don't want to include start/end/padding
    model = setup_model(args, len(maps["vocab_to_index"]), len(maps["actions_to_index"]), len(maps["targets_to_index"]), device)
    print(model)

    # get optimizer and loss functions
    action_criterion, target_criterion, optimizer = setup_optimizer(args, model)

    if args.eval:
        val_loss, val_acc = validate(
            args,
            model,
            loaders["val"],
            optimizer,
            action_criterion,
            target_criterion,
            device,
        )
    else:
        train(args, model, loaders, optimizer, action_criterion, target_criterion, device)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_data_fn", type=str, help="data file")
    parser.add_argument(
        "--model_output_dir", type=str, help="where to save model outputs"
    )
    parser.add_argument(
        "--batch_size", type=int, default=62, help="size of each batch in loader"
    )
    parser.add_argument("--force_cpu", action="store_true", help="debug mode")
    parser.add_argument("--eval", action="store_true", help="run eval")
    parser.add_argument("--num_epochs", default=1000, help="number of training epochs")
    parser.add_argument(
        "--val_every", default=1, help="number of epochs between every eval loop"
    )

    # ================== TODO: CODE HERE ================== #
    # Task (optional): Add any additional command line
    # parameters you may need here
    # ===================================================== #

    parser.add_argument(
        "--emb_dim", type=int, default=32, help="number of features/columns to learn for every vector (each vector represents a word)"
    )

    parser.add_argument(
        "--learning_rate", type=float, default=0.001, help="how often to stepr"
    )

    args = parser.parse_args()

    main(args)
