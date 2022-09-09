import tqdm
import torch
import argparse
from sklearn.metrics import accuracy_score
import json
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt


from utils import (
    get_device,
    preprocess_string,
    build_tokenizer_table,
    build_output_tables,
    stem_data
)

def encode_data(data, vocab_to_index, seq_len, actions_to_index, targets_to_index):

    # create encoded instruction and labels
    encoded_instruction = np.zeros((len(data), seq_len), dtype=np.int32)
    encoded_label = np.zeros((len(data), 2), dtype=np.int32)

    # loop through the data 
    i = 0
    for episode in data:
        for instruction, label in episode:
            instruction = preprocess_string(instruction)

            # encode start 
            encoded_instruction[i][0] = vocab_to_index["<start>"]

            # encode the rest of the sentence
            j = 1
            for word in instruction.split():
                if len(word) > 0:
                    encoded_instruction[i][j] = vocab_to_index[word] if word in vocab_to_index else vocab_to_index["<unk>"]
                    j += 1
                    if j == seq_len-1:
                        break

            # encode the end of the sentence
            encoded_instruction[i][j] = vocab_to_index["<end>"]
            encoded_label[i][0] = actions_to_index[label[0]]
            encoded_label[i][1] = targets_to_index[label[1]]

    return encoded_instruction, encoded_label


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

    # stem the data if flag is set
    if(args.stemming):
        train_data = stem_data(train_data)
        val_data = stem_data(val_data)

    # Tokenize the training set
    vocab_to_index, index_to_vocab, len_cutoff = build_tokenizer_table(train_data)
    actions_to_index, index_to_actions, targets_to_index, index_to_targets = build_output_tables(train_data)

    # Encode the training and validation set inputs/outputs.
    train_np_x, train_np_y = encode_data(train_data, vocab_to_index, len_cutoff, actions_to_index, targets_to_index)
    # train_y_weight = np.array([1. / (sum([train_np_y[jdx] == idx for jdx in range(len(train_np_y))]) / len(train_np_y)) for idx in range(len(books_to_index))], dtype=np.float32)
    train_dataset = TensorDataset(torch.from_numpy(train_np_x), torch.from_numpy(train_np_y))
    val_np_x, val_np_y = encode_data(val_data, vocab_to_index, len_cutoff, actions_to_index, targets_to_index)
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


class SemanticUnderstanding(torch.nn.Module):
    def __init__(
        self,
        device,
        vocab_size,
        input_len,
        n_actions,
        n_targets,
        embedding_dim
    ):
        super(SemanticUnderstanding, self).__init__()
        self.device = device
        self.vocab_size = vocab_size
        self.input_len = input_len
        self.n_actions = n_actions
        self.n_targets = n_targets
        self.embedding_dim = embedding_dim

        # embedding layer
        self.embedding = torch.nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        # maxpool layer
        # self.maxpool = torch.nn.MaxPool2d((input_len, 1), ceil_mode=True)

        # takes embeddings as inputs and outputs with dimensionality hidden dim 
        self.lstm = torch.nn.LSTM(input_size=embedding_dim, hidden_size=embedding_dim, batch_first=True)

        # fully connected layer is linear layer + activation function (for nonlineararity)
        # cross entropy does softmax automatically so we don't have to 
        # alternatively can have separate linear layers
        # alternatively can multiply to have every combination
        # linear layer
        self.fc = torch.nn.Linear(embedding_dim, n_actions+n_targets)

        # self.fc1 = torch.nn.Linear(embedding_dim, n_actions)
        # self.fc2 = torch.nn.Linear(embedding_dim, n_targets)

    def forward(self, instruction):
        # instruction is a matrix that is batch_size x len_cutoff
        # embedding layer adds another dimension: batch_size x len_cutoff x embedding_dim
        # print("instructions size: " + str(instruction.size()))
        embeds = self.embedding(instruction)

        # print("embeds size: " + str(embeds.size()))
        _, (h_n,_) = self.lstm(embeds)

        # print("h_n size: " + str(h_n.size()))
        # maxpooled_embeds = self.maxpool(embeds)
        # out = self.fc(maxpooled_embeds).squeeze(1) # squeeze out the singleton length dimension that we maxpool'd over
        # print(out.size())
        out = self.fc(h_n)

        # action_out = self.fc1(h_n)
        # target_out = self.fc2(h_n)
        # print("out size:" + str(out.size()))
        out = out.squeeze()
        action_tensor = out[:,:self.n_actions]
        target_tensor = out[:,self.n_actions:]
        
        # return converted tensors
        return action_tensor, target_tensor


def setup_model(device, maps, len_cutoff, embedding_dim):
    """
    return:
        - model: YourOwnModelClass
    """
    # ================== TODO: CODE HERE ================== #
    # Task: Initialize your model.
    # ===================================================== #
    
    # constructor for model
    model = SemanticUnderstanding(
        device, 
        len(maps["vocab_to_index"]), 
        len_cutoff, 
        len(maps["actions_to_index"]),
        len(maps["targets_to_index"]), 
        embedding_dim)
    return model


def setup_optimizer(args, model):
    """
    return:
        - action_criterion: loss_fn
        - target_criterion: loss_fn
        - optimizer: torch.optim
    """
    # ================== TODO: CODE HERE ================== #
    # Task: Initialize the loss function for action predictions
    # and target predictions. Also initialize your optimizer.
    # ===================================================== #
    action_criterion = torch.nn.CrossEntropyLoss()
    target_criterion = torch.nn.CrossEntropyLoss()
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
    epoch_action_loss = 0.0
    epoch_target_loss = 0.0

    # keep track of the model predictions for computing accuracy
    action_preds = []
    target_preds = []
    action_labels = []
    target_labels = []

    # iterate over each batch in the dataloader
    # NOTE: you may have additional outputs from the loader __getitem__, you can modify this
    for (instruction, labels) in loader:
        # put model inputs to device
        instruction, labels = instruction.to(device), labels.to(device)

        # calculate the loss and train accuracy and perform backprop
        # NOTE: feel free to change the parameters to the model forward pass here + outputs
        actions_out, targets_out = model(instruction)

        # calculate the action and target prediction loss
        # NOTE: we assume that labels is a tensor of size Bx2 where labels[:, 0] is the
        # action label and labels[:, 1] is the target label
        action_loss = action_criterion(actions_out.squeeze(), labels[:, 0].long())
        target_loss = target_criterion(targets_out.squeeze(), labels[:, 1].long())

        loss = action_loss + target_loss

        # step optimizer and compute gradients during training
        if training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # logging
        epoch_action_loss += action_loss.item()
        epoch_target_loss += target_loss.item()

        # take the prediction with the highest probability
        # NOTE: this could change depending on if you apply Sigmoid in your forward pass
        action_preds_ = actions_out.argmax(-1)
        target_preds_ = targets_out.argmax(-1)

        # aggregate the batch predictions + labels
        action_preds.extend(action_preds_.squeeze().cpu().numpy())
        target_preds.extend(target_preds_.squeeze().cpu().numpy())
        action_labels.extend(labels[:, 0].cpu().numpy())
        target_labels.extend(labels[:, 1].cpu().numpy())

    # calculate accuracy
    action_acc = accuracy_score(action_preds, action_labels)
    target_acc = accuracy_score(target_preds, target_labels)

    return epoch_action_loss, epoch_target_loss, action_acc, target_acc


def validate(
    args, model, loader, optimizer, action_criterion, target_criterion, device
):
    # set model to eval mode
    model.eval()

    # don't compute gradients
    with torch.no_grad():
        val_action_loss, val_target_loss, action_acc, target_acc = train_epoch(
            args,
            model,
            loader,
            optimizer,
            action_criterion,
            target_criterion,
            device,
            training=False,
        )

    return val_action_loss, val_target_loss, action_acc, target_acc


def train(args, model, loaders, optimizer, action_criterion, target_criterion, device):
    # Train model for a fixed number of epochs
    # In each epoch we compute loss on each sample in our dataset and update the model
    # weights via backpropagation
    model.train()

    fig, axs = plt.subplots(2,2)
    fig.suptitle("Model Training Trends")
    # axs[0, 0] is training loss
    # axs[0, 1] is training accuracy
    # axs[1, 0] is validation loss
    # axs[1, 1] is validation accuracy
    training_loss_list = []
    training_acc_list = []
    val_loss_list = []
    val_acc_list = []
    
    for epoch in tqdm.tqdm(range(args.num_epochs)):

        # train single epoch
        # returns loss for action and target prediction and accuracy
        (
            train_action_loss,
            train_target_loss,
            train_action_acc,
            train_target_acc,
        ) = train_epoch(
            args,
            model,
            loaders["train"],
            optimizer,
            action_criterion,
            target_criterion,
            device,
        )

        # some logging
        print(
            f"train action loss : {train_action_loss} | train target loss: {train_target_loss}"
        )
        print(
            f"train action acc : {train_action_acc} | train target acc: {train_target_acc}"
        )

        training_loss_list.append(train_action_loss+train_target_loss)
        training_acc_list.append(train_action_acc+train_target_acc)

        # run validation every so often
        # during eval, we run a forward pass through the model and compute
        # loss and accuracy but we don't update the model weights
        if epoch % args.val_every == 0:
            val_action_loss, val_target_loss, val_action_acc, val_target_acc = validate(
                args,
                model,
                loaders["val"],
                optimizer,
                action_criterion,
                target_criterion,
                device,
            )

            print(
                f"val action loss : {val_action_loss} | val target loss: {val_target_loss}"
            )
            print(
                f"val action acc : {val_action_acc} | val target accs: {val_target_acc}"
            )

            val_loss_list.append(train_action_loss+train_target_loss)
            val_acc_list.append(train_action_acc+train_target_acc)

    # ================== TODO: CODE HERE ================== #
    # Task: Implement some code to keep track of the model training and
    # evaluation loss. Use the matplotlib library to plot
    # 4 figures for 1) training loss, 2) training accuracy,
    # 3) validation loss, 4) validation accuracy
    # ===================================================== #

    axs[0,0].plot(range(args.num_epochs), training_loss_list)
    axs[0,0].set_title("Training Loss")
    axs[0,1].plot(range(args.num_epochs), training_acc_list, "tab:orange")
    axs[0,1].set_title("Training Accuracy")
    axs[1,0].plot(range(len(val_loss_list)), val_loss_list, "tab:green")
    axs[1,0].set_title("Validation Loss")
    axs[1,1].plot(range(len(val_acc_list)), val_acc_list, "tab:red")
    axs[1,1].set_title("Validation Accuracy")

    for ax in axs.flat:
        ax.set(xlabel="Epoch")

    fig.tight_layout()

    plt.savefig(args.model_output_dir)


def main(args):
    device = get_device(args.force_cpu)

    # get dataloaders
    train_loader, val_loader, maps, len_cutoff = setup_dataloader(args)
    loaders = {"train": train_loader, "val": val_loader}

    # build model
    model = setup_model(device, maps, len_cutoff, args.emb_dim)

    # get optimizer and loss functions
    action_criterion, target_criterion, optimizer = setup_optimizer(args, model)

    if args.eval:
        val_action_loss, val_target_loss, val_action_acc, val_target_acc = validate(
            args,
            model,
            loaders["val"],
            optimizer,
            action_criterion,
            target_criterion,
            device,
        )
    else:
        train(
            args, model, loaders, optimizer, action_criterion, target_criterion, device
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_data_fn", type=str, help="data file")
    parser.add_argument(
        "--model_output_dir", type=str, help="where to save model outputs"
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="size of each batch in loader"
    )
    parser.add_argument("--force_cpu", action="store_true", help="debug mode")
    parser.add_argument("--eval", action="store_true", help="run eval")
    parser.add_argument("--num_epochs", type=int, default=1000, help="number of training epochs")
    parser.add_argument(
        "--val_every", type=int, default=5, help="number of epochs between every eval loop"
    )

    # ================== TODO: CODE HERE ================== #
    # Task (optional): Add any additional command line
    # parameters you may need here
    # ===================================================== #
    parser.add_argument(
        "--emb_dim", type=int, default=128, help="number of 'columns' to learn for every feature vector"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=0.001, help="how often to stepr"
    )
    parser.add_argument(
        "--stemming", action="store_true", help="run with stemming"
    )
    args = parser.parse_args()

    main(args)
