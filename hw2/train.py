import argparse
from operator import index
import os
import tqdm
import torch
# from sklearn.metrics import accuracy_score

from eval_utils import downstream_validation
import utils
import data_utils

# Packages added
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import TensorDataset, DataLoader
import model

# Skip-gram, so need to return words and their context
# words as input, context as output
def create_inputs(encoded_sentences, window=4):
    words = []
    contexts = []
    
    print("creating inputs")
    i = 0
    for sentence in encoded_sentences:       
        i += 1
        if i % 1000 == 0:
            print(i)
        for idx in range(window, len(sentence)-window-1):
            
            word = sentence[idx]
            context = np.concatenate((sentence[idx-window:idx], sentence[idx+1: idx+window+1]))
            
            # exclude word/context pairs that includes padding to save on memory
            if 0 not in context and word != 0:
                words.append(word)
                # context = np.array(context)
                contexts.append(context)

    print("finished!")
    words = np.array(words)
    contexts = np.array(contexts)            
    return words, contexts

def setup_dataloader(args):
    """
    return:
        - train_loader: torch.utils.data.Dataloader
        - val_loader: torch.utils.data.Dataloader
    """

    if args.save_encodings:

        # read in training data from books dataset
        sentences = data_utils.process_book_dir(args.data_dir)

        # build one hot maps for input and output
        (
            vocab_to_index,
            index_to_vocab,
            suggested_padding_len,
        ) = data_utils.build_tokenizer_table(sentences, vocab_size=args.vocab_size)

        # create encoded input and output numpy matrices for the entire dataset and then put them into tensors
        encoded_sentences, lens = data_utils.encode_data(
            sentences,
            vocab_to_index,
            suggested_padding_len,
        )

        # ================== TODO: CODE HERE ================== #
        # Task: Given the tokenized and encoded text, you need to
        # create inputs to the LM model you want to train.
        # E.g., could be target word in -> context out or
        # context in -> target word out.
        # You can build up that input/output table across all
        # encoded sentences in the dataset!
        # Then, split the data into train set and validation set
        # (you can use utils functions) and create respective
        # dataloaders.
        # ===================================================== #

        encoded_words, encoded_context = create_inputs(encoded_sentences, args.window_size)
        
        # store these values for use later
        with open("encoded_words.txt", "w") as f_words:
            for word in encoded_words:
                f_words.write(str(word) + "\n")

        with open("encoded_context.txt", "w") as f_context:
            for context in encoded_context:
                for word in context:
                    f_context.write(str(word) + " ")
                f_context.write("\n")    

        with open("index_to_vocab.txt", "w") as f_i2v:
            for key, value in index_to_vocab.items():
                f_i2v.write(str(key) + ":" + str(value) + "\n")

    # use stored encodings
    else:

        index_to_vocab = {}
        encoded_words = []
        encoded_context = []

        # i = 0
        with open("encoded_words.txt", "r") as f_words:
            for word in f_words:
                word = word.strip()  
                encoded_words.append(word)
                # i += 1
                # if i == 1000:
                #     break

        # i = 0
        with open("encoded_context.txt", "r") as f_context:
            for context in f_context:
                context = context.strip()
                context = context.split(" ")
                # transform string back to int
                context = list(map(int, context))
                context = np.array(context)
                encoded_context.append(context)
                # i += 1
                # if i == 1000:
                #     break

        with open("index_to_vocab.txt", "r") as f_i2v:
            for pair in f_i2v:
                pair = pair.strip()
                pair = pair.split(":")
                index_to_vocab[int(pair[0])] = pair[1] 

        # transform string back to int
        encoded_words = list(map(int, encoded_words))
        encoded_words = np.array(encoded_words)
        encoded_contex = np.array(encoded_context)  

    # ok to split down here because we don't care if the val set knows information from training in this case since our accuracy is based on analogies
    x_train, x_test, y_train, y_test = train_test_split(encoded_words, encoded_context, test_size=args.val_size, shuffle=True)

    # convert everything to tensors
    train_dataset = TensorDataset(torch.from_numpy(np.array(x_train)), torch.from_numpy(np.array(y_train)))
    val_dataset = TensorDataset(torch.from_numpy(np.array(x_test)), torch.from_numpy(np.array(y_test)))

    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch_size)
    val_loader = DataLoader(val_dataset, shuffle=True, batch_size=args.batch_size)
    return train_loader, val_loader, index_to_vocab


def setup_model(args, device):
    """
    return:
        - model: YourOwnModelClass
    """
    # ================== TODO: CODE HERE ================== #
    # Task: Initialize your CBOW or Skip-Gram model.
    # ===================================================== #

    m = model.SkipGram(
        device,
        args.vocab_size,
        args.emb_dim
    )
    return m


def setup_optimizer(args, model):
    """
    return:
        - criterion: loss_fn
        - optimizer: torch.optim
    """
    # ================== TODO: CODE HERE ================== #
    # Task: Initialize the loss function for predictions. 
    # Also initialize your optimizer.
    # ===================================================== #

    # BCE to account for multilabel predictions (context window means we're predicting multiple words)
    context_criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.learning_rate)
    return context_criterion, optimizer

# our own accuracy calculation using Intersection over Union (IoU)
def IoU_accuracy(pred, actual):
    # pred and accuracy are matrices with dimensions (batch_size, |vocab|)
    # each row represents a word, each column has the predicted probabilities for the context
    avg_accuracy = []

    for idx in range(len(pred)):
        # get the indices (which correspond to tokenized words)
        actual_context = [i for i, x in enumerate(actual[idx]) if x == 1]
        actual_context = set(actual_context)

        # get the top (num distinct context predictions) predicted probabilities
        # that's because a context might contain multiple of the same word, e.g. "the very very big bad wolf"
        # so we see how many distinct words there actually are, then compare to see if our highest probabilities match them
        pred_top_n = sorted(range(len(pred[idx])), key=lambda i: pred[idx][i])[-len(actual_context):]

        intersection = actual_context.intersection(pred_top_n)
        union = actual_context.union(pred_top_n)

        # append IoU accuracy for every word individually
        avg_accuracy.append(len(intersection)/len(union))

    # calculate average accuracy for this batch
    return sum(avg_accuracy)/len(avg_accuracy)

def train_epoch(
    args,
    model,
    loader,
    optimizer,
    criterion,
    device,
    training=True,
):
    model.train()
    epoch_loss = 0.0

    # keep track of the model predictions for computing accuracy
    pred_labels = []
    target_labels = []

    # iterate over each batch in the dataloader
    # NOTE: you may have additional outputs from the loader __getitem__, you can modify this
    for (inputs, labels) in tqdm.tqdm(loader):
        # put model inputs to device
        inputs, labels = inputs.to(device).long(), labels.to(device).long()

        # calculate the loss and train accuracy and perform backprop
        # NOTE: feel free to change the parameters to the model forward pass here + outputs
        # pred_logits should be a matrix with dimensions batch_size x |vocab| with multiple 1's in each row, each representing a context word.
        # pred_logits = model(inputs, labels)
        pred_logits = model(inputs)
        # transform labels from B x (2*context_window) to B x |V|
        labels = data_utils.token_to_multihot(labels, args.vocab_size)

        # calculate prediction loss
        loss = criterion(pred_logits, labels)

        # step optimizer and compute gradients during training
        if training:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # logging
        epoch_loss += loss.item()

        # compute metrics
       # preds = pred_logits.argmax(-1)
        #pred_labels.extend(preds.cpu().numpy())
        pred_labels.extend(pred_logits.detach().cpu().numpy())
        target_labels.extend(labels.cpu().numpy())

    # calculate our own accuracy
    # acc = accuracy_score(pred_labels, target_labels)
    # print("\npred: " + str(pred_labels))
    # print(str(type(pred_labels)) + "\n")
    # print("\ntarget: " + str(target_labels))
    # print(str(type(target_labels)) + "\n")
    acc = IoU_accuracy(pred_labels, target_labels)
    epoch_loss /= len(loader)

    return epoch_loss, acc


def validate(args, model, loader, optimizer, criterion, device):
    # set model to eval mode
    model.eval()

    # don't compute gradients
    with torch.no_grad():
        val_loss, val_acc = train_epoch(
            args,
            model,
            loader,
            optimizer,
            criterion,
            device,
            training=False,
        )

    return val_loss, val_acc


def main(args):
    device = utils.get_device(args.force_cpu)

    # load analogies for downstream eval
    external_val_analogies = utils.read_analogies(args.analogies_fn)

    if args.downstream_eval:
        word_vec_file = os.path.join(args.outputs_dir, args.word_vector_fn)
        assert os.path.exists(word_vec_file), "need to train the word vecs first!"
        downstream_validation(word_vec_file, external_val_analogies)
        return

    # get dataloaders
    train_loader, val_loader, i2v = setup_dataloader(args)
    loaders = {"train": train_loader, "val": val_loader}

    # build model
    model = setup_model(args, device)
    print(model)

    # get optimizer
    criterion, optimizer = setup_optimizer(args, model)

    # store validation accuracies
    val_accuracies = []

    for epoch in range(args.num_epochs):
        # train model for a single epoch
        print(f"Epoch {epoch}")
        train_loss, train_acc = train_epoch(
            args,
            model,
            loaders["train"],
            optimizer,
            criterion,
            device,
        )

        print(f"train loss : {train_loss} | train acc: {train_acc}")

        if epoch % args.val_every == 0:
            val_loss, val_acc = validate(
                args,
                model,
                loaders["val"],
                optimizer,
                criterion,
                device,
            )
            print(f"val loss : {val_loss} | val acc: {val_acc}")
            val_accuracies.append(val_acc)

            # ======================= NOTE ======================== #
            # Saving the word vectors to disk and running the eval
            # can be costly when you do it multiple times. You could
            # change this to run only when your training has concluded.
            # However, incremental saving means if something crashes
            # later or you get bored and kill the process you'll still
            # have a word vector file and some results.
            # ===================================================== #

            # save word vectors
            word_vec_file = os.path.join(args.outputs_dir, args.word_vector_fn)
            print("saving word vec to ", word_vec_file)
            utils.save_word2vec_format(word_vec_file, model, i2v)

            # evaluate learned embeddings on a downstream task
            downstream_validation(word_vec_file, external_val_analogies)

        if epoch % args.save_every == 0:
            ckpt_file = os.path.join(args.outputs_dir, "model.ckpt")
            print("saving model to ", ckpt_file)
            torch.save(model, ckpt_file)

    with open("val_accuracy.txt", "w") as f_validation_accs:
        for i in val_accuracies:
            f_validation_accs.write(str(i)+"\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs_dir", type=str, default="training_output", help="where to save training outputs")
    parser.add_argument("--data_dir", type=str, help="where the book dataset is stored")
    parser.add_argument(
        "--downstream_eval",
        action="store_true",
        help="run downstream eval on trained word vecs",
    )
    # ======================= NOTE ======================== #
    # If you adjust the vocab_size down below 3000, there 
    # may be analogies in the downstream evaluation that have
    # words that are not in your vocabulary, resulting in
    # automatic (currently) zero score for an ABCD where one
    # of A, B, C, or D is not in the vocab. A visible warning
    # will be generated by the evaluation loop for these examples.
    # ===================================================== #
    parser.add_argument(
        "--vocab_size", type=int, default=3000, help="size of vocabulary"
    )
    parser.add_argument(
        "--batch_size", type=int, default=32, help="size of each batch in loader"
    )
    parser.add_argument("--force_cpu", action="store_true", help="debug mode")
    parser.add_argument(
        "--analogies_fn", type=str, help="filepath to the analogies json file"
    )
    parser.add_argument(
        "--word_vector_fn", type=str, help="filepath to store the learned word vectors",
        default='learned_word_vectors.txt'
    )
    parser.add_argument(
        "--num_epochs", default=30, type=int, help="number of training epochs"
    )
    parser.add_argument(
        "--val_every",
        default=5,
        type=int,
        help="number of epochs between every eval loop",
    )
    parser.add_argument(
        "--save_every",
        default=5,
        type=int,
        help="number of epochs between saving model checkpoint",
    )
    # ================== TODO: CODE HERE ================== #
    # Task (optional): Add any additional command line
    # parameters you may need here
    # ===================================================== #
    parser.add_argument(
        "--save_encodings",
        action="store_true",
        help="calling this flag re-encodes and saves encodings as opposed to using stored encodings"
    )

    parser.add_argument(
        "--val_size",
        default=0.3,
        type=float,
        help="proportion of total data to put in validation expressed as a decimal"
    )

    parser.add_argument(
        "--emb_dim", type=int, default=128, help="number of 'columns' to learn for every feature vector"
    )

    parser.add_argument(
        "--learning_rate", type=float, default=0.001, help="how often to stepr"
    )

    parser.add_argument(
        "--window_size",
        default=4,
        type=int,
        help="number of words to look on each side of the current word"
    )

    args = parser.parse_args()
    main(args)
