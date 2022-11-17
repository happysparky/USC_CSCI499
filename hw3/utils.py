import re
import torch
from collections import Counter
import torch.nn.functional as functional


def get_device(force_cpu, status=True):
    # if not force_cpu and torch.backends.mps.is_available():
    # 	device = torch.device('mps')
    # 	if status:
    # 		print("Using MPS")
    # elif not force_cpu and torch.cuda.is_available():
    if not force_cpu and torch.cuda.is_available():
        device = torch.device("cuda")
        if status:
            print("Using CUDA")
    else:
        device = torch.device("cpu")
        if status:
            print("Using CPU")
    return device


def preprocess_string(s):
    # remove beginning and trailing whitespace
    s = s.strip()
    # Remove all non-word characters (everything except numbers and letters)
    s = re.sub(r"[^\w\s]", "", s)
    # Replace all runs of whitespaces with one space
    s = re.sub(r"\s+", " ", s)
    # replace digits with no space
    s = re.sub(r"\d", "", s)
    return s


def build_tokenizer_table(train, vocab_size=1000):
    word_list = []

    # store the longest length of all words from each instruction in an episode

    max_len = 0
    for episode in train:
        curr_max_len = 0
        for inst, _ in episode:
            inst = preprocess_string(inst)
            padded_len = 0
            for word in inst.lower().split():
                if len(word) > 0:
                    word_list.append(word)
                    padded_len += 1
                    curr_max_len += 1

        if curr_max_len > max_len:
            max_len = curr_max_len

    corpus = Counter(word_list)
    corpus_ = sorted(corpus, key=corpus.get, reverse=True)[
        : vocab_size - 4
    ]  # save room for <pad>, <start>, <end>, and <unk>
    vocab_to_index = {w: i + 4 for i, w in enumerate(corpus_)}
    vocab_to_index["<pad>"] = 0
    vocab_to_index["<start>"] = 1
    vocab_to_index["<end>"] = 2
    vocab_to_index["<unk>"] = 3
    index_to_vocab = {vocab_to_index[w]: w for w in vocab_to_index}
    return (
        vocab_to_index,
        index_to_vocab,
        max_len+2 # start and end 
        # int(np.average(padded_lens) + np.std(padded_lens) * 2 + 0.5),
    )


def build_output_tables(train):
    actions = set()
    targets = set()
    for episode in train:
        for _, outseq in episode:
            a, t = outseq
            actions.add(a)
            targets.add(t)


    actions_to_index = {a: i+3 for i, a in enumerate(actions)}
    actions_to_index["<pad>"] = 0
    actions_to_index["<bos>"] = 1
    actions_to_index["<eos>"] = 2
    targets_to_index = {t: i+2 for i, t in enumerate(targets)}
    targets_to_index["<pad>"] = 0
    targets_to_index["<bos>"] = 1
    targets_to_index["<eos>"] = 2
    index_to_actions = {actions_to_index[a]: a for a in actions_to_index}
    index_to_targets = {targets_to_index[t]: t for t in targets_to_index}
    return actions_to_index, index_to_actions, targets_to_index, index_to_targets

def prefix_match(pred_actions, pred_targets, true_actions, true_targets):
    # predicted and gt are sequences of (action, target) labels, the sequences should be of same length
    # computes how many matching (action, target) labels there are between predicted and gt
    # is a number between 0 and 1 

    # I spent four hours figuring out how to properly do this matrix transformation i think i'm going insane 

    # true_actions is of dimensions: batch_size x seq_len
    # need to convert to same dimensions as pred_actions, which is: batch_size x seq_length x actions_size
    # since the last dimension is a one hot encoding of true_actions's second dimension

    true_actions = torch.unsqueeze(true_actions, dim=2)
    true_targets = torch.unsqueeze(true_targets, dim=2)
    true_actions = functional.one_hot(true_actions.max(dim=2).values.to(torch.int64), 11)
    true_targets = functional.one_hot(true_targets.max(dim=2).values.to(torch.int64), 83)

    # must be the same datatype, and can't cast the true_actions/prediciton to floats for whatever reason 
    pred_actions = pred_actions.type(torch.int64)
    pred_targets = pred_targets.type(torch.int64)

    assert pred_actions.size() == true_actions.size(), "actions not the same shape"
    assert pred_targets.size() == true_targets.size(), "targets not the same shape"

    avg_prefix_match = 0
    for idx in range(pred_actions.size(0)):
        prefix_match = 0
        for jdx in range(pred_actions.size(1)):
            # print("here")
            # print()
            # print(pred_actions[idx, jdx, :])
            # print(true_actions[idx, jdx, :])
            if (not torch.equal(true_actions[idx, jdx, :], pred_actions[idx, jdx, :])) or (not torch.equal(true_targets[idx, jdx, :], pred_targets[idx, jdx, :])):
                break
            prefix_match += 1
        prefix_match /= pred_actions.size(1)
        avg_prefix_match += prefix_match
    avg_prefix_match /= pred_actions.size(0)

    return avg_prefix_match