import os
import torch
import random
import json
import numpy as np
import pickle as pkl
import torch.utils.data as data

from torch import nn
from torch.autograd import Variable
from copy import deepcopy


def load_kenlm():
    global kenlm
    import kenlm


def to_gpu(gpu, var):
    if gpu:
        return var.cuda()
    return var


class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.lvt = True
        self.word2idx['<pad>'] = 0
        self.word2idx['<sos>'] = 1
        self.word2idx['<eos>'] = 2
        self.word2idx['<oov>'] = 3
        self.wordcounts = {}

    # to track word counts
    def add_word(self, word):
        if word not in self.wordcounts:
            self.wordcounts[word] = 1
            if not self.lvt:
                self.word2idx[word] = len(self.word2idx)
        else:
            self.wordcounts[word] += 1

    # prune vocab based on count k cutoff or most frequently seen k words
    def prune_vocab(self, k=5, cnt=False):
        # get all words and their respective counts
        vocab_list = [(word, count) for word, count in self.wordcounts.items()]
        if cnt:
            # prune by count
            self.pruned_vocab = \
                {pair[0]: pair[1] for pair in vocab_list if pair[1] > k}
        else:
            # prune by most frequently seen words
            vocab_list.sort(key=lambda x: (x[1], x[0]), reverse=True)
            k = min(k, len(vocab_list))
            self.pruned_vocab = [pair[0] for pair in vocab_list[:k]]
        # sort to make vocabulary determistic
        self.pruned_vocab.sort()

        for word in self.pruned_vocab:
            if word not in self.word2idx:
                self.word2idx[word] = len(self.word2idx)
        print("original vocab {}; pruned to {}".
              format(len(self.wordcounts), len(self.word2idx)))
        self.idx2word = {v: k for k, v in self.word2idx.items()}

    def __len__(self):
        return len(self.word2idx)


class Corpus(object):
    def __init__(self, path, maxlen, vocab_size=11000, lowercase=False, load_vocab=None):
        self.dictionary = Dictionary()
        self.maxlen = maxlen
        self.lowercase = lowercase
        self.vocab_size = vocab_size
        self.train_path = os.path.join(path, 'train.txt')
        self.test_path = os.path.join(path, 'test.txt')

        # make the vocabulary from training set
        if load_vocab:
            self.dictionary.word2idx = json.load(open(load_vocab))
            self.dictionary.idx2word = {v: k for k, v in self.dictionary.word2idx.items()}
            print("Finished loading vocabulary from " + load_vocab + "\n")
        else:
            self.make_vocab()

        if os.path.exists(self.train_path):
            self.train = self.tokenize(self.train_path)
        if os.path.exists(self.test_path):
            self.test = self.tokenize(self.test_path)

    def get_indices(self, words):
        vocab = self.dictionary.word2idx
        unk_idx = vocab['<oov>']
        return torch.LongTensor([[vocab[w] if w in vocab else unk_idx for w in words]])

    def make_vocab(self):
        assert os.path.exists(self.train_path)
        # Add words to the dictionary
        with open(self.train_path, 'r') as f:
            for line in f:
                if self.lowercase:
                    # -1 to get rid of \n character
                    words = line[:-1].lower().split(" ")
                else:
                    words = line[:-1].split(" ")
                for word in words:
                    # word = word.decode('Windows-1252').encode('utf-8')
                    self.dictionary.add_word(word)

        # prune the vocabulary
        self.dictionary.prune_vocab(k=self.vocab_size, cnt=False)

    def tokenize(self, path):
        """Tokenizes a text file."""
        dropped = 0

        with open(path, 'r') as f:
            linecount = 0
            lines = []
            for line in f:
                linecount += 1
                if self.lowercase:
                    words = line[:-1].lower().strip().split(" ")
                else:
                    words = line[:-1].strip().split(" ")
                if len(words) > self.maxlen - 1:
                    dropped += 1
                    continue
                lens = len(words) + 1
                words = ['<sos>'] + words
                words += ['<eos>']

                # vectorize
                vocab = self.dictionary.word2idx
                unk_idx = vocab['<oov>']
                indices = [vocab[w] if w in vocab else unk_idx for w in words]
                lines.append((indices, lens))

        print("Number of sentences dropped from {}: {} out of {} total".
              format(path, dropped, linecount))
        return lines


def batchify(data, bsz, max_len, packed_rep=False, shuffle=False, gpu=False):
    if shuffle:
        random.shuffle(data)
    nbatch = len(data) // bsz
    batches = []

    for i in range(nbatch):
        batch, lengths = zip(*(data[i * bsz:(i + 1) * bsz]))

        batch, lengths = length_sort(batch, lengths)

        # source has no end symbol
        source = [x[:-1] for x in batch]
        # target has no start symbol
        target = [x[1:] for x in batch]

        # find length to pad to
        if packed_rep:
            maxlen = max(lengths)
        else:
            maxlen = max_len

        lengths = [min(x, max_len) for x in lengths]
        # lengths = [max(x, max_len) for x in lengths]

        count = 0
        for x, y in zip(source, target):
            if len(x) > maxlen:
                source[count] = x[:maxlen]
            if len(y) > maxlen:
                target[count] = y[:maxlen]

            zeros = (maxlen - len(x)) * [0]
            x += zeros
            y += zeros
            count += 1

        source = torch.LongTensor(np.array(source))
        target = torch.LongTensor(np.array(target)).view(-1)

        if gpu:
            source = source.cuda()
            target = target.cuda()

        batches.append((source, target, lengths))

    return batches


def length_sort(items, lengths, descending=True):
    """In order to use pytorch variable length sequence package"""
    items = list(zip(items, lengths))
    items.sort(key=lambda x: x[1], reverse=True)
    items, lengths = zip(*items)
    return list(items), list(lengths)


def train_ngram_lm(kenlm_path, data_path, output_path, N):
    """
    Trains a modified Kneser-Ney n-gram KenLM from a text file.
    Creates a .arpa file to store n-grams.
    """
    # create .arpa file of n-grams
    curdir = os.path.abspath(os.path.curdir)
    #
    command = "bin/lmplz -o " + str(N) + " <" + os.path.join(curdir, data_path) + \
              " >" + os.path.join(curdir, output_path)
    os.system("cd " + os.path.join(kenlm_path, 'build') + " && " + command)

    load_kenlm()
    # create language model
    model = kenlm.Model(output_path)

    return model


def load_ngram_lm(model_path):
    load_kenlm()
    model = kenlm.Model(model_path)
    return model


def get_ppl(lm, sentences):
    """
    Assume sentences is a list of strings (space delimited sentences)
    """
    total_nll = 0
    total_wc = 0
    for sent in sentences:
        words = sent.strip().split()
        score = lm.score(sent, bos=True, eos=False)
        word_count = len(words)
        total_wc += word_count
        total_nll += score
    ppl = 10 ** -(total_nll / total_wc)
    return ppl


class SNLIDataset(data.Dataset):

    def __init__(self, path="./data/classifier", train=True,
                 vocab_size=11000, maxlen=10, reset_vocab=None, attack_label=None):
        self.train = train
        self.train_data = []
        self.test_data = []
        self.root = path
        self.train_path = os.path.join(path, 'train.txt')
        self.test_path = os.path.join(path, 'test.txt')
        self.lowercase = True
        self.sentence_path = path + "/sentences.dlnlp"
        self.dictionary = Dictionary()
        self.sentence_ids = {}
        self.vocab_size = vocab_size
        self.labels = {'entailment': 0, 'neutral': 1, 'contradiction': 2}
        self.maxlen = maxlen
        self.attack_label = attack_label

        if reset_vocab:
            self.dictionary.word2idx = deepcopy(reset_vocab)
            self.dictionary.idx2word = {v: k for k, v in self.dictionary.word2idx.items()}
        else:
            self.make_vocab()

        if os.path.exists(self.root + "/sent_ids.pkl"):
            self.sentence_ids = pkl.load(open(self.root + "/sent_ids.pkl", 'rb'))
        else:
            print("Sentence IDs not found!!")

        if self.train and os.path.exists(self.train_path):
            self.train_data = self.tokenize(self.train_path)
        if (not self.train) and os.path.exists(self.test_path):
            self.test_data = self.tokenize(self.test_path)

    def __getitem__(self, index):
        if self.train:
            return self.train_data[index]
        else:
            return self.test_data[index]

    def __len__(self):
        if self.train:
            return len(self.train_data)
        else:
            return len(self.test_data)

    def make_vocab(self):
        with open(self.sentence_path, 'r') as f:
            for lines in f:
                toks = lines.strip().split('\t')
                self.sentence_ids[toks[0]] = toks[1].strip()
                line = self.sentence_ids[toks[0]]
                if self.lowercase:
                    # -1 to get rid of \n character
                    words = line.strip().lower().split(" ")
                else:
                    words = line.strip().split(" ")
                for word in words:
                    # word = word.decode('Windows-1252').encode('utf-8')
                    self.dictionary.add_word(word)

        self.dictionary.prune_vocab(k=self.vocab_size, cnt=False)
        with open(self.root + "/sent_ids.pkl", 'wb') as f:
            pkl.dump(self.sentence_ids, f)
        with open(self.root + "/vocab_" + str(len(self.dictionary.word2idx)) + ".pkl", 'wb') as f:
            pkl.dump(self.dictionary.word2idx, f)
        # pkl.dump(self.dictionary.word2idx, open(self.root+"/vocab_"+str(len(self.dictionary.word2idx))+".pkl", 'wb'))

    def get_indices(self, words):
        vocab = self.dictionary.word2idx
        unk_idx = vocab['<oov>']
        return torch.LongTensor([[vocab[w] if w in vocab else unk_idx for w in words]])

    def tokenize(self, path):
        """Tokenizes a text file."""
        dropped = 0
        with open(path, 'r') as f:
            linecount = 0
            lines = []
            for line in f:

                tokens = line.strip().split('\t')
                label = self.labels[tokens[0]]
                if self.attack_label is None or (
                        self.attack_label is not None and label == self.attack_label):  # 找我们要攻击的正的案例
                    linecount += 1

                    premise = self.sentence_ids[tokens[1]]
                    hypothesis = self.sentence_ids[tokens[2]]

                    if self.lowercase:
                        hypothesis = hypothesis.strip().lower()
                        premise = premise.strip().lower()

                    premise_words = premise.strip().split(" ")
                    hypothesis_words = hypothesis.strip().split(" ")

                    if ((len(premise_words) > self.maxlen - 1) or \
                            (len(hypothesis_words) > self.maxlen - 1)):
                        dropped += 1
                        continue
                    premise_words = ['<sos>'] + premise_words
                    premise_words += ['<eos>']
                    hypothesis_words = ['<sos>'] + hypothesis_words
                    # hypothesis_words += ['<eos>']

                    # vectorize
                    vocab = self.dictionary.word2idx
                    unk_idx = vocab['<oov>']
                    hypothesis_indices = [vocab[w] if w in vocab else unk_idx for w in hypothesis_words]
                    premise_indices = [vocab[w] if w in vocab else unk_idx for w in premise_words]
                    premise_words = [w if w in vocab else '<oov>' for w in premise_words]
                    hypothesis_words = [w if w in vocab else '<oov>' for w in hypothesis_words]
                    hypothesis_length = min(len(hypothesis_words), self.maxlen)
                    premise_length = min(len(premise_words), self.maxlen)
                    # hypothesis_length = max(hypothesis_length, self.maxlen)

                    if len(premise_indices) < self.maxlen:
                        premise_indices += [0] * (self.maxlen - len(premise_indices))
                        premise_words += ["<pad>"] * (self.maxlen - len(premise_words))
                    #
                    if len(hypothesis_indices) < self.maxlen:
                        hypothesis_indices += [0] * (self.maxlen - len(hypothesis_indices))
                        hypothesis_words += ["<pad>"] * (self.maxlen - len(hypothesis_words))
                    #
                    premise_indices = premise_indices[:self.maxlen]
                    hypothesis_indices = hypothesis_indices[:self.maxlen]
                    premise_words = premise_words[:self.maxlen]
                    hypothesis_words = hypothesis_words[:self.maxlen]

                    lines.append([premise_indices, hypothesis_indices, label, hypothesis_length, premise_length])


        print("Number of sentences dropped from {}: {} out of {} total".
              format(path, dropped, linecount))
        return lines

    def build_embedding_matrix(self, embeddings_file):
        """
        Build an embedding matrix with pretrained weights for object's
        worddict.

        Args:
            embeddings_file: A file containing pretrained word embeddings.

        Returns:
            A numpy matrix of size (num_words+n_special_tokens, embedding_dim)
            containing pretrained word embeddings (the +n_special_tokens is for
            the padding and out-of-vocabulary tokens, as well as BOS and EOS if
            they're used).
        """
        # Load the word embeddings in a dictionnary.
        embeddings = {}
        with open(embeddings_file, "r", encoding="utf8") as input_data:
            for line in input_data:
                line = line.split()

                try:
                    # Check that the second element on the line is the start
                    # of the embedding and not another word. Necessary to
                    # ignore multiple word lines.
                    float(line[1])
                    word = line[0]
                    if word in self.dictionary.word2idx:
                        embeddings[word] = line[1:]

                # Ignore lines corresponding to multiple words separated
                # by spaces.
                except ValueError:
                    continue

        num_words = len(self.dictionary.word2idx)
        embedding_dim = len(list(embeddings.values())[0])
        embedding_matrix = np.zeros((num_words, embedding_dim))

        # Actual building of the embedding matrix.
        missed = 0
        for word, i in self.dictionary.word2idx.items():
            if word in embeddings:
                embedding_matrix[i] = np.array(embeddings[word], dtype=float)
            else:
                print(word)
                if word == "<pad>":
                    continue
                missed += 1
                # Out of vocabulary words are initialised with random gaussian
                # samples.
                embedding_matrix[i] = np.random.normal(size=(embedding_dim))
        print("Missed words: ", missed)
        return embedding_matrix


def load_embeddings(root='./data/classifier/', root1='./output/1593075369/'):
    vocab_path = root1 + 'vocab.json'
    file_path = root + 'embeddings'
    vocab = json.load(open(vocab_path, "rb"))

    embeddings = torch.FloatTensor(len(vocab), 100).uniform_(-0.1, 0.1)
    embeddings[0].fill_(0)
    embeddings[1].copy_(torch.FloatTensor(
        list(
            map(float, open(file_path).read().split('\n')[0].strip().split(" ")[1:])))
    )
    embeddings[2].copy_(embeddings[1])
    with open(file_path) as fr:
        for line in fr:
            elements = line.strip().split(" ")
            word = elements[0]
            emb = torch.FloatTensor(list(map(float, elements[1:])))
            if word in vocab:
                embeddings[vocab[word]].copy_(emb)
    return embeddings


def get_delta(tensor, right, z_range, gpu):
    bs, dim = tensor.size()
    num_sample = bs * dim
    tensor1 = np.random.uniform(-1 * right, -1 * right + z_range, num_sample).tolist()
    tensor2 = np.random.uniform(right - z_range, right, num_sample).tolist()
    samples_delta_z = list(set(tensor1).union(set(tensor2)))
    random.shuffle(samples_delta_z)
    samples_delta_z = to_gpu(gpu, torch.FloatTensor(samples_delta_z[:num_sample]).view(bs, dim))
    return samples_delta_z


# x, y, z ,l
def length_sort_hypothesis(batch, descending=True):
    """In order to use pytorch variable length sequence package"""
    premise, hypothesis, labels, lengths = zip(*(batch))
    items = list(zip(premise, hypothesis, labels, lengths))
    items.sort(key=lambda x: x[3], reverse=True)
    return zip(*items)


def collate_snli(batch):
    premise = []
    hypothesis = []
    labels = []
    hypothesis_length = []
    premise_length = []
    premise_words = []
    hypothesis_words = []
    if len(batch[0]) == 5:
        for b in batch:
            x, y, z, l, m = b
            premise.append(x)
            hypothesis.append(y)
            labels.append(z)
            hypothesis_length.append(l)
            premise_length.append(m)
        return Variable(torch.LongTensor(premise)), Variable(torch.LongTensor(hypothesis)), Variable(
            torch.LongTensor(labels)), \
               torch.LongTensor(hypothesis_length), torch.LongTensor(premise_length)
    else:
        print("sentence length doesn't match")


def one_hot_prob(y, ind):
    # covert the probability to one-hot coding in a differentiable way.
    shape = y.size()
    y_hard = torch.zeros_like(y).view(-1, shape[-1])
    y_hard.scatter_(1, ind.view(-1, 1), 1)
    y_hard = y_hard.view(*shape)
    y_hard = (y_hard - y).detach() + y
    return y_hard


# https://github.com/allenai/allennlp/blob/master/allennlp/nn/util.py.
def sort_by_seq_lens(batch, sequences_lengths, descending=True):
    """
    Sort a batch of padded variable length sequences by their length.

    Args:
        batch: A batch of padded variable length sequences. The batch should
            have the dimensions (batch_size x max_sequence_length x *).
        sequences_lengths: A tensor containing the lengths of the sequences in the
            input batch. The tensor should be of size (batch_size).
        descending: A boolean value indicating whether to sort the sequences
            by their lengths in descending order. Defaults to True.

    Returns:
        sorted_batch: A tensor containing the input batch reordered by
            sequences lengths.
        sorted_seq_lens: A tensor containing the sorted lengths of the
            sequences in the input batch.
        sorting_idx: A tensor containing the indices used to permute the input
            batch in order to get 'sorted_batch'.
        restoration_idx: A tensor containing the indices that can be used to
            restore the order of the sequences in 'sorted_batch' so that it
            matches the input batch.
    """
    sorted_seq_lens, sorting_index = \
        sequences_lengths.sort(0, descending=descending)

    sorted_batch = batch.index_select(0, sorting_index)

    idx_range = \
        sequences_lengths.new_tensor(torch.arange(0, len(sequences_lengths)))
    _, reverse_mapping = sorting_index.sort(0, descending=False)
    restoration_index = idx_range.index_select(0, reverse_mapping)

    return sorted_batch, sorted_seq_lens, sorting_index, restoration_index


def get_mask(sequences_batch, sequences_lengths):
    """
    Get the mask for a batch of padded variable length sequences.

    Args:
        sequences_batch: A batch of padded variable length sequences
            containing word indices. Must be a 2-dimensional tensor of size
            (batch, sequence).
        sequences_lengths: A tensor containing the lengths of the sequences in
            'sequences_batch'. Must be of size (batch).

    Returns:
        A mask of size (batch, max_sequence_length), where max_sequence_length
        is the length of the longest sequence in the batch.
    """
    batch_size = sequences_batch.size()[0]
    max_length = torch.max(sequences_lengths)
    mask = torch.ones(batch_size, max_length, dtype=torch.float)
    mask[sequences_batch[:, :max_length] == 0] = 0.0
    return mask


# Code widely inspired from:
# https://github.com/allenai/allennlp/blob/master/allennlp/nn/util.py.
def masked_softmax(tensor, mask):
    """
    Apply a masked softmax on the last dimension of a tensor.
    The input tensor and mask should be of size (batch, *, sequence_length).

    Args:
        tensor: The tensor on which the softmax function must be applied along
            the last dimension.
        mask: A mask of the same size as the tensor with 0s in the positions of
            the values that must be masked and 1s everywhere else.

    Returns:
        A tensor of the same size as the inputs containing the result of the
        softmax.
    """
    tensor_shape = tensor.size()
    reshaped_tensor = tensor.view(-1, tensor_shape[-1])

    # Reshape the mask so it matches the size of the input tensor.
    while mask.dim() < tensor.dim():
        mask = mask.unsqueeze(1)
    mask = mask.expand_as(tensor).contiguous().float()
    reshaped_mask = mask.view(-1, mask.size()[-1])

    result = nn.functional.softmax(reshaped_tensor * reshaped_mask, dim=-1)
    result = result * reshaped_mask
    # 1e-13 is added to avoid divisions by zero.
    result = result / (result.sum(dim=-1, keepdim=True) + 1e-13)

    return result.view(*tensor_shape)


# Code widely inspired from:
# https://github.com/allenai/allennlp/blob/master/allennlp/nn/util.py.
def weighted_sum(tensor, weights, mask):
    """
    Apply a weighted sum on the vectors along the last dimension of 'tensor',
    and mask the vectors in the result with 'mask'.

    Args:
        tensor: A tensor of vectors on which a weighted sum must be applied.
        weights: The weights to use in the weighted sum.
        mask: A mask to apply on the result of the weighted sum.

    Returns:
        A new tensor containing the result of the weighted sum after the mask
        has been applied on it.
    """
    weighted_sum = weights.bmm(tensor)

    while mask.dim() < weighted_sum.dim():
        mask = mask.unsqueeze(1)
    mask = mask.transpose(-1, -2)
    mask = mask.expand_as(weighted_sum).contiguous().float()

    return weighted_sum * mask


# Code inspired from:
# https://github.com/allenai/allennlp/blob/master/allennlp/nn/util.py.
def replace_masked(tensor, mask, value):
    """
    Replace the all the values of vectors in 'tensor' that are masked in
    'masked' by 'value'.

    Args:
        tensor: The tensor in which the masked vectors must have their values
            replaced.
        mask: A mask indicating the vectors which must have their values
            replaced.
        value: The value to place in the masked vectors of 'tensor'.

    Returns:
        A new tensor of the same size as 'tensor' where the values of the
        vectors masked in 'mask' were replaced by 'value'.
    """
    mask = mask.unsqueeze(1).transpose(2, 1)
    reverse_mask = 1.0 - mask
    values_to_add = value * reverse_mask
    return tensor * mask + values_to_add
