import argparse
import json
import pickle
import random
import torch
import numpy as np
import torch.nn as nn
import torch.utils.data

from models import ESIM
from utils import SNLIDataset, collate_snli


def evaluate_model(model, testloader, criterion):
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_index, batch in enumerate(testloader):
            premises, hypotheses, labels, hypotheses_lengths, premises_lengths = batch
            premises = premises.cuda()
            hypotheses = hypotheses.cuda()
            hypotheses_lengths = hypotheses_lengths.cuda()
            premises_lengths = premises_lengths.cuda()
            labels = labels.cuda()
            logits, probs = model(premises,
                                  premises_lengths,
                                  hypotheses,
                                  hypotheses_lengths)
            loss = criterion(logits, labels)
            running_loss += loss.item()
            predictions = np.argmax(probs.data.cpu().numpy(), 1)
            correct += len(np.where(labels.data.cpu().numpy() == predictions)[0])
            total += premises.size(0)
        acc = correct / float(total)
        print("Accuracy:{0}".format(acc))
        return acc


parser = argparse.ArgumentParser(description='PyTorch baseline for Text')
parser.add_argument('--data_path', type=str, default='./data/classifier',
                    help='location of the data corpus')
parser.add_argument('--model_type', type=str, default="esim",
                    help='location of the data corpus')
parser.add_argument('--epochs', type=int, default=30,
                    help='maximum number of epochs')
parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                    help='batch size')
parser.add_argument('--train_mode', type=bool, default=True,
                    help='set training mode')
parser.add_argument('--maxlen', type=int, default=10,
                    help='maximum sentence length')
parser.add_argument('--lr', type=float, default=1e-05,
                    help='learning rate')
parser.add_argument('--seed', type=int, default=1111,
                    help='seed')
parser.add_argument('--cuda', action='store_true', default=True,
                    help='use CUDA')
parser.add_argument('--save_path', type=str, default='./models/baseline',
                    help='used for saving the models')
parser.add_argument('--vocab_size', type=int, default=11000,
                    help='vocabulary size')
parser.add_argument('--vocab_path', type=str, default='./output/1593075369/vocab.json',
                    help='vocabulary size')
parser.add_argument('--hidden_size', type=int, default=300,
                    help='hidden  size')
parser.add_argument('--dropout', type=int, default=0.5,
                    help='drop_out')
args = parser.parse_args()

word2idx = json.load(open(args.vocab_path, "rb"))
# model_idx2word = {v: k for k, v in model_word2idx.items()}
corpus_train = SNLIDataset(train=True, vocab_size=args.vocab_size, path=args.data_path, reset_vocab=word2idx)
corpus_test = SNLIDataset(train=False, vocab_size=args.vocab_size, path=args.data_path, reset_vocab=word2idx)

# embed_matrix = corpus_train.build_embedding_matrix('./data/embeddings/glove.840B.300d.txt')
embeddings_file = './data/embeddings/embeddings.pkl'
with open(embeddings_file, "rb") as pkl:
    embeddings = torch.tensor(pickle.load(pkl), dtype=torch.float)

trainloader = torch.utils.data.DataLoader(corpus_train, batch_size=args.batch_size, collate_fn=collate_snli,
                                          shuffle=True)

testloader = torch.utils.data.DataLoader(corpus_test, batch_size=args.batch_size, collate_fn=collate_snli,
                                         shuffle=False)

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = ESIM(embeddings.shape[0],
             embeddings.shape[1],
             args.hidden_size,
             embeddings=embeddings,
             dropout=args.dropout,
             num_classes=3, device=device).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0004)
start_epoch = 0
max_gradient_norm = 10
best_accuracy = 0
for epoch in range(start_epoch, args.epochs):
    niter = 0
    loss_total = 0

    for batch_index, batch in enumerate(trainloader):
        premises, hypotheses, labels, hypotheses_lengths, premises_lengths = batch
        premises = premises.cuda()
        hypotheses = hypotheses.cuda()
        hypotheses_lengths = hypotheses_lengths.cuda()
        premises_lengths = premises_lengths.cuda()
        labels = labels.cuda()

        logits, probs = model(premises,
                              premises_lengths,
                              hypotheses,
                              hypotheses_lengths)

        optimizer.zero_grad()
        loss = criterion(probs, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_gradient_norm)
        optimizer.step()
        loss_total += loss.data.item()
    print(loss_total / float(len(trainloader)))
    curr_acc = evaluate_model(model, testloader, criterion)
    if curr_acc > best_accuracy:
        print("saving model...")
        with open(args.save_path + "/" + args.model_type + '.pt', 'wb') as f:
            torch.save(model.state_dict(), f)
        best_accuracy = curr_acc

print("Best accuracy :{0}".format(best_accuracy))
