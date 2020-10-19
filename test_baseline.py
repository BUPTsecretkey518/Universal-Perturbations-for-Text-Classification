import argparse
import json
import random
import numpy as np
import torch.utils.data

from models import Baseline_Embeddings, Baseline_LSTM, ESIM
from utils import SNLIDataset, collate_snli


def evaluate_model():
    test_iter = iter(testloader)
    correct = 0
    total = 0
    for batch in test_iter:
        premise, hypothesis, target, hypotheses_lengths, premises_lengths = batch
        if args.cuda:
            premise = premise.cuda()
            hypothesis = hypothesis.cuda()
            target = target.cuda()
            hypotheses_lengths = hypotheses_lengths.cuda()
            premises_lengths = premises_lengths.cuda()
        _, prob_distrib = baseline_model.forward(premise, premises_lengths, hypothesis, hypotheses_lengths)
        predictions = np.argmax(prob_distrib.data.cpu().numpy(), 1)
        correct += len(np.where(target.data.cpu().numpy() == predictions)[0])
        total += premise.size(0)
        print(correct)
        print(total)
    acc = correct / float(total)
    print("Accuracy:{0}".format(acc))
    return acc


parser = argparse.ArgumentParser(description='PyTorch baseline for Text')
parser.add_argument('--data_path', type=str, default='./data/classifier',
                    help='location of the data corpus')
parser.add_argument('--model_type', type=str, default="esim",
                    help='location of the data corpus')
parser.add_argument('--epochs', type=int, default=20,
                    help='maximum number of epochs')
parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                    help='batch size')
parser.add_argument('--train_mode', type=bool, default=True,
                    help='set training mode')
parser.add_argument('--maxlen', type=int, default=10,
                    help='maximum sentence length')
parser.add_argument('--seed', type=int, default=1111,
                    help='seed')
parser.add_argument('--beta1', type=float, default=0.9,
                    help='beta1 for adam. default=0.9')
parser.add_argument('--cuda', action='store_true', default=True,
                    help='use CUDA')
parser.add_argument('--save_path', type=str, default='./models/baseline',
                    help='used for saving the models')
parser.add_argument('--vocab_size', type=int, default=11000,
                    help='vocabulary size')
parser.add_argument('--attack_label', type=int, default=0,
                    help='attack_label')
parser.add_argument('--vocab_path', type=str, default='./output/1593075369/vocab.json',
                    help='vocabulary size')
args = parser.parse_args()

word2idx = json.load(open(args.vocab_path, "rb"))

corpus_test = SNLIDataset(train=False, vocab_size=args.vocab_size, path=args.data_path, attack_label=args.attack_label,
                          reset_vocab=word2idx)
print(len(corpus_test.test_data))

testloader = torch.utils.data.DataLoader(corpus_test, batch_size=args.batch_size, collate_fn=collate_snli,
                                         shuffle=False)
random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
if args.model_type == "lstm":
    baseline_model = Baseline_LSTM(100, 300, maxlen=args.maxlen, gpu=args.cuda)
    baseline_model.load_state_dict(torch.load(args.save_path + "/" + args.model_type + '.pt'))
elif args.model_type == "emb":
    baseline_model = Baseline_Embeddings(100, vocab_size=args.vocab_size + 4)
    baseline_model.load_state_dict(torch.load(args.save_path + "/" + args.model_type + '.pt'))
elif args.model_type == "esim":
    baseline_model = ESIM(args.vocab_size + 4, 300, 300, num_classes=3, device=device).to(device)
    baseline_model.load_state_dict(torch.load(args.save_path + "/" + args.model_type + '.pt'))
if args.cuda:
    baseline_model = baseline_model.cuda()

baseline_model.eval()
evaluate_model()
