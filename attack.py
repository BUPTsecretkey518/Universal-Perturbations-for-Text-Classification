import argparse
import json
import os
import torch
import torch.nn as nn

from models import Baseline_LSTM, Baseline_Embeddings, ESIM
from attack_utils import x_to_c_to_z, z_to_c_to_x, train_data_require, get_targetmodel_embdding, project_r, \
    evaluate_attack, find_indices_lengths

parser = argparse.ArgumentParser(description='PyTorch baseline for Text')
parser.add_argument('--data_path', type=str, default='./data/classifier',
                    help='location of the data corpus')
parser.add_argument('--model_type', type=str, default="esim",
                    help='location of the data corpus')
parser.add_argument('--epochs', type=int, default=20,
                    help='maximum number of epochs')
parser.add_argument('--batch_size', type=int, default=64, metavar='N',
                    help='batch size')
parser.add_argument('--cuda', action='store_true', default=True,
                    help='use CUDA')
parser.add_argument('--save_path', type=str, default='./models/baseline',
                    help='used for saving the models')
parser.add_argument('--vocab_size', type=int, default=11000,
                    help='vocabulary size')
parser.add_argument('--attack_label', type=int, default=2,
                    help='attack label')
parser.add_argument('--noise_r', type=float, default=0.2,
                    help='variance of r')
parser.add_argument('--pre_model_dir', type=str, default='./output/1593075369',
                    help='pretrained model path')
parser.add_argument('--cur_dir', type=str, default='output_attack_esim',
                    help='output_attack dir')
parser.add_argument('--max_len', type=int, default=10,
                    help='maxlen')
parser.add_argument('--z_dim', type=int, default=100,
                    help='dimension of z')
parser.add_argument('--step_size', type=int, default=3000,
                    help='gradient factor')
parser.add_argument('--log_loss', type=int, default=250,
                    help='result log')
parser.add_argument('--vocab_path', type=str, default='./output/1593075369/vocab.json',
                    help='vocabulary size')
args = parser.parse_args()

word2idx = json.load(open(args.vocab_path, "rb"))
trainloader, testloader = train_data_require(args, word2idx)

criterion = nn.CrossEntropyLoss()
r_threshold = [0.1]
# r_threshold= [random.uniform(0.1,0.13) for _ in range(8)]
# r_threshold=[0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5]
# r_threshold=[0.05,0.1,0.15,0.2,0.25,0.3]
r_iter = 0
while r_iter < len(r_threshold):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if args.model_type == "lstm":
        baseline_model = Baseline_LSTM(100, 300, maxlen=args.max_len, gpu=args.cuda)
        baseline_model.load_state_dict(torch.load(args.save_path + "/" + args.model_type + '.pt'))
    elif args.model_type == "emb":
        baseline_model = Baseline_Embeddings(100, vocab_size=args.vocab_size + 4)
        baseline_model.load_state_dict(torch.load(args.save_path + "/" + args.model_type + '.pt'))
    elif args.model_type == "esim":
        baseline_model = ESIM(args.vocab_size + 4, 300, 300, num_classes=3, device=device).to(device)
        baseline_model.load_state_dict(torch.load(args.save_path + "/" + args.model_type + '.pt'))
    if args.cuda:
        baseline_model = baseline_model.cuda()
    target_emb_parm = get_targetmodel_embdding(baseline_model)

    baseline_model.train()

    r = torch.ones(size=(args.batch_size, args.z_dim))
    r.normal_(mean=0, std=args.noise_r)
    r = r.cuda()
    start_r_data = r.data.clone()
    print('start_r_data:', start_r_data)

    step_bound = r_threshold[r_iter] / 100
    patience = 0
    patience_lim = 3
    update = True
    i_trial = 0
    max_trial = 3
    old_loss = float('-Inf')
    old_r = None
    step_scale = 0.1
    iter_total = 0
    max_iterations = 1000
    flag = False

    path = os.path.join(args.cur_dir, )

    with open('./output/{0}/logs_{1}.txt'.format(args.cur_dir, r_threshold[r_iter]), 'w') as f:
        f.write("Create experiment at " + args.cur_dir)
        f.write("\n")
        f.write("r_threshold is " + str(r_threshold[r_iter]))
        f.write("\n")
    print(" r_threshold is {0}".format(r_threshold[r_iter]))

    for epoch in range(0, args.epochs):
        niter = 0
        loss_total = 0
        log_total = 0

        train_iter = iter(trainloader)
        while niter < len(trainloader):

            niter += 1
            iter_total += 1
            premise, hypothesis, target, hypotheses_lengths, premises_lengths = train_iter.next()
            premise.requires_grad = False
            hypothesis.requires_grad = False

            if args.cuda:
                premise = premise.cuda()
                hypothesis = hypothesis.cuda()
                target = target.cuda()
                premises_lengths = premises_lengths.cuda()

            z = x_to_c_to_z(hypothesis, args, hypotheses_lengths)

            r.requires_grad = True
            z1 = z + r[:z.size()[0]]

            max_indices, out_emb = z_to_c_to_x(z1, args, target_emb_parm, hypothesis, hypotheses_lengths)
            max_indices_lengths = find_indices_lengths(max_indices).cuda()

            _, output = baseline_model.forward_with_trigger(premise, premises_lengths, max_indices, max_indices_lengths,
                                                            out_emb)

            loss = criterion(output, target)
            loss.backward()
            loss_total += loss.item()
            log_total += loss.item()

            r_diff = args.step_size * r.grad.data
            r_diff = project_r(r_diff, r_threshold=step_bound)
            r.data = r.data + r_diff

            whole_diff = r.data - start_r_data
            whole_diff = project_r(whole_diff, r_threshold=r_threshold[r_iter])
            r.data = start_r_data + whole_diff

            if niter % args.log_loss == 0:
                cur_loss = log_total / args.log_loss
                print('current iter:{}'.format(niter))
                print('current loss:{}'.format(cur_loss))
                log_total = 0

                if cur_loss > old_loss:
                    patience = 0
                    old_loss = cur_loss
                    old_r = r.data.clone()
                    update = True
                else:
                    patience += 1

                print('current patience:{}'.format(patience))
                print('\n')

                if patience >= patience_lim:
                    patience = 0
                    args.step_size *= step_scale
                    r.data = old_r
                    print('current step size:{}'.format(args.step_size))
                    i_trial += 1
                    print('current trial:{}'.format(i_trial))
                    print('\n')
            if i_trial >= max_trial or iter_total >= max_iterations:
                best_r, best_acc = evaluate_attack(args, update, old_r, testloader, word2idx, r_threshold, r_iter)

                print('best_acc:', best_acc)
                print('best_r:', best_r)
                flag = True
                break

        if flag:
            r_iter += 1
            break
