import os
import torch
import numpy as np
import torch.nn.functional as F

from models import ESIM
from utils import SNLIDataset, collate_snli, one_hot_prob


def train_data_require(args, word2idx):
    corpus_train = SNLIDataset(train=True, vocab_size=args.vocab_size, path=args.data_path,
                               attack_label=args.attack_label, reset_vocab=word2idx)
    corpus_test = SNLIDataset(train=False, vocab_size=args.vocab_size, path=args.data_path,
                              attack_label=args.attack_label, reset_vocab=word2idx)
    trainloader = torch.utils.data.DataLoader(corpus_train, batch_size=args.batch_size, collate_fn=collate_snli,
                                              shuffle=True)
    testloader = torch.utils.data.DataLoader(corpus_test, batch_size=args.batch_size, collate_fn=collate_snli,
                                             shuffle=False)
    return trainloader, testloader


def x_to_c_to_z(x, args, lengths):
    autoencoder = torch.load(args.pre_model_dir + '/models/autoencoder_model.pt')
    inverter = torch.load(args.pre_model_dir + '/models/inverter_model.pt')
    autoencoder = autoencoder.cuda()
    inverter = inverter.cuda()
    real_hidden = autoencoder(x, lengths, noise=True, encode_only=True)
    z = inverter(real_hidden)
    return z


def z_to_c_to_x(z1, args, target_emb_parm, hypothesis, lengths):
    autoencoder = torch.load(args.pre_model_dir + '/models/autoencoder_model.pt')
    gan_gen = torch.load(args.pre_model_dir + '/models/gan_gen_model.pt')
    autoencoder = autoencoder.cuda()
    gan_gen = gan_gen.cuda()
    c1 = gan_gen(z1)
    max_len = hypothesis.size()[1]
    batchsize = hypothesis.size()[0]
    max_indices, decoded = autoencoder.generate_decoding(c1, max_len, sample=False, temp=1.0)
    decoded_prob = F.softmax(decoded, dim=-1)
    decoded_prob = one_hot_prob(decoded_prob, max_indices)
    out_emb = torch.matmul(decoded_prob, target_emb_parm)
    return max_indices, out_emb


def save_models(args, autoencoder, gan_gen, inverter, r_threshold):
    if not os.path.isdir('./output/{0}/{1}'.format(args.cur_dir, r_threshold)):
        os.makedirs('./output/{0}/{1}'.format(args.cur_dir, r_threshold))
    with open('./output/{0}/{1}/autoencoder_model.pt'.format(args.cur_dir, r_threshold), 'wb') as f:
        torch.save(autoencoder, f)
    with open('./output/{0}/{1}/inverter_model.pt'.format(args.cur_dir, r_threshold), 'wb') as f:
        torch.save(inverter, f)
    with open('./output/{0}/{1}/gan_gen_model.pt'.format(args.cur_dir, r_threshold), 'wb') as f:
        torch.save(gan_gen, f)


def get_targetmodel_embdding(target_model):
    params = list(target_model.named_parameters())
    embedding_name, embedding_weight = params[0]
    return embedding_weight


def project_r(noise, r_threshold=2):
    # project the noise into the ball with radius r_threshold
    # since radius is correlated with dimension of noise, we treat the r_threshold as
    # effective radius for each dimension.
    # noise is a tensor of dimension
    noise_dim = noise.size(1)
    r_threshold_alldim = r_threshold ** 2 * noise_dim
    with torch.no_grad():
        noise_radius = torch.sum(noise ** 2, dim=1)
        mask_proj = noise_radius > r_threshold_alldim
        noise_radius = torch.sqrt(noise_radius)
        noise[mask_proj, :] = noise[mask_proj, :] / torch.unsqueeze(noise_radius[mask_proj], dim=1) * np.sqrt(
            r_threshold_alldim)
    return noise


def write_file(args, hypothesis, ae_indices, x1, predictions, r_threshold, idx2word, premise):
    hypothesis = hypothesis.cpu().numpy()
    ae_indices = ae_indices.cpu().numpy()
    x1 = x1.cpu().numpy()
    premise = premise.cpu().numpy()
    with open('./output/{0}/logs_{1}.txt'.format(args.cur_dir, r_threshold), 'a') as f:
        for t, ae, idx, pre, prem in zip(hypothesis, ae_indices, x1, predictions, premise):
            # real sentence
            f.write("# # # original sentence # # #\n")
            chars = " ".join([idx2word[x] for x in t])
            f.write('' + chars)
            # ae output sentence
            f.write("\n# # # sentence -> encoder -> decoder # # #\n")
            chars = " ".join([idx2word[x] for x in ae])
            f.write(chars)
            # autoencoder output sentence
            f.write("\n# # # sentence -> encoder -> inverter -> generator "
                    "-> decoder # # #\n")
            chars = " ".join([idx2word[x] for x in idx])
            f.write(chars)
            f.write("\n prediction \n")
            f.write(str(pre))
            f.write("\n# # # premise  # # #\n")
            chars = " ".join([idx2word[x] for x in prem])
            f.write(chars)
            f.write("\n\n")


def evaluate_r(args, r_new, testloader, write, idx2word, r_threshold, autoencoder, gan_gen, inverter, baseline_model):
    test_iter = iter(testloader)
    niter = 0
    correct = 0
    total = 0
    while niter < len(testloader):
        niter += 1
        premise, hypothesis, target, hypotheses_lengths, premises_lengths = test_iter.next()
        if args.cuda:
            premise = premise.cuda()
            hypothesis = hypothesis.cuda()
            target = target.cuda()
            hypotheses_lengths = hypotheses_lengths.cuda()
            premises_lengths = premises_lengths.cuda()
        with torch.no_grad():
            max_len = hypothesis.size()[1]
            hidden = autoencoder.encode(hypothesis, hypotheses_lengths, noise=True)
            ae_indices = autoencoder.generate(hidden, max_len, sample=False)
            ae_lengths = find_indices_lengths(ae_indices).cuda()
            real_hidden = autoencoder(hypothesis, hypotheses_lengths, noise=True, encode_only=True)
            z = inverter(real_hidden)
            z1 = z + r_new
            c1 = gan_gen(z1)
            max_len = hypothesis.size()[1]
            batchsize = hypothesis.size()[0]
            x1 = autoencoder.generate(c1, max_len, sample=False, temp=1.0)
            x1_lengths = find_indices_lengths(x1).cuda()
            _, prob_distrib = baseline_model(premise, premises_lengths, x1, x1_lengths)
            predictions = np.argmax(prob_distrib.data.cpu().numpy(), 1)
            correct += len(np.where(target.data.cpu().numpy() == predictions)[0])
            total += premise.size(0)
        if write:
            write_file(args, hypothesis, ae_indices, x1, predictions, r_threshold, idx2word, premise)
    acc = correct / float(total)
    return acc, hypothesis, ae_indices, x1, predictions


def evaluate_attack(args, update, old_r, testloader, word2idx, r_threshold, r_iter):
    autoencoder = torch.load(args.pre_model_dir + '/models/autoencoder_model.pt').cuda()
    gan_gen = torch.load(args.pre_model_dir + '/models/gan_gen_model.pt').cuda()
    inverter = torch.load(args.pre_model_dir + '/models/inverter_model.pt').cuda()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    baseline_model = ESIM(args.vocab_size + 4, 300, 300, num_classes=3, device=device).to(device)
    baseline_model.load_state_dict(torch.load(args.save_path + "/" + args.model_type + '.pt'))
    autoencoder.eval()
    gan_gen.eval()
    inverter.eval()
    baseline_model.eval()
    idx2word = {v: k for k, v in word2idx.items()}
    if update:
        best_acc = 1
        best_r = torch.ones(args.z_dim)
        for noise_iter in range(old_r.size()[0]):
            r_new = torch.ones(args.z_dim, requires_grad=False).cuda()
            r_new.data = old_r[noise_iter]
            acc, _, _, _, _ = evaluate_r(args, r_new, testloader, False, idx2word, r_threshold[r_iter], autoencoder,
                                         gan_gen, inverter, baseline_model)
            print("Accuracy:{0}".format(acc))
            if (acc < best_acc):
                best_acc = acc
                best_r = r_new
        with open('./output/{0}/logs_{1}.txt'.format(args.cur_dir, r_threshold[r_iter]), 'a') as f:
            f.write("best_acc: " + str(best_acc))
            f.write("\n")
            f.write("best_r: " + str(best_r))
            f.write("\n")
        acc, _, _, _, _ = evaluate_r(args, best_r, testloader, True, idx2word, r_threshold[r_iter], autoencoder,
                                     gan_gen, inverter, baseline_model)
    return best_r, acc


def find_indices_lengths(max_indices):
    indices_lengths = []
    for i in range(max_indices.size()[0]):
        j = 0
        sen_len = max_indices.size()[1]
        while j < sen_len:
            if max_indices[i][j] == 2:
                length = j
                indices_lengths.append(length + 1)
                break
            j += 1
            if j == sen_len:
                indices_lengths.append(sen_len)
    return torch.LongTensor(indices_lengths)
