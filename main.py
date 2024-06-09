import random
import time
import math

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

import numpy as np
from torch.utils.data import TensorDataset, DataLoader, RandomSampler

from Lang import EOS_token
from utils import filterPairs, readLangs, read_single_lang, DEVICE, MAX_LENGTH, SUB_SEQ_LEN, HIDDEN_SIZE, timeSince, filterWords

from model import EncoderRNN, AttnDecoderRNN, Variator

device = DEVICE

def indexesFromSentence(lang, sentence):
    return [lang.char2index[char] for char in sentence]

def tensorFromSentence(lang, sentence):
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_token)
    return torch.tensor(indexes, dtype=torch.long, device=device).view(1, -1)

def tensorsFromPair(pair):
    input_tensor = tensorFromSentence(input_lang, pair[0])
    target_tensor = tensorFromSentence(output_lang, pair[1])
    return (input_tensor, target_tensor)

def get_dataloader(batch_size):
    input_lang, output_lang, pairs = prepare_single_data('chem', True)
    n = len(pairs)
    input_ids = np.zeros((n, MAX_LENGTH), dtype=np.int32)
    target_ids = np.zeros((n, MAX_LENGTH), dtype=np.int32)

    for idx, (inp, tgt) in enumerate(pairs):
        inp_ids = indexesFromSentence(input_lang, inp)
        tgt_ids = indexesFromSentence(output_lang, tgt)
        inp_ids.append(EOS_token)
        tgt_ids.append(EOS_token)
        input_ids[idx, :len(inp_ids)] = inp_ids
        target_ids[idx, :len(tgt_ids)] = tgt_ids

    train_data = TensorDataset(torch.LongTensor(input_ids).to(device),
                               torch.LongTensor(target_ids).to(device))

    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)
    return input_lang, output_lang, train_dataloader

def prepareData(lang1, lang2, reverse=False):
    input_lang, output_lang, pairs = readLangs(lang1, lang2, reverse)
    print("Read %s sentence pairs" % len(pairs))
    pairs = filterPairs(pairs)
    print("Trimmed to %s sentence pairs" % len(pairs))
    print("Counting chars...")
    for pair in pairs:
        input_lang.addSentence(pair[0])
        output_lang.addSentence(pair[1])
    print("Counted chars:")
    print(input_lang.name, input_lang.n_chars)
    print(output_lang.name, output_lang.n_chars)
    return input_lang, output_lang, pairs


def prepare_single_data(lang1, reverse=False):
    input_lang, output_lang, pairs = read_single_lang(lang1, reverse)
    print("Read %s sentence pairs" % len(pairs))
    pairs = filterWords(pairs)
    print("Trimmed to %s sentence pairs" % len(pairs))
    print("Counting chars...")
    for pair in pairs:
        input_lang.addSentence(pair[0])
        output_lang.addSentence(pair[1])
    print("Counted chars:")
    print(input_lang.name, input_lang.n_chars)
    print(output_lang.name, output_lang.n_chars)
    return input_lang, output_lang, pairs

def kl_loss(mu=None, logstd=None):
    """
    Closed formula of the KL divergence for normal distributions
    """
    MAX_LOGSTD = 10
    logstd = logstd.clamp(max=MAX_LOGSTD)
    mu = mu.to(device)
    logstd = logstd.to(device)
    kl_div = -0.5 * torch.mean(torch.sum(1 + 2 * logstd - mu**2 - torch.exp(2 * logstd), dim=1))

    # Limit numeric errors
    kl_div = kl_div.clamp(max=1000)
    return kl_div    

def loss_fn(outputs, target, criterion, mus, log_vars, kl_beta = 0.05):
    recon_loss = criterion(
        outputs.view(-1, outputs.size(-1)),
        target.view(-1)
    )
    
    kl_losses = 0

    for i in range(len(mus)):
        kl_losses += kl_loss(mus[i], log_vars[i])

    return recon_loss + kl_beta/len(mus) * kl_losses, recon_loss, kl_losses

def train_epoch(dataloader, encoder, variator, hidden_variator, decoder, encoder_optimizer,
          decoder_optimizer, variator_optimizer, hidden_variator_optimizer, criterion, percent_done):

    total_loss = 0
    for data in (dataloader):
        input_tensor, target_tensor = data

        input_tensor.to(device)
        target_tensor.to(device)

        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()
        variator_optimizer.zero_grad()
        hidden_variator_optimizer.zero_grad()

        encoder_outputs, encoder_hidden = encoder(input_tensor)
        variator_outputs, mu, log_var = variator(encoder_outputs, isTraining = True)
        hidden_variator_outputs, hid_mu, hid_logvar = hidden_variator(encoder_hidden)
        decoder_outputs, _, _ = decoder(variator_outputs, hidden_variator_outputs, target_tensor)

        means = [mu, hid_mu]
        log_vars = [log_var, hid_logvar]

        kl_beta = percent_done * 0.1

        loss, recon_loss, kl_loss = loss_fn(decoder_outputs, target_tensor, criterion, means, log_vars)

        loss.backward()

        encoder_optimizer.step()
        decoder_optimizer.step()
        variator_optimizer.step()
        hidden_variator_optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader), recon_loss / len(dataloader), kl_loss / len(dataloader)


def train(train_dataloader, encoder, variator, hidden_variator, decoder, n_epochs, learning_rate=0.001,
               print_every=100, plot_every=100):
    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    print_total_recon_loss = 0
    print_total_kl_loss = 0
    plot_loss_total = 0  # Reset every plot_every

    encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)
    variator_optimizer = optim.Adam(variator.parameters(), lr = learning_rate)
    hidden_variator_optimizer = optim.Adam(hidden_variator.parameters(), lr = learning_rate)
    criterion = nn.NLLLoss()

    for epoch in range(1, n_epochs + 1):
        loss, print_total_recon_loss, print_total_kl_loss = train_epoch(train_dataloader, encoder, variator, hidden_variator, decoder, encoder_optimizer, decoder_optimizer, variator_optimizer, hidden_variator_optimizer, criterion, (epoch - 1)/num_epochs)
        print_loss_total += loss
        plot_loss_total += loss

        if epoch % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (timeSince(start, epoch / n_epochs),
                                        epoch, epoch / n_epochs * 100, print_loss_avg))

        if epoch % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0


def evaluate(encoder, variator, hidden_variator, decoder, sentence, input_lang, output_lang):
    with torch.no_grad():
        input_tensor = tensorFromSentence(input_lang, sentence)

        encoder_outputs, encoder_hidden = encoder(input_tensor)
        variator_outputs,_,_ = variator(encoder_outputs)
        hidden_vaiator_outputs,_,_ = hidden_variator(encoder_hidden)
        decoder_outputs, decoder_hidden, decoder_attn = decoder(variator_outputs, hidden_vaiator_outputs)

        _, topi = decoder_outputs.topk(1)
        decoded_ids = topi.squeeze()

        decoded_chars = []
        for idx in decoded_ids:
            if idx.item() == EOS_token:
                decoded_chars.append('<EOS>')
                break
            decoded_chars.append(output_lang.index2char[idx.item()])
    return decoded_chars, decoder_attn


def evaluateRandomly(encoder, variator, hidden_variator, decoder, n=10):
    for i in range(n):
        pair = random.choice(pairs)
        print('>', pair[0])
        print('=', pair[1])
        output_chars, _ = evaluate(encoder, variator, hidden_variator, decoder, pair[0], input_lang, output_lang)
        output_sentence = ''.join(output_chars)
        print('<', output_sentence)
        print('')



##MAIN
num_sub_seqs = math.ceil(MAX_LENGTH / SUB_SEQ_LEN)
input_lang, output_lang, pairs = prepare_single_data('chem', True)
print(random.choice(pairs))


hidden_size = HIDDEN_SIZE
batch_size = 32
num_epochs = 200
input_lang, output_lang, train_dataloader = get_dataloader(batch_size)

encoder = EncoderRNN(input_lang.n_chars, hidden_size).to(device)
variator = Variator(hidden_size)
hidden_variator = Variator(hidden_size, output_size=num_sub_seqs * hidden_size)
decoder = AttnDecoderRNN(hidden_size, output_lang.n_chars).to(device)

encoder.to(device)
variator.to(device)
hidden_variator.to(device)
decoder.to(device)
print("begin train")
train(train_dataloader, encoder, variator, hidden_variator, decoder, num_epochs, print_every=5, plot_every=5)

full_model = (encoder, variator, hidden_variator, decoder)

torch.save(full_model, 'model.pt')
encoder.eval()
decoder.eval()
evaluateRandomly(encoder, variator, hidden_variator, decoder, n = 100)