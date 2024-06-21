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
from utils import DEVICE, MAX_LENGTH, SUB_SEQ_LEN, HIDDEN_SIZE, timeSince, tensorFromSentence, get_dataloader

from model import EncoderRNN, AttnDecoderRNN, Variator

device = DEVICE



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
        input_tensor, target_tensor, y_s = data

        input_tensor.to(device)
        target_tensor.to(device)
        y_s = y_s.to(device)

        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()
        variator_optimizer.zero_grad()
        hidden_variator_optimizer.zero_grad()

        encoder_outputs, encoder_hidden = encoder(input_tensor, y_s, train=True)
        variator_outputs, mu, log_var = variator(encoder_outputs, y_s, isTraining = True)
        hidden_variator_outputs, hid_mu, hid_logvar = hidden_variator(encoder_hidden, y_s, isTraining = True)
        # go from batchsize x 1 x hiddenshape to 1 x batchsize x hidden_shape
        hidden_variator_outputs = hidden_variator_outputs.squeeze(dim=1).unsqueeze(dim = 0)
        decoder_outputs, _, _ = decoder(variator_outputs, hidden_variator_outputs, is_training = True, target_tensor=target_tensor)

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


def train(train_dataloader, encoder, variator, hidden_variator, decoder, n_epochs, y_s, learning_rate=0.001,
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
    scheduler = optim.lr_scheduler.CosineAnnealingLR(encoder_optimizer, T_max=num_epochs)
    scheduler_decode = optim.lr_scheduler.CosineAnnealingLR(decoder_optimizer, T_max=num_epochs)
    scheduler_variator = optim.lr_scheduler.CosineAnnealingLR(variator_optimizer, T_max=num_epochs)
    scheduler_hidden = optim.lr_scheduler.CosineAnnealingLR(hidden_variator_optimizer, T_max=num_epochs)

    schedulers = [scheduler, scheduler_decode, scheduler_hidden, scheduler_variator]

    criterion = nn.NLLLoss()

    for epoch in range(1, n_epochs + 1):
        loss, print_total_recon_loss, print_total_kl_loss = train_epoch(train_dataloader, encoder, variator, hidden_variator, decoder, encoder_optimizer, decoder_optimizer, variator_optimizer, hidden_variator_optimizer, criterion, (epoch - 1)/num_epochs)
        print_loss_total += loss
        plot_loss_total += loss

        for s in schedulers:
            s.step()

        if epoch % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            print('%s (%d %d%%) %.4f' % (timeSince(start, epoch / n_epochs),
                                        epoch, epoch / n_epochs * 100, print_loss_avg), flush=True)
                # Log the learning rates for all optimizers
            current_lr_encoder = encoder_optimizer.param_groups[0]['lr']
            current_lr_decoder = decoder_optimizer.param_groups[0]['lr']
            current_lr_variator = variator_optimizer.param_groups[0]['lr']
            current_lr_hidden_variator = hidden_variator_optimizer.param_groups[0]['lr']
            
            print(f'Epoch {epoch+1}/{num_epochs}, Encoder LR: {current_lr_encoder}, Decoder LR: {current_lr_decoder}, Variator LR: {current_lr_variator}, Hidden Variator LR: {current_lr_hidden_variator}')

        if epoch % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0


def evaluate(encoder, variator, hidden_variator, decoder, sentence, input_lang, output_lang, y):
    with torch.no_grad():
        input_tensor = tensorFromSentence(input_lang, sentence)
        input_tensor = input_tensor.to(device)
        y = y.to(device)

        encoder_outputs, encoder_hidden = encoder(input_tensor, y)
        variator_outputs,_,_ = variator(encoder_outputs, y)
        hidden_variator_outputs,_,_ = hidden_variator(encoder_hidden, y)
        hidden_variator_outputs = hidden_variator_outputs.squeeze(dim=1).unsqueeze(dim = 0)
        decoder_outputs, decoder_hidden, decoder_attn = decoder(variator_outputs, hidden_variator_outputs)
        _, topi = decoder_outputs.topk(1)
        decoded_ids = topi.squeeze()

        decoded_chars = []
        for idx in decoded_ids:
            if idx.item() == EOS_token:
                decoded_chars.append('<EOS>')
                break
            decoded_chars.append(output_lang.index2char[idx.item()])
    return decoded_chars, decoder_attn


def evaluateRandomly(encoder, variator, hidden_variator, decoder, y_s, n=10):
    for i in range(n):
        index = random.randint(0, len(pairs))
        pair = pairs[index]
        y = y_s[index].unsqueeze(dim = 0)
        print('>', pair[0])
        print('=', pair[1])
        output_chars, _ = evaluate(encoder, variator, hidden_variator, decoder, pair[0], input_lang, output_lang, y)
        output_sentence = ''.join(output_chars)
        print('<', output_sentence)
        print('')



##MAIN
file_name = 'chem'
print("Device: " + str(device), flush=True)


num_sub_seqs = math.ceil(MAX_LENGTH / SUB_SEQ_LEN)

hidden_size = HIDDEN_SIZE
batch_size = 32
num_epochs = 75
input_lang, output_lang, train_dataloader, quantiles, pairs, y_s = get_dataloader(file_name, batch_size)
num_bins = len(quantiles) + 1
print(random.choice(pairs))
encoder = EncoderRNN(input_lang.n_chars, hidden_size, num_bins).to(device)
variator = Variator(hidden_size, num_bins)
hidden_variator = Variator(hidden_size, num_bins, output_size=num_sub_seqs * hidden_size)
decoder = AttnDecoderRNN(hidden_size, output_lang.n_chars).to(device)

encoder.to(device)
variator.to(device)
hidden_variator.to(device)
decoder.to(device)
print("begin train", flush=True)
train(train_dataloader, encoder, variator, hidden_variator, decoder, num_epochs, y_s, print_every=5, plot_every=5)

full_model = (encoder, variator, hidden_variator, decoder, input_lang, quantiles)

torch.save(full_model, 'model.pt')
encoder.eval()
decoder.eval()
evaluateRandomly(encoder, variator, hidden_variator, decoder, y_s, n = 100)