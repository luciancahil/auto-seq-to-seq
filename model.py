import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from utils import MAX_LENGTH, DEVICE, SUB_SEQ_LEN, HIDDEN_SIZE
from Lang import SOS_token, EOS_token
import math

device = DEVICE

class DecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size):
        super(DecoderRNN, self).__init__()
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, encoder_outputs, encoder_hidden, target_tensor=None):
        batch_size = encoder_outputs.size(0)
        decoder_input = torch.empty(batch_size, 1, dtype=torch.long, device=device).fill_(SOS_token)
        decoder_hidden = encoder_hidden
        decoder_outputs = []

        for i in range(MAX_LENGTH):
            decoder_output, decoder_hidden  = self.forward_step(decoder_input, decoder_hidden)
            decoder_outputs.append(decoder_output)

            if target_tensor is not None:
                # Teacher forcing: Feed the target as the next input
                decoder_input = target_tensor[:, i].unsqueeze(1) # Teacher forcing
            else:
                # Without teacher forcing: use its own predictions as the next input
                _, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze(-1).detach()  # detach from history as input

        decoder_outputs = torch.cat(decoder_outputs, dim=1)
        decoder_outputs = F.log_softmax(decoder_outputs, dim=-1)
        return decoder_outputs, decoder_hidden, None # We return `None` for consistency in the training loop

    def forward_step(self, input, hidden):
        output = self.embedding(input)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = self.out(output)
        return output, hidden

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, dropout_p=0.1):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, input, train=False):
        embedded = self.embedding(input)
        hidden = torch.zeros(1, embedded.shape[0], self.hidden_size)
        hidden = hidden.to(device)
        if(train):
            embedded = self.dropout(embedded)
        
        output, hidden = self.gru(embedded, hidden)
        # make sure that hidden and encoder both have num_batch x seq_length x hidden_dims as shapes
        hidden = hidden.squeeze(dim=0).unsqueeze(dim = 1)
        return output, hidden
    
class BahdanauAttention(nn.Module):
    def __init__(self, hidden_size):
        super(BahdanauAttention, self).__init__()
        self.Wa = nn.Linear(hidden_size, hidden_size)
        self.Ua = nn.Linear(hidden_size, hidden_size)
        self.Va = nn.Linear(hidden_size, 1)

    def forward(self, query, keys):
        scores = self.Va(torch.tanh(self.Wa(query) + self.Ua(keys)))
        scores = scores.squeeze(2).unsqueeze(1)

        weights = F.softmax(scores, dim=-1)
        context = torch.bmm(weights, keys)

        return context, weights

class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1):
        super(AttnDecoderRNN, self).__init__()
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.attention = BahdanauAttention(hidden_size)
        self.gru = nn.GRU(2 * hidden_size, hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, encoder_outputs, encoder_hidden, is_training=False, target_tensor=None):
        batch_size = encoder_outputs.size(0)
        decoder_input = torch.empty(batch_size, 1, dtype=torch.long, device=device).fill_(SOS_token)
        decoder_hidden = encoder_hidden
        decoder_outputs = []
        attentions = []
        sub_seq_num = -1 # add to 0 to start

        for i in range(MAX_LENGTH):
            if i % SUB_SEQ_LEN == 0:
                # we need a new subsequence hidden vector.
                sub_seq_num += 1
                decoder_hidden = encoder_hidden[:, :, sub_seq_num * HIDDEN_SIZE: (sub_seq_num + 1) * HIDDEN_SIZE]

            decoder_hidden = decoder_hidden.contiguous()
            decoder_output, decoder_hidden, attn_weights = self.forward_step(
                decoder_input, decoder_hidden, encoder_outputs, isTraining = is_training
            )
            decoder_outputs.append(decoder_output)
            attentions.append(attn_weights)

            if target_tensor is not None:
                # Teacher forcing: Feed the target as the next input
                decoder_input = target_tensor[:, i].unsqueeze(1) # Teacher forcing
            else:
                # Without teacher forcing: use its own predictions as the next input
                _, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze(-1).detach()  # detach from history as input

        decoder_outputs = torch.cat(decoder_outputs, dim=1)
        decoder_outputs = F.log_softmax(decoder_outputs, dim=-1)
        attentions = torch.cat(attentions, dim=1)

        return decoder_outputs, decoder_hidden, attentions


    def forward_step(self, input, hidden, encoder_outputs, isTraining = False):
        embedded =  self.embedding(input)

        if(isTraining):
            embedded = self.dropout(embedded)

        query = hidden.permute(1, 0, 2)
        context, attn_weights = self.attention(query, encoder_outputs)
        input_gru = torch.cat((embedded, context), dim=2)

        output, hidden = self.gru(input_gru, hidden)
        output = self.out(output)

        return output, hidden, attn_weights
    

class Variator(nn.Module):
    def __init__(self, input_size, latent_size = 100, output_size = None):
        super(Variator, self).__init__()
        if(output_size == None):
            output_size = input_size
        self.encode_mu = nn.Linear(input_size, latent_size)
        self.encode_log_var = nn.Linear(input_size, latent_size)
        self.decode_layer = nn.Linear(latent_size, output_size)    

    def encode(self, x):
        log_var = self.encode_log_var(x)
        x = self.encode_mu(x)
        return x, log_var

    def decode(self, x):
        x = self.decode_layer(x)

        return x

    def reparameterize(self, x, log_var):
        # Get standard deviation
        std = torch.exp(log_var)
        # Returns random numbers from a normal distribution
        eps = torch.randn_like(std)
        # Return sampled values
        return eps.mul(std).add_(x)

    def forward(self, x, isTraining = False):
        mu, log_var = self.encode(x)

        if(isTraining):
            x = self.reparameterize(mu, log_var)
            x = self.decode(x)
        else:
            x = self.decode(mu)
        
        return x, mu, log_var
    
class LinearRegression(nn.Module):
    def __init__(self, width, height, kernel_size = 3, hidden_channels = 16, hidden_size = 10, output_dim = 1):
        super(LinearRegression, self).__init__()
        self.conv_one = nn.Conv2d(1, hidden_channels, kernel_size)
        self.conv_two = nn.Conv2d(hidden_channels, 1, kernel_size)
        self.pool = nn.AvgPool2d(2, stride=2, padding=1)
        self.activation = nn.LeakyReLU()
        new_width = math.ceil((math.ceil((width + 1 - kernel_size + 2) / 2) + 1 + 2 - kernel_size)/2)
        new_height = math.ceil((math.ceil((height + 1 - kernel_size + 2) / 2) + 1 + 2 - kernel_size)/2)
        new_num_pixels = new_width * new_height
        self.hidden = nn.Linear(new_num_pixels, hidden_size)
        self.mem_hidden = nn.Linear(width, hidden_size)
        self.output = nn.Linear(2 * hidden_size, output_dim)
        self.dropout = nn.Dropout(0.3)
    

    def forward(self, sequence, memory, isTraining = False):
        sequence = sequence.unsqueeze(dim = 1)

        sequence = self.conv_one(sequence)
        sequence = self.activation(sequence)
        sequence = self.pool(sequence)

        sequence = self.conv_two(sequence)
        sequence = self.activation(sequence)
        sequence = self.pool(sequence)


        sequence = torch.flatten(sequence, start_dim=1, end_dim=3)
        sequence = self.hidden(sequence)
        memory = torch.squeeze(memory, dim = 1)
        memory = self.mem_hidden(memory)
        sequence = torch.cat((sequence, memory), dim = 1)
        

        if(isTraining):
            sequence = self.dropout(sequence)
        
        sequence = self.output(sequence)
        return sequence





