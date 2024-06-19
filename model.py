import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from utils import MAX_LENGTH, DEVICE, SUB_SEQ_LEN, HIDDEN_SIZE
from Lang import SOS_token, EOS_token

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
    def __init__(self, input_size, hidden_size, num_bins, dropout_p=0.1):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.y_embedding = nn.Embedding(num_bins, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.dropout = nn.Dropout(dropout_p)

    def forward(self, input, y_s, train=False):
        embedded = self.embedding(input)
        y_embedded = self.embedding(y_s)
        if(train):
            embedded = self.dropout(embedded)
            y_embedded = self.dropout(y_embedded)
        
        y_embedded = torch.unsqueeze(y_embedded, dim = 0)
        
        output, hidden = self.gru(embedded, y_embedded)
        # make sure that hidden and encoder both have num_batch x seq_length x hidden_dims as shapes
        hidden = hidden.squeeze().unsqueeze(dim = 1)
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
    def __init__(self, input_size, num_bins, hidden_size = 100, output_size = None):
        super(Variator, self).__init__()
        if(output_size == None):
            output_size = input_size
        self.encode_mu = nn.Linear(input_size, hidden_size)
        self.encode_log_var = nn.Linear(input_size, hidden_size)
        self.decode_layer = nn.Linear(hidden_size + 1, output_size)
        self.embed_condition = nn.Embedding(num_bins, 1)
    

    def encode(self, x):
        log_var = self.encode_log_var(x)
        x = self.encode_mu(x)
        return x, log_var

    def decode(self, x, y_s):
        y_s = self.embed_condition(y_s)
        y_s = y_s.unsqueeze(1).expand(-1, x.shape[1], 1)
        x = torch.cat((x, y_s), dim=2)
        x = self.decode_layer(x)

        return x

    def reparameterize(self, x, log_var):
        # Get standard deviation
        std = torch.exp(log_var)
        # Returns random numbers from a normal distribution
        eps = torch.randn_like(std)
        # Return sampled values
        return eps.mul(std).add_(x)

    def forward(self, x, y_s, isTraining = False):
        mu, log_var = self.encode(x)

        if(isTraining):
            x = self.reparameterize(mu, log_var)
            x = self.decode(x, y_s)
        else:
            x = self.decode(mu, y_s)
        
        return x, mu, log_var