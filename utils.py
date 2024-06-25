import unicodedata
import re
from Lang import Lang
import torch
import numpy as np
import time
import math
from Lang import EOS_token
import random
from torch.utils.data import TensorDataset, DataLoader, RandomSampler
import sys
# 
SUB_SEQ_LEN = 15
HIDDEN_SIZE = 128
MAX_LENGTH = 40
MAX_NUM_SAMPLES = 100000
NUM_BINS = -1

def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))


"""def showPlot(points):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)"""

# Turn a Unicode string to plain ASCII, thanks to
# https://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

# Lowercase, trim, and remove non-letter characters
def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z!?]+", r" ", s)
    return s.strip()


def readLangs(lang1, lang2, reverse=False):
    print("Reading lines...")

    # Read the file and split into lines
    lines = open('data/%s-%s.txt' % (lang1, lang2), encoding='utf-8').\
        read().strip().split('\n')

    # Split every line into pairs and normalize
    pairs = [[normalizeString(s) for s in l.split('\t')] for l in lines]

    # Reverse pairs, make Lang instances
    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        input_lang = Lang(lang2)
        output_lang = Lang(lang1)
    else:
        input_lang = Lang(lang1)
        output_lang = Lang(lang2)

    return input_lang, output_lang, pairs

def normalizeYs(y_s):

    if(NUM_BINS == -1):
        # calculate number of bins, such that each bin has 1000 samples. 
        # at least 2, at most 20.
        num_bins = min(20, max(2, int(MAX_NUM_SAMPLES / 1000)))
    else:
        num_bins = NUM_BINS
    
    num_bins = 100
    quantiles = np.quantile(y_s, np.linspace(0, 1, num_bins + 1))
    start = quantiles[1]
    end = quantiles [-1]
    endpoints = (start, end)
    y_s = [(y - start)/(end - start) for y in y_s]
    # drop first and last elements, which will be the smallest and largest values
    quantiles = quantiles[1:-1]
    print("Quantile Cutoffs: " + str(quantiles))

    

    return y_s, endpoints


    


def read_single_lang(lang, reverse=False, prevLang = None):
    print("Reading lines...")

    # Read the file and split into lines
    lines = open('data/%s.txt' % (lang), encoding='utf-8').\
        read().strip().split('\n')
    
    new_lines = []
    y_s = []

    if ',' in lines[0]:
        for l in lines:
            parts = l.split(',')
            new_lines.append(parts[0])
            y_s.append(float(parts[1]))
        lines = new_lines
        pairs = [[l, l, y_s[i]] for i, l in enumerate(lines)]
    else:
        pairs = [[l, l, 0] for l in lines]

    
    input_lang = Lang(lang)
    output_lang = Lang(lang)
    print("Read %s sentence pairs" % len(pairs))
    pairs = filterWords(pairs)
    pairs = pairs[0:MAX_NUM_SAMPLES]



    y_s = [pair[2] for pair in pairs]
    pairs = [pair[0:2] for pair in pairs]

    y_s, endpoints = normalizeYs(y_s)



    return input_lang, output_lang, pairs, y_s, endpoints

def readLang(lang):
    print("Reading lines...")

    # Read the file and split into lines
    lines = open('data/%s-%s.txt' % (lang), encoding='utf-8').\
        read().strip().split('\n')

    # Split every line into pairs and normalize
    pairs = [[normalizeString(s) for s in l.split('\t')] for l in lines]

    # Reverse pairs, make Lang instances
    input_lang = Lang(lang)
    output_lang = Lang(lang)


    return input_lang, output_lang, pairs


eng_prefixes = (
    "i am ", "i m ",
    "he is", "he s ",
    "she is", "she s ",
    "you are", "you re ",
    "we are", "we re ",
    "they are", "they re "
)

def filterPair(p):
    return len(p[0].split(' ')) < MAX_LENGTH and \
        len(p[1].split(' ')) < MAX_LENGTH and \
        p[1].startswith(eng_prefixes)

def filterWord(w):
    return len(w) < MAX_LENGTH# and w.startswith(eng_prefixes)

def filterPairs(pairs):
    return [pair for pair in pairs if filterPair(pair)]

def filterWords(words):
    return [word for word in words if filterWord(word[0])]


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def indexesFromSentence(lang, sentence):
    return [lang.char2index[char] for char in sentence]

def tensorFromSentence(lang, sentence):
    indexes = indexesFromSentence(lang, sentence)
    indexes.append(EOS_token)
    padding = [0] * max(0, MAX_LENGTH - len(indexes))
    indexes += padding
    return torch.tensor(indexes, dtype=torch.long, device=DEVICE).view(1, -1)

def tensorsFromPair(pair, input_lang, output_lang):
    input_tensor = tensorFromSentence(input_lang, pair[0])
    target_tensor = tensorFromSentence(output_lang, pair[1])
    return (input_tensor, target_tensor)

def get_dataloader(file_name, batch_size):
    input_tensor, output_tensor, input_lang, output_lang, y_s, endpoints, pairs = get_data_tensors(file_name)
    train_data = TensorDataset(input_tensor, output_tensor, y_s)

    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)
    return input_lang, output_lang, train_dataloader, endpoints, pairs, y_s


# fix this. Allow an input_lang to be specified.
def get_data_tensors(file_name, prev_lang = None):
    input_lang, output_lang, pairs, y_s, endpoints = prepare_single_data(file_name, True, prev_lang)
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
    
    return torch.LongTensor(input_ids).to(DEVICE), torch.LongTensor(target_ids).to(DEVICE), input_lang, output_lang, torch.FloatTensor(y_s), endpoints, pairs

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


def prepare_single_data(lang1, reverse=False, prevLang = None):
    input_lang, output_lang, pairs, y_s, endpoints = read_single_lang(lang1, reverse, prevLang)

    if(prevLang is None):
        print("Trimmed to %s sentence pairs" % len(pairs))
        print("Counting chars...")
        for pair in pairs:
            input_lang.addSentence(pair[0])
            output_lang.addSentence(pair[1])
        print("Counted chars:")
    else:
        input_lang = prevLang
        output_lang = prevLang
    print(input_lang.name, input_lang.n_chars)
    print(output_lang.name, output_lang.n_chars)
    return input_lang, output_lang, pairs, y_s, endpoints
