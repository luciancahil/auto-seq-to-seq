import unicodedata
import re
from Lang import Lang
import torch
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import matplotlib.ticker as ticker
import numpy as np
import time
import math


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


def showPlot(points):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)

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


def read_single_lang(lang, reverse=False):
    print("Reading lines...")

    # Read the file and split into lines
    lines = open('data/%s.txt' % (lang), encoding='utf-8').\
        read().strip().split('\n')

    # Split every line into pairs and normalize
    pairs = [[normalizeString(l), normalizeString(l)] for l in lines]

    # Reverse pairs, make Lang instances
    if reverse:
        pairs = [list(reversed(p)) for p in pairs]
        input_lang = Lang(lang)
        output_lang = Lang(lang)
    else:
        input_lang = Lang(lang)
        output_lang = Lang(lang)

    return input_lang, output_lang, pairs

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

MAX_LENGTH = 40

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
    return len(w) < MAX_LENGTH and w.startswith(eng_prefixes)

def filterPairs(pairs):
    return [pair for pair in pairs if filterPair(pair)]

def filterWords(words):
    return [word for word in words if filterWord(word[0])]


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


token_to_enum = {'g': 2, 'o': 3, '.': 4, 'r': 5, 'u': 6, 'n': 7, '!': 8, 'w': 9, 'f': 10, 'i': 11, 'e': 12, 'h': 13, 'l': 14, 'p': 15, 'j': 16, 'm': 17, 's': 18, 't': 19, 'a': 20, ' ': 21, 'y': 22, 'c': 23, 'k': 24, '?': 25, "'": 26, 'b': 27, 'd': 28, 'q': 29, ',': 30, 'v': 31, 'z': 32, 'x': 33, '0': 34, '-': 35, '"': 36}
enum_to_token = ['[START]', '[END]', 'g', 'o', '.', 'r', 'u', 'n', '!', 'w', 'f', 'i', 'e', 'h', 'l', 'p', 'j', 'm', 's', 't', 'a', ' ', 'y', 'c', 'k', '?', "'", 'b', 'd', 'q', ',', 'v', 'z', 'x', '0', '-', '"']