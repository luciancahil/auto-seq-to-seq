SOS_token = 0
EOS_token = 1

class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.char2index = {}
        self.word2count = {}
        self.car2count  = {}
        self.index2word = {0: "SOS", 1: "EOS"}
        self.index2char = {0: "SOS", 1: "EOS"}
        self.n_words = 2  # Count SOS and EOS
        self.n_chars = 2  # Count SOS and EOS


    def addSentence(self, sentence):
        for c in sentence:
            self.add_char(c)
        
    
    def add_char(self, c):
        if c not in self.char2index:
            self.char2index[c] = self.n_chars
            self.car2count[c] = 1
            self.index2char[self.n_chars] = c
            self.n_chars += 1
        else:
            self.car2count[c] += 1


    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1