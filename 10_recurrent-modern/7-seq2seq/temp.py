import collections
import math
import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l

# the custom d2l class which is meant to download and play around with the eng-french data
class MTFraEng(d2l.DataModule): #@save
	"""The English-French dataset."""
	def _download(self):
		d2l.extract(d2l.download(
			d2l.DATA_URL+'fra-eng.zip', self.root,
			'94646ad1522d915e7b0f9296181140edcf86a4f5'))
		with open(self.root + '/fra-eng/fra.txt', encoding='utf-8') as f:
			return f.read()

	def __init__(self, batch_size, num_steps=9, num_train=512, num_val=128):
		super(MTFraEng, self).__init__()
		self.save_hyperparameters()
		self.arrays, self.src_vocab, self.tgt_vocab = self._build_arrays(
			self._download())

	def _build_arrays(self, raw_text, src_vocab=None, tgt_vocab=None):
		def _build_array(sentences, vocab, is_tgt=False):
			pad_or_trim = lambda seq, t: (
				seq[:t] if len(seq) > t else seq + ['<pad>'] * (t - len(seq)))
			sentences = [pad_or_trim(s, self.num_steps) for s in sentences]
			if is_tgt:
				sentences = [['<bos>'] + s for s in sentences]
			if vocab is None:
				vocab = d2l.Vocab(sentences, min_freq=2)
			array = torch.tensor([vocab[s] for s in sentences])
			valid_len = (array != vocab['<pad>']).type(torch.int32).sum(1)
			return array, vocab, valid_len
		src, tgt = self._tokenize(self._preprocess(raw_text), self.num_train + self.num_val)
		src_array, src_vocab, src_valid_len = _build_array(src, src_vocab)
		tgt_array, tgt_vocab, _ = _build_array(tgt, tgt_vocab, True)
		return ((src_array, tgt_array[:,:-1], src_valid_len, tgt_array[:,1:]),src_vocab, tgt_vocab)

	def _tokenize(self, text, max_examples=None):
		src, tgt = [], []
		for i, line in enumerate(text.split('\n')):
			if max_examples and i > max_examples: break
			parts = line.split('\t')
			if len(parts) == 2:
				# Skip empty tokens
				src.append([t for t in f'{parts[0]} <eos>'.split(' ') if t])
				tgt.append([t for t in f'{parts[1]} <eos>'.split(' ') if t])
		return src, tgt

	def _preprocess(self, text):
		# Replace non-breaking space with space
		text = text.replace('\u202f', ' ').replace('\xa0', ' ')
		# Insert space between words and punctuation marks
		no_space = lambda char, prev_char: char in ',.!?' and prev_char != ' '
		out = [' ' + char if i > 0 and no_space(char, text[i - 1]) else char
			for i, char in enumerate(text.lower())]
		return ''.join(out)

data = MTFraEng(batch_size = 128)

# function to initialize weights of a supplied module for seq2seq learning task
def init_seq2seq(module):
	# if the type of module is nn.Linear, then we simply initialize its weight matrix using
	# xavier uniform. If it is GRU, then we initialize only its weight matrix parameters using
	# xavier uniform (and not its bias vectors).
	if type(module) == nn.Linear:
		nn.init.xavier_uniform_(module.weight)
	if type(module) == nn.GRU:
		# the module is nn.GRU, so we loop through all its params names, and find the names of 
		# the params which are corresponding to weight matrices (which we want to initialize with xavier uniform)
		for param in module._flat_weights_names:
			if 'weight' in param:
				nn.init.xavier_uniform_(module._parameters[param]) # type: ignore

# encoder class for the seq2seq architecture
class Seq2SeqEncoder(nn.Module):
	"""RNN Encoder for sequence to sequence learning"""
	def __init__(self, vocab_size, input_size, hidden_size, num_layers):
		super().__init__()
		self.embedding = nn.Embedding(num_embeddings = vocab_size, embedding_dim = input_size)
		self.rnn = nn.GRU(input_size = input_size, hidden_size = hidden_size, num_layers = num_layers)
		# apply the initialization on every submodule of the class
		self.apply(init_seq2seq) # .apply method comes from nn.Module superclass, and it calls the specified function (here init_seq2seq) on every submodule of the nn.Module (as returned by .children())

	# now, the forward method, which takes in the input (batch_size, seq_length), and returns the
	# output, state (hidden states at all timepoints of last layer, and hidden state at last 
	# timepoint of all layers) when that input is passed through the encoder.
	def forward(self, X):
		# X stores the indices of the words in the embedding. Hence, shape is (batch_size, seq_length)
		embs = self.embedding(X.t().type(torch.int64))
		# embs shape: (batch_size, seq_length, input_size)
		output, states = self.rnn(embs)
		# outputs shape: (seq_length, batch_size, input_size)
		# states shape: (num_layers, batch_size, input_size)
		return output, states

# decoder class for seq2seq architecture
class Seq2SeqDecoder(nn.Module):
	def __init__(self, input_size, vocab_size, hidden_size, num_layers):
		super().__init__()
		# the nn.Embedding instance which stores mapping information from indices to word vector embeddings
		self.embedding = nn.Embedding(num_embeddings = vocab_size, embedding_dim = input_size)
		self.rnn = nn.GRU(input_size + hidden_size, hidden_size, num_layers)
		self.dense = nn.Linear(in_features = hidden_size, out_features = vocab_size)

	def init_state(self, enc_all_outputs, *args):
		return enc_all_outputs

	def forward(self, X, state): # state is the combined output of encoder (outputs, states)
		# shape of X (batch_size, X's seq_length)
		# first, convert X into input embedding style
		X = self.embedding(X.swapaxes(0,1)) # (X's seq_length, batch_size, input_size)
		# first, extract the information of the encoder
		# enc_states = sts
		# enc_outputs = outs
		enc_outputs, enc_states = state
		# get the context variable (hidden state at the last timepoint for last layer)
		context_var = enc_outputs[-1] # (batch_size, hidden_size)
		# repeat the context variable, X's (text sequence for output) seq_length times and append 
		# to X, the context variable
		input_to_rnn = torch.cat([X, context_var.repeat(X.shape[0], 1, 1)], dim = -1) # (seq_length, batch_size, input_size + hidden_size)
		# pass through the rnn, the input_to_rnn and enc_states
		dec_outputs, dec_states = self.rnn(input_to_rnn, enc_states)
		# finally, obtain the probs of words using dec_outputs
		dec_outputs = self.dense(dec_outputs).swapaxes(0,1)
		# dec_outputs shape: (batch_size, seq_length, vocab_size)
		return dec_outputs, [enc_outputs, dec_states]

# the overall class, which has encompasses the seq2seq encoder and decoder
class Seq2Seq(nn.Module):
	def __init__(self, encoder, decoder, tgt_pad, lr):
		self.encoder = encoder
		self.decoder = decoder
		self.lr = lr

	def loss(self, Y_hat, Y):
		l = 

	def forward(self, enc_X, dec_X, *args):
		enc_all_outputs = self.encoder(enc_X)
		return self.decoder(dec_X, enc_all_outputs)[0] # shape of returned object: (batch_size, seq_length, vocab_size)

	def configure_optimizers(self):
		return torch.optim.Adam(self.parameters(), lr = self.lr)

	# ignoring validation_step for now as I just want to run training for now.
	# def validation_step(self, batch):


# OKAY, I MADE THE WHOLE ENCODER AND DECODER CLASSES, BUT WASN'T ABLE TO ACTUALLY TEST TRAINING.
# THIS IS BECAUSE I NEED TO ALSO SET UP BEING ABLE TO PLAY WITH THE DATA. I PROBABLY NEED TO SET UP
# PROPER FILES SO THAT I CAN HAVE THE D2L CODE FOR THINGS LIKE DATA AND OTHER CLASSES IN THEM, 
# SO THAT I CAN ACTUALLY TRAIN A MODEL AND SEE IF IT WORKS. IDK IF I SHOULD DO THIS TO BE ABLE TO 
# ACTUALLY TEST TRAINING ON THE ENCODER/DECODER CODE I WROTE OR IF I SHOULD MOVE ON TO THE 
# TRANSFORMERS CHAPTER (WHERE INEVITABLY I WILL NEED TO WORK ON THIS TO BE ABLE TO TRAIN THAT 
# CODE) OR IF I SHOULD WORK ON THE KAGGLE THING...


# now, lets pass some input to gru_module. Dimensions of the input will be as follows:
# x = (seq_length, batch_size, input_size)
# h_0 = (1, batch_size, hidden_size) # here the first dimension indicates num_layers, which is a necessary shape in h_0.
seq_length = 10
batch_size = 100
input_size = 5
hidden_size = 3
vocab_size = 10000
num_layers = 2

temp_enc = Seq2SeqEncoder(vocab_size, input_size, hidden_size, num_layers)
x = torch.randint(low = 0, high = 10000, size = (batch_size, seq_length))
outs, sts = temp_enc(x)
temp_dec = Seq2SeqDecoder(input_size, vocab_size, hidden_size, num_layers)
dec_outputs, others = temp_dec(x, (outs, sts))
dec_outputs.shape

X = x
outs.shape
sts.shape
sts[-1].shape

sts.repeat(3,3,3).shape
sts.repeat(1,2,3,4,5).shape

temp = nn.Linear(in_features=10, out_features=10)
temp = nn.Conv2d(in_channels = 10, out_channels = 2, kernel_size = 2)
temp = nn.Sequential(nn.Linear(10,10), nn.ReLU(), nn.Linear(10,10))
temp = nn.Embedding(num_embeddings=vocab_size, embedding_dim=input_size)
temp = nn.GRU(input_size + hidden_size, hidden_size, num_layers)


x = torch.tensor([1,2,3], dtype = torch.float)
y = torch.randn((10,3))
z = torch.tensor([1,2,3])
p = torch.tensor([4,5,6])
torch.cat((z,p), 1)

z.type(torch.float)
z.dtype
y[:-1]


w = torch.empty(3, 5)
nn.init.xavier_uniform(w)
nn.init.xavier_uniform_(w)
nn.init.xavier_uniform_(w, gain=nn.init.calculate_gain('relu'))

x@y
x @ y.T



gru_module = nn.GRU(input_size = input_size, hidden_size = hidden_size, num_layers = 2, dropout = 0) # dropout has an effect only when we have multiple layers because dropout is between different layers
for i in gru_module.named_parameters():
	print(i)
x = torch.randn((seq_length, batch_size, input_size))
h_0 = torch.randn((1, batch_size, hidden_size))
# temp is a tuple with its first entry as each timepoint's hidden state for last layer (here, 
# the only layer). Its second entry is each layer's last timepoint hidden state
temp = gru_module(x, h_0) 

temp[0].shape
temp[1].shape

len(temp)

gru_module._flat_weights_names
len(gru_module.all_weights[0])
type(gru_module._flat_weights)
gru_module._all_weights


temp._parameters['weight']
temp.children()
for i in temp.children():
	print(i)

temp = nn.GRU(input_size = 10, hidden_size = 5, num_layers = 3)

len(temp._flat_weights)
len(temp._flat_weights_names)

# was just playing around a bit with the params of nn.gru instance. Tomorrow, play around with a 
# bit more (pass an input and see how it looks like), and try to understand that do we use 
# multiple gru instances for the idfferent timepoints/layers? and then proceed fwd with the 
# init_seq2seq func.
