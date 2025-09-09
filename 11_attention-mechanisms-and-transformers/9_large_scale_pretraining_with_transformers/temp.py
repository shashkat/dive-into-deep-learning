# BELOW IS SOME TEMP CODE I WROTE WHILE GOING THROUGH THIS CHAPTER. ALSO, I BELIEVE THE 
# EXERCISES FOR THIS CHAPTER ARE NOT THAT USEFUL, HENCE NOT DOING THEM NOW.

# from d2l import torch as d2l
# from torch import nn
# import torch
# import math

# data = d2l.MTFraEng(batch_size=128)

# num_hiddens, num_blks, dropout = 256, 2, 0.2
# ffn_num_hiddens, num_heads = 64, 4

# class TransformerDecoderBlock(nn.Module):
#     # The i-th block in the Transformer decoder
#     def __init__(self, num_hiddens, ffn_num_hiddens, num_heads, dropout, i):
#         super().__init__()
#         self.i = i
#         self.attention1 = d2l.MultiHeadAttention(num_hiddens, num_heads,
#                                                  dropout)
#         self.addnorm1 = d2l.AddNorm(num_hiddens, dropout)
#         self.attention2 = d2l.MultiHeadAttention(num_hiddens, num_heads,
#                                                  dropout)
#         self.addnorm2 = d2l.AddNorm(num_hiddens, dropout)
#         self.ffn = d2l.PositionWiseFFN(ffn_num_hiddens, num_hiddens)
#         self.addnorm3 = d2l.AddNorm(num_hiddens, dropout)

#     def forward(self, X, state):
#         enc_outputs, enc_valid_lens = state[0], state[1]
#         # During training, all the tokens of any output sequence are processed
#         # at the same time, so state[2][self.i] is None as initialized. When
#         # decoding any output sequence token by token during prediction,
#         # state[2][self.i] contains representations of the decoded output at
#         # the i-th block up to the current time step
#         if state[2][self.i] is None:
#             key_values = X
#         else:
#             key_values = torch.cat((state[2][self.i], X), dim=1)
#         state[2][self.i] = key_values
#         if self.training:
#             batch_size, num_steps, _ = X.shape
#             # Shape of dec_valid_lens: (batch_size, num_steps), where every
#             # row is [1, 2, ..., num_steps]
#             dec_valid_lens = torch.arange(
#                 1, num_steps + 1, device=X.device).repeat(batch_size, 1)
#         else:
#             dec_valid_lens = None
#         # Self-attention
#         X2 = self.attention1(X, key_values, key_values, dec_valid_lens)
#         Y = self.addnorm1(X, X2)
#         # Encoder-decoder attention. Shape of enc_outputs:
#         # (batch_size, num_steps, num_hiddens)
#         Y2 = self.attention2(Y, enc_outputs, enc_outputs, enc_valid_lens)
#         Z = self.addnorm2(Y, Y2)
#         return self.addnorm3(Z, self.ffn(Z)), state

# class TransformerDecoder(d2l.AttentionDecoder):
#     def __init__(self, vocab_size, num_hiddens, ffn_num_hiddens, num_heads,
#                  num_blks, dropout):
#         super().__init__()
#         self.num_hiddens = num_hiddens
#         self.num_blks = num_blks
#         self.embedding = nn.Embedding(vocab_size, num_hiddens)
#         self.pos_encoding = d2l.PositionalEncoding(num_hiddens, dropout)
#         self.blks = nn.Sequential()
#         for i in range(num_blks):
#             self.blks.add_module("block"+str(i), TransformerDecoderBlock(
#                 num_hiddens, ffn_num_hiddens, num_heads, dropout, i))
#         self.dense = nn.LazyLinear(vocab_size)

#     def init_state(self, enc_outputs, enc_valid_lens):
#         return [enc_outputs, enc_valid_lens, [None] * self.num_blks]

#     def forward(self, X, state):
#         X = self.pos_encoding(self.embedding(X) * math.sqrt(self.num_hiddens))
#         self._attention_weights = [[None] * len(self.blks) for _ in range (2)]
#         for i, blk in enumerate(self.blks):
#             X, state = blk(X, state)
#             # Decoder self-attention weights
#             self._attention_weights[0][i] = blk.attention1.attention.attention_weights
#             # Encoder-decoder attention weights
#             self._attention_weights[1][i] = blk.attention2.attention.attention_weights
#         return self.dense(X), state

#     @property
#     def attention_weights(self):
#         return self._attention_weights

# encoder = d2l.TransformerEncoder(
#     len(data.src_vocab), num_hiddens, ffn_num_hiddens, num_heads,
#     num_blks, dropout)
# decoder = TransformerDecoder(
#     len(data.tgt_vocab), num_hiddens, ffn_num_hiddens, num_heads,
#     num_blks, dropout)
# model = d2l.Seq2Seq(encoder, decoder, tgt_pad=data.tgt_vocab['<pad>'],
#                     lr=0.001)

# trainer = d2l.Trainer(max_epochs=30, gradient_clip_val=1, num_gpus=1)

# model.validation_step()

# batch = next(iter(data.val_dataloader()))
# y_hat = model(*batch[:-1])
# y_hat.shape

# trainer.fit(model, data)
# import matplotlib.pyplot as plt
# plt.show()

# # doing predictions and looking into how the prediction is done token by token.
# engs = ['go .', 'i lost .', 'he\'s calm .', 'i\'m home .']
# fras = ['va !', 'j\'ai perdu .', 'il est calme .', 'je suis chez moi .']
# preds, _ = model.predict_step(
#     data.build(engs, fras), d2l.try_gpu(), data.num_steps)
# for en, fr, p in zip(engs, fras, preds):
#     translation = []
#     for token in data.tgt_vocab.to_tokens(p):
#         if token == '<eos>':
#             break
#         translation.append(token)
#     print(f'{en} => {translation}, bleu,'
#           f'{d2l.bleu(" ".join(translation), fr, k=2):.3f}')
