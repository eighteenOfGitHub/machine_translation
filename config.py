import collections
import math
import torch
from torch import nn

import utils
from EncoderDecoder import *
from dataset import load_data_nmt

embed_size, num_hiddens, num_layers, dropout = 32, 32, 2, 0.1 
batch_size, num_steps = 64, 10
lr, num_epochs, device = 0.005, 200, utils.try_gpu()

train_iter, tgt_vocab, src_vocab = load_data_nmt(batch_size, num_steps)

ffn_num_input, ffn_num_hiddens, num_heads = 32, 32, 4
key_size, query_size, value_size = 32, 32, 32
norm_shape = [32]

net_name = 'EncoderDecoder_Transformer_fra2eng' # EncoderDecoder_GRU EncoderDecoder_LSTM EncoderDecoder_Transformer

# 法语到英语则交换源词库和目标词库
if 'fra2eng' in net_name:
    src_vocab, tgt_vocab = tgt_vocab, src_vocab

if 'EncoderDecoder_GRU' in net_name:
    encoder = Seq2SeqEncoder(len(src_vocab), embed_size, num_hiddens, num_layers, dropout, net_name='LSTM')
    decoder = Seq2SeqDecoder(len(tgt_vocab), embed_size, num_hiddens, num_layers, dropout, net_name='LSTM')
elif 'EncoderDecoder_LSTM' in net_name:
    encoder = Seq2SeqEncoder(len(src_vocab), embed_size, num_hiddens, num_layers, dropout, net_name='GRU')
    decoder = Seq2SeqDecoder(len(tgt_vocab), embed_size, num_hiddens, num_layers, dropout, net_name='GRU')
elif 'EncoderDecoder_Transformer' in net_name:
    encoder = TransformerEncoder(
        len(src_vocab), key_size, query_size, value_size, num_hiddens,
        norm_shape, ffn_num_input, ffn_num_hiddens, num_heads,
        num_layers, dropout)
    decoder = TransformerDecoder(
        len(tgt_vocab), key_size, query_size, value_size, num_hiddens,
        norm_shape, ffn_num_input, ffn_num_hiddens, num_heads,
        num_layers, dropout)

net = EncoderDecoder(encoder, decoder)