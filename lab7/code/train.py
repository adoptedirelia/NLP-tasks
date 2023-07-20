import os 
os.environ["CUDA_VISIBLE_DEVICES"] = '4,5,6'

import torch
import torch.nn as nn 
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.utils import data

from tqdm import tqdm
import config
import argparse
import math
import collections
import re
import jieba
import model

def preprocess_en(text):
    def no_space(char, prev_char):
        return char in set(',.!?()\'') and prev_char != ' '

    text = text.replace('\u202f', ' ').replace('\xa0', ' ').lower()

    text = re.sub(r'[\'\"\(\)]','',text)
    out = [' ' + char if i > 0 and no_space(char, text[i - 1]) else char
           for i, char in enumerate(text)]
    return ''.join(out)

def preprocess_zh(text):
    def no_space(char, prev_char):
        return char in set(',.!?，。！？（）') and prev_char != ' '

    text = re.sub(r'[\'\"\(\)]','',text)

    text = jieba.lcut(text)
    
    # Insert space between words and punctuation marks
    out = [char if i == 0 else ' '+char
           for i, char in enumerate(text)]
    return ''.join(out)

def count_corpus(tokens):

    # Here `tokens` is a 1D list or 2D list
    if len(tokens) == 0 or isinstance(tokens[0], list):
        # Flatten a list of token lists into a list of tokens
        tokens = [token for line in tokens for token in line]
    return collections.Counter(tokens)

class Vocab:
    """Vocabulary for text."""
    def __init__(self, tokens=None, min_freq=0, reserved_tokens=None):
        """Defined in :numref:`sec_text_preprocessing`"""
        if tokens is None:
            tokens = []
        if reserved_tokens is None:
            reserved_tokens = []
        # Sort according to frequencies
        counter = count_corpus(tokens)
        self._token_freqs = sorted(counter.items(), key=lambda x: x[1],
                                   reverse=True)
        # The index for the unknown token is 0
        self.idx_to_token = ['<unk>'] + reserved_tokens
        self.token_to_idx = {token: idx
                             for idx, token in enumerate(self.idx_to_token)}
        for token, freq in self._token_freqs:
            if freq < min_freq:
                break
            if token not in self.token_to_idx:
                self.idx_to_token.append(token)
                self.token_to_idx[token] = len(self.idx_to_token) - 1

    def __len__(self):
        return len(self.idx_to_token)

    def __getitem__(self, tokens):
        if not isinstance(tokens, (list, tuple)):
            return self.token_to_idx.get(tokens, self.unk)
        return [self.__getitem__(token) for token in tokens]

    def to_tokens(self, indices):
        if not isinstance(indices, (list, tuple)):
            return self.idx_to_token[indices]
        return [self.idx_to_token[index] for index in indices]

    @property
    def unk(self):  # Index for the unknown token
        return 0

    @property
    def token_freqs(self):  # Index for the unknown token
        return self._token_freqs

def truncate_pad(line, num_steps, padding_token):
    """Truncate or pad sequences.

    Defined in :numref:`sec_machine_translation`"""
    if len(line) > num_steps:
        return line[:num_steps]  # Truncate
    return line + [padding_token] * (num_steps - len(line))  # Pad

def build_array_nmt(lines, vocab, num_steps):
    """Transform text sequences of machine translation into minibatches.
    
    Defined in :numref:`subsec_mt_data_loading`"""
    reduce_sum = lambda x, *args, **kwargs: x.sum(*args, **kwargs)
    astype = lambda x, *args, **kwargs: x.type(*args, **kwargs)
    lines = [vocab[l] for l in lines]
    lines = [l + [vocab['<eos>']] for l in lines]
    array = torch.tensor([truncate_pad(
        l, num_steps, vocab['<pad>']) for l in lines])
    valid_len = reduce_sum(
        astype(array != vocab['<pad>'], torch.int32), 1)
    return array, valid_len

def load_array(data_arrays, batch_size, is_train=True):
    """Construct a PyTorch data iterator.

    Defined in :numref:`sec_linear_concise`"""
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)

def mydata(args,cls=0,envoc=None,zhvoc=None):
    if cls == 0:
        path1 = './trans/train_20W/zh_20W'
        path2 = './trans/train_20W/en_20W'
    elif cls == 1:
        path1 = './trans/dev_and_test/dev_zh'
        path2 = './trans/dev_and_test/dev_en'
    else:
        path1 = './trans/dev_and_test/test_zh'
        path2 = './trans/dev_and_test/test_en'        
    
    file1 = open(path1)
    lines = file1.readlines()
    zh = []
    zhh = []
    if cls==0:
        lines = lines[:50000]

    for line in tqdm(lines):
        zh.append(preprocess_zh(line.strip()).split(" "))
        zhh.append(preprocess_zh(line.strip()))

    file2 = open(path2)
    lines = file2.readlines()
    en = []
    enn = []
    if cls == 0:
        lines = lines[:50000]

    for line in tqdm(lines):
        en.append(preprocess_en(line.strip()).split(" "))
        enn.append(preprocess_en(line.strip()))

    if envoc == None or zhvoc == None:
        envoc = Vocab(en,2,reserved_tokens=['<pad>', '<bos>', '<eos>'])

        zhvoc = Vocab(zh,2,reserved_tokens=['<pad>', '<bos>', '<eos>'])

    src_array, src_valid_len = build_array_nmt(en, envoc, args.num_steps)
    tgt_array, tgt_valid_len = build_array_nmt(zh, zhvoc, args.num_steps)
    
    data_arrays = (src_array, src_valid_len, tgt_array, tgt_valid_len)
    data_iter = load_array(data_arrays, args.batch_size)

    return data_iter,envoc,zhvoc,zhh,enn

def predict_seq2seq(net, src_sentence, src_vocab, tgt_vocab, num_steps,
                    device, save_attention_weights=False):
    """Predict for sequence to sequence.

    Defined in :numref:`sec_seq2seq_training`"""
    # Set `net` to eval mode for inference
    net.eval()
    src_tokens = src_vocab[src_sentence.lower().split(' ')] + [
        src_vocab['<eos>']]
    enc_valid_len = torch.tensor([len(src_tokens)]).cuda()
    src_tokens = truncate_pad(src_tokens, num_steps, src_vocab['<pad>'])
    # Add the batch axis
    enc_X = torch.unsqueeze(
        torch.tensor(src_tokens, dtype=torch.long).cuda(), dim=0)
    enc_outputs = net.encoder(enc_X, enc_valid_len)
    dec_state = net.decoder.init_state(enc_outputs, enc_valid_len)
    # Add the batch axis
    dec_X = torch.unsqueeze(torch.tensor(
        [tgt_vocab['<bos>']], dtype=torch.long).cuda(), dim=0)
    output_seq, attention_weight_seq = [], []
    for _ in range(num_steps):
        Y, dec_state = net.decoder(dec_X, dec_state)
        # We use the token with the highest prediction likelihood as the input
        # of the decoder at the next time step
        dec_X = Y.argmax(dim=2)
        pred = dec_X.squeeze(dim=0).type(torch.int32).item()
        # Save attention weights (to be covered later)
        if save_attention_weights:
            attention_weight_seq.append(net.decoder.attention_weights)
        # Once the end-of-sequence token is predicted, the generation of the
        # output sequence is complete
        if pred == tgt_vocab['<eos>']:
            break
        output_seq.append(pred)
    return ' '.join(tgt_vocab.to_tokens(output_seq)), attention_weight_seq


def get_parser():
    parser = argparse.ArgumentParser(description='Convolutional Neural Networks for Sentence Classification')
    parser.add_argument('--config', type=str, default='./config.yaml', help='config file')
    args = parser.parse_args()
    assert args.config is not None
    cfg = config.load_cfg_from_cfg_file(args.config)
    return cfg

def data_prepare(args):
    #先准备训练集
    train_iter,envoc,zhvoc,train_zh,train_en = mydata(args,0)
    val_iter,_,_,val_zh,val_en = mydata(args,1,envoc,zhvoc)
    test_iter,_,_,test_zh,test_en = mydata(args,2,envoc,zhvoc)

    return train_iter,val_iter,test_iter,envoc,zhvoc,test_zh,test_en


def train(train_iter,args,src_vocab,tgt_vocab):

    num_hiddens, num_layers, dropout = args.num_hiddens, args.num_layers,args.dropout
    ffn_num_input, ffn_num_hiddens, num_heads = args.ffn_num_input, args.ffn_num_hiddens, args.num_heads
    key_size, query_size, value_size = args.key_size, args.query_size, args.value_size
    norm_shape = args.norm_shape

    encoder = model.TransformerEncoder(
    len(src_vocab), key_size, query_size, value_size, num_hiddens,
    norm_shape, ffn_num_input, ffn_num_hiddens, num_heads,
    num_layers, dropout)
    decoder = model.TransformerDecoder(
        len(tgt_vocab), key_size, query_size, value_size, num_hiddens,
        norm_shape, ffn_num_input, ffn_num_hiddens, num_heads,
        num_layers, dropout)
    net = model.EncoderDecoder(encoder, decoder).cuda()
    net = nn.DataParallel(net)
    print(net)

    optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    loss = model.MaskedSoftmaxCELoss()

    for epoch in range(args.epoch):
        
        loss_sum,n = 0.0,0
        for batch in tqdm(train_iter):
            optimizer.zero_grad()
            X, X_valid_len, Y, Y_valid_len = [x.cuda() for x in batch]
            bos = torch.tensor([tgt_vocab['<bos>']] * Y.shape[0]).reshape(-1, 1).cuda()
            dec_input = torch.concat([bos, Y[:, :-1]], 1)  # Teacher forcing
            Y_hat, _ = net(X, dec_input, X_valid_len)
            l = loss(Y_hat, Y, Y_valid_len)
            #print(l,l.shape)
            loss_sum += l.sum()
            n += X.shape[0]
            l.sum().backward()  # Make the loss scalar for `backward`
            optimizer.step()

        print(f'epoch: {epoch+1} loss {loss_sum/n:.4f}')
    return net

def bleu(pred_seq, label_seq, k):
    """Compute the BLEU.

    Defined in :numref:`sec_seq2seq_training`"""
    pred_tokens, label_tokens = pred_seq.split(' '), label_seq.split(' ')
    len_pred, len_label = len(pred_tokens), len(label_tokens)
    score = math.exp(min(0, 1 - len_label / len_pred))
    for n in range(1, k + 1):

        num_matches, label_subs = 0, collections.defaultdict(int)
        for i in range(len_label - n + 1):
            label_subs[' '.join(label_tokens[i: i + n])] += 1
        for i in range(len_pred - n + 1):
            if label_subs[' '.join(pred_tokens[i: i + n])] > 0:
                num_matches += 1
                label_subs[' '.join(pred_tokens[i: i + n])] -= 1
        score *= math.pow(num_matches / (len_pred - n + 1), math.pow(0.5, n))
    return score

def eval(model,test_zh,test_en,src_vocab, tgt_vocab,args):
    score = 0
    model = model.module
    for eng, zh in zip(test_en, test_zh):
        translation, dec_attention_weight_seq = predict_seq2seq(
            model, eng, src_vocab, tgt_vocab, args.num_steps, 1, True)
        print(f'{eng} => {translation}, ',
            f'bleu {bleu(translation, zh, k=2):.3f}')
        score += bleu(translation, zh, k=2)
    print(f"total score: {score/len(test_en)}")

def main(args):
    train_iter,val_iter,test_iter,envoc,zhvoc,test_zh,test_en = data_prepare(args)
    for a,b,c,d in train_iter:
        print(a.shape)
        print(b.shape)
        print(c.shape)
        print(d.shape)
        break

    model = train(train_iter,args,envoc,zhvoc)
    eval(model,test_zh,test_en,envoc,zhvoc,args)


if __name__ == '__main__':
    args = get_parser()
    print(args)
    main(args)