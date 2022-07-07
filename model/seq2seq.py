import torch
import torch.nn as nn
from torch import Tensor


class Encoder(nn.Module):
    """编码器 解码器架构的基本编码器接口"""

    def __init__(self, **kwargs):
        super(Encoder, self).__init__()

    def forward(self, x, *args):
        raise NotImplementedError


class Decoder(nn.Module):
    """编码器 解码器架构的基本解码器接口"""

    def __init__(self, **kwargs):
        super(Decoder, self).__init__()

    def forward(self, x, state):
        raise NotImplementedError


class EncoderDecoder(nn.Module):
    def __init__(self, encoder: Encoder, decoder: Decoder, **kwargs):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, enc_x, dec_x, *args):
        enc_outputs = self.encoder(enc_x, *args)
        dec_state = self.decoder.init_state(enc_outputs, *args)
        return self.decoder(dec_x, dec_state)


class Seq2SeqEncoder(Encoder):
    def __init__(self, vocab_size, embedding_dim, num_hidden, num_layers, dropout=0, **kwargs):
        super(Seq2SeqEncoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.GRU(embedding_dim, num_hidden, num_layers, dropout=dropout, batch_first=True)

    def forward(self, x, *args):
        """
        :param x:  形状[batch_size,seq_len]
        :param args:
        :return:
        """
        # 进行词嵌入 embedded形状 [batch_size,seq_len,embedding_dim]
        embedded = self.embedding(x)
        # 获取最后输出 与 隐层状态
        # output形状与embedded相同(需要设置batch_first=True)
        # state形状 [num_layers,batch_size,embedding_dim]
        output, state = self.rnn(embedded)
        return output, state


class Seq2SeqDecoder(Decoder):
    def __init__(self, vocab_size, embedding_dim, num_hidden, num_layers, dropout=0, **kwargs):
        super(Seq2SeqDecoder, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.GRU(embedding_dim + num_hidden, num_hidden, num_layers, dropout=dropout, batch_first=True)
        self.classify = nn.Linear(num_hidden, vocab_size)

    def forward(self, x, state):
        """
        :param x: 形状batch_size,seq_len
        :param state:
        :return:
        """
        # 目标句子词嵌入
        x = self.embedding(x)
        # 拿编码器ht状态中的某一个layer和输入进行拼接 进一步获取输入序列的信息
        # layer的形状[batch_size,num_hidden]
        # 将其重复seq_len次 变成[seq_len,batch_size,num_hidden]
        context = state[-1].repeat(x.shape[1], 1, 1)
        # 进行拼接 形状 [batch_size,seq_len,num_hidden+embedding_dim]
        x_and_context = torch.cat((x, context.permute(1, 0, 2)), 2)
        # 将拼接后的信息作为rnn输入 输入下一步的hidden和out
        output, state = self.rnn(x_and_context, state)
        output = self.classify(output)
        return output, state


class MaskedSoftmaxCELoss(nn.CrossEntropyLoss):
    @classmethod
    def sequence_mask(cls, x: torch.Tensor, valid_len, value=0):
        max_length = x.size(1)
        mask = torch.arange(max_length, dtype=torch.float32, device=x.device)[None, :] < valid_len[:, None]
        x[~mask] = value
        return x

    def forward(self, input: Tensor, target: Tensor, valid_len: Tensor, value=0) -> Tensor:
        weights = torch.ones_like(target)
        weights = MaskedSoftmaxCELoss.sequence_mask(weights, valid_len, value)
        self.reduction = 'none'
        unweight_loss = super(MaskedSoftmaxCELoss, self).forward(input.permute(0, 2, 1), target)
        weighted_loss = (unweight_loss * weights).mean(dim=1)
        return weighted_loss
