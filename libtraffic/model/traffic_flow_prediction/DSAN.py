import numpy as np
import torch
import torch.nn as nn
from libtraffic.model.abstract_traffic_state_model import AbstractTrafficStateModel
from libtraffic.model.loss import masked_rmse_torch


def get_angles(pos, l, d):
    """
    equ (5)
    Args:
        pos: row(r) / column(c) in equ (5)
        l: the l-th dimension, with shape (1, d)
        d: d dimension in total
    Returns: angles with shape (1, d)

    """
    angle_rates = 1 / np.power(10000, (2 * (l // 2)) / np.float32(d))
    return torch.tensor(pos * angle_rates)


def spatial_posenc(r, c, d, device):
    """
    get SPE
    Args:
        r: row of the spatial position
        c: column of the spatial position
        d: d dimension in total

    Returns:

    """
    angle_rads_r = get_angles(pos=r, l=np.arange(d)[np.newaxis, :], d=d)  # l and ret with shape (1, d)

    angle_rads_c = get_angles(pos=c, l=np.arange(d)[np.newaxis, :], d=d)  # l and ret with shape (1, d)

    pos_encoding = torch.zeros(size=angle_rads_r.shape, device=device)  # shape (1, d)

    pos_encoding[:, 0::2] = torch.sin(angle_rads_r[:, 0::2])  # from 0 to d step 2

    pos_encoding[:, 1::2] = torch.cos(angle_rads_c[:, 1::2])  # from 1 to d step 2

    return pos_encoding[np.newaxis, ...]  # (1, 1, d)


def cal_attention(Q, K, V, M, n_h):
    """
    equ (3), calculate the attention mechanism performed by the i-th attention head
    Args:
        Q: query, shape (N, h, L_q, d)
        K: key, shape (N, h, L_k, d)
        V: value, shape (N, h, L_k, d)
        M: mask, shape (N, h, L_q, L_k)
        n_h: number of attention head

    Returns:
        Att: shape # (N, h, L_q, d)
    """

    QK = torch.matmul(input=Q, other=K.transpose(-1, -2))  # (h, L_q, L_k)

    d = K.shape[-1]
    d_h = d / n_h  # the split dimensionality in n_h attention heads
    QK_d_h = QK / np.sqrt(d_h)  # (h, L_q, L_k)

    if M is not None:
        M = M.unsqueeze(2)
        M = M.repeat(QK_d_h.shape[0] // M.shape[0], 1, 1, 1, 1)
        QK_d_h += (M * -1e9)

    attention_weights = torch.softmax(input=QK_d_h, dim=-1)  # (h, L_q, L_k) softmax along key axis

    output = torch.matmul(attention_weights, V)  # (h, L_q, d)

    return output


def two_layer_ffn(d, num_hid, input_dim):
    """
    implementation of two-layer feed-forward network
    Args:
        d: d-dimension representations
        num_hid: hidden layer size
        input_dim: input feature dimension

    Returns:

    """
    return nn.Sequential(
        nn.Linear(input_dim, num_hid),
        nn.ReLU(),
        nn.Linear(num_hid, d)
    )


def ex_encoding(d, num_hid, input_dim):
    """
    implementation of TPE
    Args:
        d: d-dimension representations
        num_hid: hidden layer size
        input_dim: input feature dimension

    Returns:

    """
    return nn.Sequential(
        nn.Linear(in_features=input_dim, out_features=num_hid),  
        nn.ReLU(),
        nn.Linear(in_features=num_hid, out_features=d),
        nn.Sigmoid()
    )


def create_look_ahead_mask(size):
    mask = 1 - torch.tril(torch.ones((size, size)), diagonal=-1)
    return mask.cuda()


def create_threshold_mask(inp):
    """

    Args:
        inp: [batch_size, input_window, column, row, input_dim]

    Returns:

    """

    oup = torch.sum(inp, dim=-1)
    shape = oup.shape
    oup = torch.reshape(oup, [shape[0], shape[1], -1])
    mask = (oup == 0).float()
    return mask


def create_threshold_mask_tar(inp):
    oup = torch.sum(inp, dim=-1)
    mask = (oup == 0).float()
    return mask


def create_masks(inp_g, inp_l, tar):
    """

    Args:
        inp_g: shape == [batch_size, input_window, column, row, input_dim]
        inp_l: shape == [batch_size, input_window, column, row, l_d, l_d, input_dim] torch.Size([64, 12, 192, 49, 2])
        tar: shape == [batch_size, input_window, N, ext_dim] torch.Size([64, 12, 192, 8])

    Returns:

    """

    threshold_mask_g = create_threshold_mask(inp_g).unsqueeze(2)
    inp_l = inp_l.permute([0, 2, 3, 1, 4, 5, 6])
    inp_l = torch.reshape(inp_l,
                          [inp_l.shape[0] * inp_l.shape[1] * inp_l.shape[2], inp_l.shape[3], inp_l.shape[4],
                           inp_l.shape[5], inp_l.shape[6]])
    threshold_mask = create_threshold_mask(inp_l).unsqueeze(2)
    look_ahead_mask = create_look_ahead_mask(tar.shape[1])
    tar = tar.permute(0, 2, 1, 3)
    tar = torch.reshape(tar, [tar.shape[0] * tar.shape[1], tar.shape[2], tar.shape[3]])
    dec_target_threshold_mask = create_threshold_mask_tar(
        tar).unsqueeze(1).unsqueeze(2)
    combined_mask = torch.max(dec_target_threshold_mask, look_ahead_mask)
    return threshold_mask_g, threshold_mask, combined_mask


class Convs(nn.Module):
    """
    Conv layers for input, to form a d-dimension representation
    """

    def __init__(self, n_layer, n_filter, input_window, input_dim, r_d=0.1):
        """
        Args:
            n_layer: num of conv layers
            n_filter: num of filters
            input_window: input window size
            input_dim: input dimension size
            r_d: dropout rate
        """
        super(Convs, self).__init__()

        self.n_layer = n_layer
        self.input_window = input_window

        self.convs = nn.ModuleList(
            [nn.ModuleList([nn.Conv2d(in_channels=input_dim, out_channels=n_filter, kernel_size=(3, 3), padding=(1, 1))
                            for _ in range(input_window)])])
        self.convs += nn.ModuleList(
            [nn.ModuleList([nn.Conv2d(in_channels=n_filter, out_channels=n_filter, kernel_size=(3, 3), padding=(1, 1))
                            for _ in range(input_window)]) for _ in range(n_layer - 1)])
        self.dropouts = nn.ModuleList([nn.ModuleList(
            [nn.Dropout(r_d) for _ in range(input_window)]) for _ in range(n_layer)])

    def forward(self, inps):
        """

        Args:
            inps: with shape [batch_size, input_window, row, column, N_d, input_dim]
                    or [batch_size, input_window, row, column, input_dim]

        Returns:

        """

        outputs = list(torch.split(inps, 1, dim=1))
        if len(inps.shape) == 6:
            for i in range(self.input_window):
                outputs[i] = outputs[i].permute([0, 1, 4, 5, 2, 3])
                outputs[i] = torch.reshape(input=outputs[i],
                                           shape=[-1, outputs[i].shape[3], outputs[i].shape[4], outputs[i].shape[5]])
        else:
            for i in range(self.input_window):
                outputs[i] = outputs[i].permute([0, 1, 4, 2, 3])
                outputs[i] = torch.reshape(input=outputs[i],
                                           shape=[-1, outputs[i].shape[2], outputs[i].shape[3], outputs[i].shape[4]])

        for i in range(self.n_layer):
            for j in range(self.input_window):
                outputs[j] = self.convs[i][j](outputs[j])

                outputs[j] = torch.relu(outputs[j])
                outputs[j] = self.dropouts[i][j](outputs[j])

        output = torch.stack(outputs, dim=1)
        if len(inps.shape) == 6:
            output = torch.reshape(input=output,
                                   shape=[inps.shape[0], -1, output.shape[1], output.shape[2], output.shape[3],
                                          output.shape[4]]).permute([0, 2, 4, 5, 1, 3])
        else:
            output = output.permute([0, 1, 3, 4, 2])

        return output


class MSA(nn.Module):
    """
    Multi-space attention
    """

    def __init__(self, d, n_h, self_att=True):
        """
        Args:
            d: d-dimension representations after B-layer CNN/FCN
            n_h: num of head
            self_att: whether use self attention
        """
        super(MSA, self).__init__()
        self.d = d
        self.n_h = n_h
        self.d_h = d / n_h
        self.self_att = self_att

        assert d % n_h == 0
        self.d_h = d // n_h

        if self_att:
            self.wx = nn.Linear(in_features=d, out_features=d * 3)
        else:
            self.wq = nn.Linear(in_features=d, out_features=d)
            self.wkv = nn.Linear(in_features=d, out_features=d * 2)

        self.wo = nn.Linear(in_features=d, out_features=d)

    def split_heads(self, x):
        """

        Args:
            x: shape == [batch_size, input_window, N, d]

        Returns:
                shape == [batch_size, input_window, n_h, N, d_h]
        """
        shape = x.shape
        x = torch.reshape(x, [shape[0], shape[1], shape[2], self.n_h, self.d_h])
        return x.permute([0, 1, 3, 2, 4])

    def forward(self, V, K, Q, M):

        # linear
        if self.self_att:
            wx_o = self.wx(Q)
            Q, K, V = torch.split(tensor=wx_o, split_size_or_sections=wx_o.shape[-1] // 3, dim=-1)
        else:
            Q = self.wq(Q)
            wkv_o = self.wkv(K)
            K, V = torch.split(tensor=wkv_o, split_size_or_sections=wkv_o.shape[-1] // 2, dim=-1)

        # split head
        Q = self.split_heads(Q)
        K = self.split_heads(K)
        V = self.split_heads(V)

        scaled_attention = cal_attention(Q=Q, K=K, V=V, M=M, n_h=self.n_h)

        scaled_attention = scaled_attention.permute([0, 1, 3, 2, 4])

        d_shape = scaled_attention.shape
        concat_attention = torch.reshape(
            scaled_attention, (d_shape[0], d_shape[1], d_shape[2], self.d))

        output = self.wo(concat_attention)
        return output


class EncoderLayer(nn.Module):
    """
    Enc-G implementation
    """

    def __init__(self, d, n_h, num_hid, r_d=0.1):
        """
        Args:
            d: d-dimension representations
            n_h: number of heads in Multi-space attention
            num_hid: hidden layer size
            r_d: drop out rate
        """
        super(EncoderLayer, self).__init__()

        # msa
        self.msa = MSA(d=d, n_h=n_h)

        # ffn
        self.ffn = two_layer_ffn(d=d, num_hid=num_hid, input_dim=64)

        # normalization
        self.layernorm1 = nn.LayerNorm(normalized_shape=d, eps=1e-6)
        self.layernorm2 = nn.LayerNorm(normalized_shape=d, eps=1e-6)

        # dropout
        self.dropout1 = nn.Dropout(r_d)
        self.dropout2 = nn.Dropout(r_d)

    def forward(self, x, mask):
        """

        Args:
            x: shape == [batch_size, input_window, N, d]
            mask:

        Returns:

        """

        # msa
        attn_output = self.msa(x, x, x, mask)
        attn_output = self.dropout1(attn_output)
        out1 = self.layernorm1(x + attn_output)  # Residual

        # ffn
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output)
        out2 = self.layernorm2(out1 + ffn_output)  # Residual

        return out2


class DecoderLayer(nn.Module):
    """
    Enc-D / Dec-S / Dec-T implementation
    """

    def __init__(self, d, n_h, num_hid, r_d=0.1, revert_q=False):
        """
        Args:
           d: d-dimension representations
            n_h: number of heads in Multi-space attention
            num_hid: hidden layer size
            r_d: drop out rate
            revert_q:
        """
        super(DecoderLayer, self).__init__()

        self.revert_q = revert_q

        self.msa1 = MSA(d=d, n_h=n_h)
        self.msa2 = MSA(d=d, n_h=n_h, self_att=False)

        self.ffn = two_layer_ffn(d=d, num_hid=num_hid, input_dim=d)

        self.layernorm1 = nn.LayerNorm(normalized_shape=[d], eps=1e-6)
        self.layernorm2 = nn.LayerNorm(normalized_shape=[d], eps=1e-6)
        self.layernorm3 = nn.LayerNorm(normalized_shape=[d], eps=1e-6)

        self.dropout1 = nn.Dropout(r_d)
        self.dropout2 = nn.Dropout(r_d)
        self.dropout3 = nn.Dropout(r_d)

    def forward(self, x, kv, look_ahead_mask, threshold_mask):

        # first msa
        attn1 = self.msa1(x, x, x, look_ahead_mask)
        attn1 = self.dropout1(attn1)
        out1 = self.layernorm1(attn1 + x)

        if self.revert_q:
            out1_r = out1.permute([0, 2, 1, 3])
            attn2 = self.msa2(kv, kv, out1_r, threshold_mask)
            attn2 = attn2.permute([0, 2, 1, 3])
        else:
            kv = kv.repeat(out1.shape[0] // kv.shape[0], 1, 1, 1)
            attn2 = self.msa2(kv, kv, out1, threshold_mask)
        attn2 = self.dropout2(attn2)
        out2 = self.layernorm2(attn2 + out1)

        ffn_output = self.ffn(out2)
        ffn_output = self.dropout3(ffn_output)
        out3 = self.layernorm3(ffn_output + out2)

        return out3


class DAE(nn.Module):
    """
    DAE Dynamic Attention Encoder
    """

    def __init__(self, L, d, n_h, num_hid, conv_layer, input_window, input_dim, ext_dim, r_d=0.1):
        """
        Dynamic Attention Encoder
        Args:
            L: num of Enc-G/Enc-D layers
            d: d-dimension representations
            n_h: number of heads in Multi-space attention
            num_hid: hidden layer size
            conv_layer: num of conv layers
            input_window: input window size
            input_dim: input dimension
            r_d: drop out rate
        """
        super(DAE, self).__init__()

        self.d = d
        self.L = L

        # conv layers to get d-dimension representations
        self.convs_d = Convs(n_layer=conv_layer, n_filter=d, input_window=input_window, input_dim=input_dim, r_d=r_d)
        self.convs_g = Convs(n_layer=conv_layer, n_filter=d, input_window=input_window, input_dim=input_dim, r_d=r_d)

        # get TPE
        self.ex_encoder_d = ex_encoding(d=d, num_hid=num_hid, input_dim=ext_dim)
        self.ex_encoder_g = ex_encoding(d=d, num_hid=num_hid, input_dim=ext_dim)

        # dropout layer
        self.dropout_d = nn.Dropout(p=r_d)
        self.dropout_g = nn.Dropout(p=r_d)

        #
        self.Enc_G = nn.ModuleList(
            [EncoderLayer(d=d, n_h=n_h, num_hid=num_hid, r_d=r_d) for _ in range(L)])
        self.Enc_D = nn.ModuleList(
            [DecoderLayer(d=d, n_h=n_h, num_hid=num_hid, r_d=r_d) for _ in range(L)])

    def forward(self, x_d, x_g, ex, cors_d, cors_g, threshold_mask_d, threshold_mask_g):
        """

        Args:
            x_d: a subset of 𝑿 that contains the closest neighbors that share strong correlations
                    with v_i within a local block.(X_d in figure 4)
            x_g: all the training data (X in figure 4)
            ex: time-related features for Temporal Positional Encoding
            cors_d:  Spatial Positional Encoding of x_d
            cors_g: Spatial Positional Encoding of x_g
            threshold_mask_d:
            threshold_mask_g:

        Returns:

        """

        shape = x_d.shape

        TPE_d = self.ex_encoder_d(ex)
        TPE_g = self.ex_encoder_g(ex)

        SPE_d = cors_d
        SPE_g = cors_g

        x_d = self.convs_d(inps=x_d)
        x_g = self.convs_g(inps=x_g)

        x_d *= np.sqrt(self.d)
        x_g *= np.sqrt(self.d)

        x_d = x_d.reshape([shape[0], shape[1], -1, shape[4], self.d])
        x_g = x_g.reshape([shape[0], shape[1], -1, self.d])

        TPE_d = torch.reshape(input=TPE_d, shape=[TPE_d.shape[0], TPE_d.shape[1], TPE_d.shape[2] * TPE_d.shape[3], -1,
                                                  TPE_d.shape[-1]])
        TPE_g = torch.reshape(input=TPE_g,
                              shape=[TPE_g.shape[0], TPE_g.shape[1], TPE_g.shape[2] * TPE_g.shape[3], TPE_g.shape[4]])

        x_d = x_d + TPE_d + SPE_d
        x_g = x_g + TPE_g + SPE_g

        x_d = self.dropout_d(x_d)
        x_g = self.dropout_g(x_g)

        for i in range(self.L):
            x_g = self.Enc_G[i](x_g, threshold_mask_g)

        x_d_ = x_d.permute([0, 2, 1, 3, 4])
        x_d_ = torch.reshape(x_d_, [x_d_.shape[0] * x_d_.shape[1], x_d_.shape[2], x_d_.shape[3], x_d_.shape[4]])

        for i in range(self.L):
            x_d = self.Enc_D[i](
                x_d_, x_g, threshold_mask_d, threshold_mask_g)

        return x_d


class SAD(nn.Module):
    def __init__(self, L, d, n_h, num_hid, conv_layer, ext_dim, input_window, output_window, device, r_d=0.1):
        """

        Args:
            L: num of Enc-G/Enc-D layers
            d: d-dimension representations
            n_h: number of heads in Multi-space attention
            num_hid: hidden layer size
            conv_layer: num of conv layers
            ext_dim: external data dimension
            input_window:
            output_window:
            r_d: drop out rate
        """
        super(SAD, self).__init__()

        self.d = d
        self.L = L
        self.pos_enc = spatial_posenc(0, 0, self.d, device)
        self.output_window = output_window

        self.ex_encoder = ex_encoding(d=d, num_hid=num_hid, input_dim=ext_dim)
        self.dropout = nn.Dropout(r_d)

        self.li_conv = nn.Sequential()
        self.li_conv.add_module("linear", nn.Linear(2, d))
        self.li_conv.add_module("activation_relu", nn.ReLU())
        for i in range(conv_layer - 1):
            self.li_conv.add_module(
                "linear{}".format(i), nn.Linear(
                    d, d))
            self.li_conv.add_module("activation_relu{}".format(i), nn.ReLU())

        self.dec_s = nn.ModuleList(
            [DecoderLayer(d=d, n_h=n_h, num_hid=num_hid, r_d=r_d) for _ in range(L)])

        self.linear = nn.Linear(in_features=input_window, out_features=output_window)
        self.dec_t = nn.ModuleList(
            [DecoderLayer(d=d, n_h=n_h, num_hid=num_hid, r_d=r_d, revert_q=True) for _ in range(L)])

    def forward(self, x, ex, dae_output, look_ahead_mask):
        ex_enc = self.ex_encoder(ex)

        x = self.li_conv(x)
        x *= np.sqrt(self.d)

        ex_enc = torch.reshape(input=ex_enc, shape=[ex_enc.shape[0], ex_enc.shape[1], ex_enc.shape[2] * ex_enc.shape[3],
                                                    ex_enc.shape[4]])
        x = x + ex_enc + self.pos_enc

        x = self.dropout(x)
        x_s = x
        x_t = x
        x_s = x_s.unsqueeze(3).expand(-1, -1, -1, self.output_window, -1)

        x_s_ = x_s.permute([0, 2, 1, 3, 4])
        x_s_ = torch.reshape(x_s_, [x_s_.shape[0] * x_s_.shape[1], x_s_.shape[2], x_s_.shape[3], x_s_.shape[4]])

        # linear
        dae_output = dae_output.permute(0, 2, 3, 1)
        dae_output = self.linear(dae_output)
        dae_output = dae_output.permute(0, 3, 1, 2)

        for i in range(self.L):
            x_s = self.dec_s[i](x_s_, dae_output, look_ahead_mask, None)

        x_s_ = x_s.permute([0, 2, 1, 3])
        x_t_ = x_t.permute([0, 2, 1, 3])
        x_t_ = torch.reshape(x_t_, [x_t_.shape[0] * x_t_.shape[1], 1, x_t_.shape[2], x_t_.shape[3]])
        for i in range(self.L):
            x_t = self.dec_t[i](
                x_t_, x_s_, look_ahead_mask, None)

        output = x_t.squeeze(1)

        return output


class DsanUse(nn.Module):
    """
    DSAN use
    """

    def __init__(self, L, d, n_h, row, column, num_hid, conv_layer, input_window, output_window, input_dim,
                 ext_dim,
                 device, r_d=0.1):
        """

        Args:
            L: num of layers in Enc-G/D / Dec-S/T
            d: d-dimension representations
            n_h: number of heads in Multi-space attention
            num_hid:
            conv_layer: num of conv layers
            input_window: input window size
            input_dim: input dimension
            r_d: dropout rate
        """
        super(DsanUse, self).__init__()

        self.row = row
        self.column = column

        # DAE Dynamic Attention Encoder
        self.dae = DAE(L=L, d=d, n_h=n_h, num_hid=num_hid, conv_layer=conv_layer,
                       input_window=input_window, input_dim=input_dim, ext_dim=ext_dim, r_d=r_d)

        # SAD Switch-Attention Decoder
        self.sad = SAD(L=L, d=d, n_h=n_h, num_hid=num_hid, conv_layer=conv_layer, ext_dim=ext_dim,
                       input_window=input_window, output_window=output_window, device=device, r_d=r_d)

        # final layer
        self.final_layer = nn.Linear(d, input_dim)

    def forward(self, dae_inp_g, dae_inp, dae_inp_ex, sad_inp, sad_inp_ex, cors, cors_g, threshold_mask,
                threshold_mask_g, look_ahead_mask):
        # DAE
        dae_output = self.dae(
            x_d=dae_inp,
            x_g=dae_inp_g,
            ex=dae_inp_ex,
            cors_d=cors,
            cors_g=cors_g,
            threshold_mask_d=threshold_mask,
            threshold_mask_g=threshold_mask_g
        )

        # SAD
        sad_output = self.sad(
            x=sad_inp,
            ex=sad_inp_ex,
            dae_output=dae_output,
            look_ahead_mask=look_ahead_mask
        )

        # final layer
        final_output = self.final_layer(sad_output)
        final_output = torch.tanh(final_output)
        final_output = torch.reshape(final_output,
                                     [-1, self.column, self.row, final_output.shape[-2], final_output.shape[-1]])

        final_output = final_output.permute([0, 3, 2, 1, 4])

        return final_output


class DSAN(AbstractTrafficStateModel):
    def __init__(self, config, data_feature):
        super().__init__(config, data_feature)

        # device
        self.device = config.get('device', torch.device('cpu'))

        # data_feature
        self._scaler = self.data_feature.get('scaler')  # 用于数据归一化
        # self.adj_mx = torch.tensor(self.data_feature.get('adj_mx'), device=self.device)
        self.len_row = self.data_feature.get('len_row', 16)  # row
        self.len_column = self.data_feature.get('len_column', 12)  # column
        self.num_nodes = self.data_feature.get('num_nodes', 1)  # len_row * len_column
        self.feature_dim = self.data_feature.get('feature_dim', 1)  # 输入维度
        self.ext_dim = self.data_feature.get('ext_dim', 1)  # 额外数据的维度
        self.output_dim = self.data_feature.get('output_dim', 1)  # b in paper

        # config
        self.input_window = config.get('input_window', 12)  # l in paper
        self.output_window = config.get('output_window', 12)  # F in paper
        self.L = config.get('L', 3)  # num of layers in Enc-G/D / Dec-S/T
        self.d = config.get('d', 64)  # d-dimension representations
        self.n_h = config.get('n_h', 8)  # num of head in Multi-space Attention
        self.num_hid = 4 * self.d  # hidden layer size
        self.B = config.get('B', 3)  # num of layers in conv
        self.l_d = config.get('l_d', 3)
        self.r_d = config.get('r_d', 0.1)  # dropout rate

        self.dsan = DsanUse(L=self.L, d=self.d, n_h=self.n_h, row=self.len_row, column=self.len_column,
                            num_hid=self.num_hid, conv_layer=self.B,
                            input_window=self.input_window, output_window=self.output_window,
                            input_dim=self.output_dim, ext_dim=self.ext_dim, device=self.device, r_d=self.r_d)

    def generate_x(self, batch):
        """
        from batch['X'] to
        Args:
            batch: batch['X'].shape == [batch_size, input_window, row, column, feature_dim]
                    batch['y'].shape == [batch_size, output_window, row, column, output_dim]

        Returns:
            dae_inp_g: X in figure(2) shape == [batch_size, input_window, row, column, output_dim]
            dae_inp: X_d in figure(2) shape == [batch_size, input_window, row, column, L_D, L_D output_dim]
                        N_D = L_d * L_d ,L_d = 2 * l_d + 1
            dae_inp_ex: external data for TPE shape == [batch_size, input_window, N, external_dim]
            sad_inp: x in figure(2) shape == [batch_size, output_window, N, output_dim]
            sad_inp_ex: external data for TPE shape == [batch_size, input_window, N, external_dim]
            cors: for SPE,shape == [1, 1, N_d, d]
            cors_g: for SPE, shape == [1, N, d]
            y:

        """
        X = batch['X'][:, :, :, :, :self.output_dim]
        X_ext = batch['X'][:, :, :, :, self.output_dim:]
        X_shape = X.shape  # [batch_size, input_window, row, column, feature_dim]
        l_d = self.l_d

        # dae_inp_g
        dae_inp_g = torch.reshape(input=X, shape=[X_shape[0], X_shape[1], X_shape[2], X_shape[3], X_shape[4]])

        # dae_inp
        L_d = 2 * l_d + 1  # l_d: half length of the block (L_d = 2 * l_d + 1)

        dae_inp = torch.zeros(size=[X_shape[0], X_shape[1], X_shape[2], X_shape[3], L_d, L_d, X_shape[4]],
                              device=self.device)
        for i in range(X_shape[2]):
            for j in range(X_shape[3]):
                dae_inp[:, :, i, j, max(0, l_d - i):min(L_d, X_shape[2] - i + l_d),
                max(0, l_d - j):min(L_d, X_shape[3] - j + l_d), :] = \
                    X[:, :, max(0, i - l_d):min(X_shape[2], i + l_d + 1), max(0, j - l_d):min(X_shape[3], j + l_d + 1),
                    :]

        # dae_inp_ex
        dae_inp_ex = X_ext

        # sad_inp
        sad_inp = torch.reshape(input=X[:, -self.output_window:, :, :, :self.output_dim],
                                shape=[X_shape[0], -1, X_shape[2] * X_shape[3], X_shape[4]])

        # sad_inp_ex
        sad_inp_ex = X_ext[:, -self.output_window:, :, :, :]

        # cors
        cors = torch.zeros(size=[L_d, L_d, self.d], device=self.device)
        for i in range(L_d):
            for j in range(L_d):
                cors[i, j, :] = spatial_posenc(i - L_d // 2, j - L_d // 2, self.d, device=self.device)

        cors = torch.reshape(input=cors, shape=[1, 1, cors.shape[0] * cors.shape[1], cors.shape[2]])

        # cors_g
        cors_g = torch.zeros(size=[self.len_row, self.len_column, self.d], device=self.device)
        for i in range(self.len_row):
            for j in range(self.len_column):
                cors_g[i, j, :] = spatial_posenc(i - self.len_row // 2, j - self.len_column // 2, self.d,
                                                 device=self.device)
        cors_g = torch.reshape(input=cors_g, shape=[1, cors_g.shape[0] * cors_g.shape[1], cors_g.shape[2]])

        # y
        y = batch['y']

        return dae_inp_g, dae_inp, dae_inp_ex, sad_inp, sad_inp_ex, cors, cors_g, y

    def predict(self, batch):

        # generate x
        dae_inp_g, dae_inp, dae_inp_ex, sad_inp, sad_inp_ex, cors, cors_g, y = \
            self.generate_x(batch=batch)

        # generate mask
        threshold_mask_g, threshold_mask, combined_mask = create_masks(
            dae_inp_g[..., :self.output_dim], dae_inp[..., :self.output_dim], sad_inp)

        # reshape
        dae_inp = torch.reshape(input=dae_inp,
                                shape=[dae_inp.shape[0], dae_inp.shape[1], dae_inp.shape[2], dae_inp.shape[3],
                                       dae_inp.shape[4] * dae_inp.shape[5], dae_inp.shape[6]])

        res = self.dsan(
            dae_inp_g=dae_inp_g,
            dae_inp=dae_inp,
            dae_inp_ex=dae_inp_ex,
            sad_inp=sad_inp,
            sad_inp_ex=sad_inp_ex,
            cors=cors,
            cors_g=cors_g,
            threshold_mask=threshold_mask,
            threshold_mask_g=threshold_mask_g,
            look_ahead_mask=combined_mask
        )
        return res

    def calculate_loss(self, batch):
        y_true = batch['y']
        y_pred = self.predict(batch)
        y_true = self._scaler.inverse_transform(y_true[..., :self.output_dim])
        y_predicted = self._scaler.inverse_transform(y_pred[..., :self.output_dim])
        return masked_rmse_torch(y_predicted, y_true, 0)
