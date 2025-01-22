import copy
import math
import torch
from torch import nn
from torch.nn import functional as F
import sys
from utils import commons
from modules import modules_v9_new as modules_v9
from torch.nn import Conv1d, ConvTranspose1d, AvgPool1d, Conv2d
from torch.nn.utils import weight_norm, remove_weight_norm, spectral_norm

from utils.commons import init_weights, get_padding


# class ResidualCouplingBlock(nn.Module):
#   '''
#   Normalizing Flow
# '''
#   def __init__(self,
#       channels,
#       hidden_channels,
#       kernel_size,
#       dilation_rate,
#       n_layers,
#       n_flows=4,
#       gin_channels=0):
#     super().__init__()
#     self.channels = channels
#     self.hidden_channels = hidden_channels
#     self.kernel_size = kernel_size
#     self.dilation_rate = dilation_rate
#     self.n_layers = n_layers
#     self.n_flows = n_flows
#     self.gin_channels = gin_channels

#     self.flows = nn.ModuleList()
#     for i in range(n_flows):
#       self.flows.append(modules_v9.ResidualCouplingLayer(channels, hidden_channels, kernel_size, dilation_rate, n_layers, gin_channels=gin_channels, mean_only=True))
#       self.flows.append(modules_v9.Flip())

#   def forward(self, x, x_mask, g=None, reverse=False):
#     if not reverse:
#       for flow in self.flows:
#         x, _ = flow(x, x_mask, g=g, reverse=reverse)
#     else:
#       for flow in reversed(self.flows):
#         x = flow(x, x_mask, g=g, reverse=reverse)
#     return x


# class Encoder(nn.Module):
#   def __init__(self,
#       in_channels,
#       out_channels,
#       hidden_channels,
#       kernel_size,
#       dilation_rate,
#       n_layers,
#       gin_channels=0, vq_codebook_size=None):
#     super().__init__()
#     self.in_channels = in_channels
#     self.out_channels = out_channels
#     self.hidden_channels = hidden_channels
#     self.kernel_size = kernel_size
#     self.dilation_rate = dilation_rate
#     self.n_layers = n_layers
#     self.gin_channels = gin_channels
#     self.vq_codebook_size = vq_codebook_size

#     self.pre = nn.Conv1d(in_channels, hidden_channels, 1)
#     self.enc = modules_v9.WN(hidden_channels, kernel_size, dilation_rate, n_layers, gin_channels=gin_channels)
#     self.proj = nn.Conv1d(hidden_channels, out_channels * 2, 1)
#     #Vector Quantization
#     if vq_codebook_size != None:
#         self.codebook = modules_v9.VQEmbeddingEMA(vq_codebook_size, hidden_channels)

#   def forward(self, x, x_lengths, g=None):
#     '''
#     z : (N, out_channels, T)
#     '''
#     x_mask = torch.unsqueeze(commons.sequence_mask(x_lengths, x.size(2)), 1).to(x.dtype)
#     x = self.pre(x) * x_mask
#     x = self.enc(x, x_mask, g=g)    # Bottleneck Extractor

#     if self.vq_codebook_size == None:
#         return x
#     else:
#         x, x_quan, perplexity = self.codebook(x)
#         return x, x_quan, perplexity 
    


class Generator(torch.nn.Module):
    def __init__(self, initial_channel, resblock, resblock_kernel_sizes, resblock_dilation_sizes, upsample_rates, upsample_initial_channel, upsample_kernel_sizes, gin_channels=0):
        super(Generator, self).__init__()
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)

        self.conv_pre_1 = Conv1d(initial_channel, upsample_initial_channel-8, 3, 1, padding='same')
        
        resblock = modules_v9.ResBlock1 if resblock == '1' else modules_v9.ResBlock2

        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            self.ups.append(weight_norm(
                ConvTranspose1d(upsample_initial_channel//(2**i), upsample_initial_channel//(2**(i+1)),
                                k, u, padding=(k-u)//2)))

        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = upsample_initial_channel//(2**(i+1))
            for j, (k, d) in enumerate(zip(resblock_kernel_sizes, resblock_dilation_sizes)):
                self.resblocks.append(resblock(ch, k, d))

        self.conv_post = Conv1d(ch, 1, 7, 1, padding=3, bias=False)
        self.ups.apply(init_weights)

        if gin_channels != 0:
            self.cond = nn.Conv1d(gin_channels, upsample_initial_channel, 1)
            self.cond_res = nn.Conv1d(gin_channels, 8, 1)
            
    def forward(self, x, g=None, res=None):
        x = self.conv_pre_1(x)
        res = self.cond_res(res)

        if g is not None:
            spk_ = self.cond(g)

        x = torch.cat((x, res), dim=1)
        x = x + spk_
        
        for i in range(self.num_upsamples):
            x = F.leaky_relu(x, modules_v9.LRELU_SLOPE)
            x = self.ups[i](x)
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i*self.num_kernels+j](x)
                else:
                    xs += self.resblocks[i*self.num_kernels+j](x)
            x = xs / self.num_kernels
        x = F.leaky_relu(x)
        x = self.conv_post(x)
        x = torch.tanh(x)

        return x

    def remove_weight_norm(self):
        print('Removing weight norm...')
        for l in self.ups:
            remove_weight_norm(l)
        for l in self.resblocks:
            l.remove_weight_norm()


class DiscriminatorP(torch.nn.Module):
    def __init__(self, period, kernel_size=5, stride=3, use_spectral_norm=False):
        super(DiscriminatorP, self).__init__()
        self.period = period
        self.use_spectral_norm = use_spectral_norm
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm
        self.convs = nn.ModuleList([
            norm_f(Conv2d(1, 32, (kernel_size, 1), (stride, 1), padding=(get_padding(kernel_size, 1), 0))),
            norm_f(Conv2d(32, 128, (kernel_size, 1), (stride, 1), padding=(get_padding(kernel_size, 1), 0))),
            norm_f(Conv2d(128, 512, (kernel_size, 1), (stride, 1), padding=(get_padding(kernel_size, 1), 0))),
            norm_f(Conv2d(512, 1024, (kernel_size, 1), (stride, 1), padding=(get_padding(kernel_size, 1), 0))),
            norm_f(Conv2d(1024, 1024, (kernel_size, 1), 1, padding=(get_padding(kernel_size, 1), 0))),
        ])
        self.conv_post = norm_f(Conv2d(1024, 1, (3, 1), 1, padding=(1, 0)))

    def forward(self, x):
        fmap = []

        # 1d to 2d
        b, c, t = x.shape
        if t % self.period != 0: # pad first
            n_pad = self.period - (t % self.period)
            x = F.pad(x, (0, n_pad), "reflect")
            t = t + n_pad
        x = x.view(b, c, t // self.period, self.period)

        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, modules_v9.LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap


class DiscriminatorS(torch.nn.Module):
    def __init__(self, use_spectral_norm=False):
        super(DiscriminatorS, self).__init__()
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm
        self.convs = nn.ModuleList([
            norm_f(Conv1d(1, 16, 15, 1, padding=7)),
            norm_f(Conv1d(16, 64, 41, 4, groups=4, padding=20)),
            norm_f(Conv1d(64, 256, 41, 4, groups=16, padding=20)),
            norm_f(Conv1d(256, 1024, 41, 4, groups=64, padding=20)),
            norm_f(Conv1d(1024, 1024, 41, 4, groups=256, padding=20)),
            norm_f(Conv1d(1024, 1024, 5, 1, padding=2)),
        ])
        self.conv_post = norm_f(Conv1d(1024, 1, 3, 1, padding=1))

    def forward(self, x):
        fmap = []

        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, modules_v9.LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap


class MultiPeriodDiscriminator(torch.nn.Module):
    def __init__(self, use_spectral_norm=False):
        super(MultiPeriodDiscriminator, self).__init__()
        periods = [2,3,5,7,11]

        discs = [DiscriminatorS(use_spectral_norm=use_spectral_norm)]
        discs = discs + [DiscriminatorP(i, use_spectral_norm=use_spectral_norm) for i in periods]
        self.discriminators = nn.ModuleList(discs)

    def forward(self, y, y_hat):
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        for i, d in enumerate(self.discriminators):
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            y_d_gs.append(y_d_g)
            fmap_rs.append(fmap_r)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs
        

class SynthesizerTrn(nn.Module):
  """
  Synthesizer for Training
  """

  def __init__(self, 
    spec_channels,
    segment_size,
    inter_channels,
    hidden_channels,
    filter_channels,
    n_heads,
    n_layers,
    kernel_size,
    p_dropout,
    resblock, 
    resblock_kernel_sizes, 
    resblock_dilation_sizes, 
    upsample_rates, 
    upsample_initial_channel, 
    upsample_kernel_sizes,
    gin_channels,
    ssl_dim,
    use_spk,
    codebook_path,
    **kwargs):

    super().__init__()
    self.spec_channels = spec_channels
    self.inter_channels = inter_channels
    self.hidden_channels = hidden_channels
    self.filter_channels = filter_channels
    self.n_heads = n_heads
    self.n_layers = n_layers
    self.kernel_size = kernel_size
    self.p_dropout = p_dropout
    self.resblock = resblock
    self.resblock_kernel_sizes = resblock_kernel_sizes
    self.resblock_dilation_sizes = resblock_dilation_sizes
    self.upsample_rates = upsample_rates
    self.upsample_initial_channel = upsample_initial_channel
    self.upsample_kernel_sizes = upsample_kernel_sizes
    self.segment_size = segment_size
    self.gin_channels = gin_channels
    self.ssl_dim = ssl_dim
    self.use_spk = use_spk
    self.codebook_path = codebook_path

    self.dec = Generator(inter_channels, resblock, resblock_kernel_sizes, resblock_dilation_sizes, upsample_rates, upsample_initial_channel, upsample_kernel_sizes, gin_channels=gin_channels)

    #Vector Quantization
    if self.codebook_path != None:
        print(f'--- Codebook Path :: {self.codebook_path}')
        codebook_custom = torch.load(self.codebook_path)
        self.codebook = modules_v9.VQEmbeddingEMA(codebook_custom.size(0), hidden_channels, codebook_custom=codebook_custom)

    
  def forward(self, c, c_lengths=None):
    if c_lengths == None:
      c_lengths = (torch.ones(c.size(0)) * c.size(-1)).to(c.device)
    
    # Quantization
    quantized, commitment_loss, perplexity = self.codebook(c)
    if quantized.size(1) != c.size(1):
        quantized = quantized.permute(0, 2, 1)
        
    # speaker emb
    speaker_emb = c - quantized
    z = quantized # d: (B, D, T)
    z_slice, ids_slice = commons.rand_slice_segments(z, c_lengths, self.segment_size)
    spk_slice = commons.slice_segments(speaker_emb, ids_slice, self.segment_size)
    
    spk_slice_avg = torch.mean(spk_slice, dim=-1, keepdim=True)
    res_slice = spk_slice - spk_slice_avg
    o = self.dec(z_slice, g=spk_slice_avg, res=res_slice)
    
    return o, ids_slice, (commitment_loss, perplexity)

  def infer(self, c, c_lengths=None):
    if c_lengths == None:
      c_lengths = (torch.ones(c.size(0)) * c.size(-1)).to(c.device)

    quantized, commitment_loss, perplexity = self.codebook(c)
    fig = None
    if quantized.size(1) != c.size(1):
        quantized = quantized.permute(0, 2, 1)
        
    # speaker emb
    speaker_emb = c - quantized
    spk_emb_avg = torch.mean(speaker_emb, dim=-1, keepdim=True)
    residual_emb = speaker_emb - spk_emb_avg
    z = quantized
    o = self.dec(z, g=spk_emb_avg, res=residual_emb)
    
    return o, fig

  def convert(self, src_c, tgt_c, c_lengths=None):
    quantized_src, commitment_loss, perplexity = self.codebook(src_c)
    quantized_tgt, commitment_loss, perplexity = self.codebook(tgt_c)
    if quantized_src.size(1) != src_c.size(1):
        quantized_src = quantized_src.permute(0, 2, 1)
        quantized_tgt = quantized_tgt.permute(0, 2, 1)
        
    speaker_emb_tgt = tgt_c - quantized_tgt
    speaker_emb_src = src_c - quantized_src
    speaker_emb_avg_tgt = torch.mean(speaker_emb_tgt, dim=-1, keepdim=True)
    speaker_emb_avg_src = torch.mean(speaker_emb_src, dim=-1, keepdim=True)
    
    residual_emb_src = speaker_emb_src - speaker_emb_avg_src
    z_src = quantized_src
    o = self.dec(z_src, g=speaker_emb_avg_tgt, res=residual_emb_src)
    
    return o



