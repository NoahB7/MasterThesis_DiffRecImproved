import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np
import math
import random

class DNN(nn.Module):
    """
    A deep neural network for the reverse diffusion preocess.
    """
    def __init__(self, in_dims, out_dims, emb_size, time_type="cat", norm=False, dropout=0.5):
        super(DNN, self).__init__()
        self.in_dims = in_dims
        self.out_dims = out_dims
        assert out_dims[0] == in_dims[-1], "In and out dimensions must equal to each other."
        self.time_type = time_type
        self.time_emb_dim = emb_size
        self.norm = norm
        self.training = False

        self.embinput_layer = nn.Linear(in_dims[0], self.time_emb_dim)
        self.emb_layer = nn.Linear(self.time_emb_dim, self.time_emb_dim)

        self.time_emb_layer = nn.Linear(10,10)

        if self.time_type == "cat":
            in_dims_temp = [self.in_dims[0] + self.time_emb_dim] + self.in_dims[1:]
            # in_dims_temp = [self.in_dims[0] + self.time_emb_dim + 10] + self.in_dims[1:]
            # in_dims_temp = [self.in_dims[0]] + self.in_dims[1:]
        else:
            raise ValueError("Unimplemented timestep embedding type %s" % self.time_type)

        # for d_in, d_out in zip(self.in_dims[:-1], self.in_dims[1:]):
        #     print(d_in, d_out)

        # for d_in, d_out in zip(self.out_dims[:-1], self.out_dims[1:]):
        #     print(d_in, d_out)
        
        self.in_layers = nn.ModuleList([nn.Linear(d_in, d_out) \
            for d_in, d_out in zip(in_dims_temp[:-1], in_dims_temp[1:])])
        self.out_layers = nn.ModuleList([nn.Linear(d_in, d_out) \
            for d_in, d_out in zip(self.out_dims[:-1], self.out_dims[1:])])
        
        self.drop = nn.Dropout(dropout)
        self.init_weights()
    
    def init_weights(self):
        for layer in self.in_layers:
            # Xavier Initialization for weights
            size = layer.weight.size()
            fan_out = size[0]
            fan_in = size[1]
            std = np.sqrt(2.0 / (fan_in + fan_out))
            layer.weight.data.normal_(0.0, std)

            # Normal Initialization for weights
            layer.bias.data.normal_(0.0, 0.001)
        
        for layer in self.out_layers:
            # Xavier Initialization for weights
            size = layer.weight.size()
            fan_out = size[0]
            fan_in = size[1]
            std = np.sqrt(2.0 / (fan_in + fan_out))
            layer.weight.data.normal_(0.0, std)

            # Normal Initialization for weights
            layer.bias.data.normal_(0.0, 0.001)
        
        size = self.emb_layer.weight.size()
        fan_out = size[0]
        fan_in = size[1]
        std = np.sqrt(2.0 / (fan_in + fan_out))
        self.emb_layer.weight.data.normal_(0.0, std)
        self.emb_layer.bias.data.normal_(0.0, 0.001)
    
    def forward(self, x, timesteps, guidance, p_uncond):
        if self.training: # during training converting p_uncond percentage of guidances to zero gradients to replicate classifier-free guidance, shouldn not happen during inference
            guidance = guidance.cpu() # same here?
            p_uncond_indices = random.sample( list( range( 0, len(guidance) ) ), int(len(guidance)*p_uncond))
            for i in range(0,len(guidance)):
                if i in p_uncond_indices:
                    guidance[i] = torch.from_numpy(np.zeros_like(guidance[i]))
            guidance = guidance.to("cuda") # not sure why just guidance.to() doesnt work
        # print(guidance.shape)
        # time_emb = timestep_embedding(timesteps, 10).to(x.device)
        # time_emb = self.time_emb_layer(time_emb)
        emb = self.embinput_layer(guidance)
        emb = self.emb_layer(emb)
        if self.norm:
            x = F.normalize(x)
        x = self.drop(x)
        # h = x
        # h = torch.cat([x, emb, time_emb], dim=-1)
        h = torch.cat([x, emb], dim=-1)

        for i, layer in enumerate(self.in_layers):
            h = layer(h)
            h = torch.tanh(h)
        
        for i, layer in enumerate(self.out_layers):
            h = layer(h)
            if i != len(self.out_layers) - 1:
                h = torch.tanh(h)
        
        return h
# create embeddings for sequence, curernt sequence embedding size is 100, 2810+100 input size 400x2910 to 2910 x 1000 to 1000 x 1000 to 1000 x 2810 for final output


def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.

    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """

    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding

# noising and denoising on stacked vectors of input sequence?