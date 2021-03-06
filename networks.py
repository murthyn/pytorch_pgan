# Source: https://github.com/tkarras/progressive_growing_of_gans/

import numpy as np

import torch.nn as nn
import torch.nn.functional as F
import torch


#----------------------------------------------------------------------------

def lerp(a, b, t): return a + (b - a) * t
def lerp_clip(a, b, t): return a + (b - a) * torch.clamp(t, 0.0, 1.0)
def cset(cur_lambda, new_cond, new_lambda): return lambda: torch.where(new_cond, new_lambda, cur_lambda)

#----------------------------------------------------------------------------
# Get/create weight tensor for a convolutional or fully-connected layer.

# TODO

#----------------------------------------------------------------------------
# Pixelwise feature vector normalization.

def pixel_norm(x, epsilon=1e-8):
    return x * torch.rsqrt(torch.mean(torch.pow(x, 2), dim=1, keepdim=True) + epsilon)

#----------------------------------------------------------------------------
# Fully-connected layer.

class Dense(nn.Module):
    def __init__(self, gain=np.sqrt(2), use_wscale=False):
        self.gain = gain
        self.use_wscale = use_wscale


    (x, fmaps, gain=np.sqrt(2), use_wscale=False):
    if len(x.shape) > 2:
        x = tf.reshape(x, [-1, np.prod([d.value for d in x.shape[1:]])])
    w = get_weight([x.shape[1].value, fmaps], gain=gain, use_wscale=use_wscale)
    w = tf.cast(w, x.dtype)
    return tf.matmul(x, w)

#----------------------------------------------------------------------------
# Generator from paper

class Generator(nn.Module):
    def __init__(self,
                 num_channels = 1,          # Number of output color channels. Overridden based on dataset.
                 resolution = 32,           # Output resolution. Overridden based on dataset.
                 label_size = 0,            # Dimensionality of the labels, 0 if no labels. Overridden based on dataset.
                 fmap_base=8192,            # Overall multiplier for the number of feature maps.
                 fmap_decay=1.0,            # log2 feature map reduction when doubling the resolution.
                 fmap_max=512,              # Maximum number of feature maps in any layer.
                 latent_size=None,          # Dimensionality of the latent vectors. None = min(fmap_base, fmap_max).
                 normalize_latents=True,    # Normalize latent vectors before feeding them to the network?
                 use_wscale=True,           # Enable equalized learning rate?
                 use_pixelnorm=True,        # Enable pixelwise feature vector normalization?
                 pixelnorm_epsilon=1e-8,    # Constant epsilon for pixelwise feature vector normalization.
                 use_leakyrelu=True,        # True = leaky ReLU, False = ReLU.
                 dtype='float32',           # Data type to use for activations and outputs.
                 fused_scale=True,          # True = use fused upscale2d + conv2d, False = separate upscale2d layers.
                 structure=None,            # 'linear' = human-readable, 'recursive' = efficient, None = select automatically.
                 is_template_graph=False,   # True = template graph constructed by the Network class, False = actual evaluation.
                 **kwargs):                 # Ignore unrecognized keyword args.

        super(Generator, self).__init__()

        self.resolution_log2 = int(np.log2(resolution))
        assert resolution == 2 ** self.resolution_log2 and resolution >= 4

        def nf(stage): return min(int(fmap_base / (2.0 ** (stage * fmap_decay))), fmap_max)
        def PN(x): return pixel_norm(x, epsilon=pixelnorm_epsilon) if use_pixelnorm else x

        if latent_size is None: latent_size = nf(0)
        if structure is None: structure = 'linear' if is_template_graph else 'recursive'
        act = nn.leakyReLU if use_leakyrelu else nn.ReLU

        self.num_channels = num_channels
        self.img_size = 0
        self.img_shape = (self.num_channels, self.img_size, self.img_size)

        def block(x, res):
            if res == 2:  # 4x4
                return
            else:  # 8x8 and up
                return
            # TODO: define block

        self.model = None # TODO:

    def forward(self,
                latents_in,                 # First input: Latent vectors [minibatch, latent_size].
                labels_in,                  # Second input: Labels [minibatch, label_size].
                z):
        img = self.model(z)
        img = img.view(img.size(0), self.img_shape)
        return img

