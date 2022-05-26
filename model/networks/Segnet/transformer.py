import torch
import math
import torch.nn.functional
from torch import nn

class TransformerEncoder2D(nn.Module):

    def __init__(self, in_channels, d_model, nhead=8, num_layers=6):
        super().__init__()
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.pe = None
        
        self.preconv = nn.Conv2d(in_channels, d_model, (1, 1)) if in_channels != d_model else nn.Identity()
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        

    def forward(self, src):
        # reduce dimension if necessary
        compressed_src = self.preconv(src)
        bs, c, h, w = compressed_src.shape
        # # add positional encoding
        # if self.pe is None:
        #     self.pe = positionalencoding2d(c, h, w)[None].to(
        #         compressed_src.device)  # 1xCxHxW
        # output = compressed_src + self.pe
        # transformer forward
        output = compressed_src.flatten(2).permute(2, 0, 1)  # HWxBxC
        output = self.transformer_encoder(output)
        # reshape to 2D
        output = output.permute(1, 2, 0).reshape(bs, c, h, w) # BxCxHxW
        return output


def positionalencoding2d(d_model, height, width):
    """
    :param d_model: dimension of the model
    :param height: height of the positions
    :param width: width of the positions
    :return: d_model*height*width position matrix
    """
    if d_model % 4 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with "
                         "odd dimension (got dim={:d})".format(d_model))
    pe = torch.zeros(d_model, height, width)
    # Each dimension use half of d_model
    d_model = int(d_model / 2)
    div_term = torch.exp(torch.arange(0., d_model, 2) *
                         -(math.log(10000.0) / d_model))
    pos_w = torch.arange(0., width).unsqueeze(1)
    pos_h = torch.arange(0., height).unsqueeze(1)
    pe[0:d_model:2, :, :] = torch.sin(
        pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    pe[1:d_model:2, :, :] = torch.cos(
        pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, height, 1)
    pe[d_model::2, :, :] = torch.sin(
        pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)
    pe[d_model + 1::2, :,
        :] = torch.cos(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, width)

    return pe
