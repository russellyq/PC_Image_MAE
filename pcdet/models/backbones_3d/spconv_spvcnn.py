import torch
import spconv.pytorch as spconv
import torch.nn as nn
import torch.nn.functional as F
import sptr

def base_block(in_channels, out_channels, indice_key):
    conv = spconv.SparseSequential(
        spconv.SubMConv3d(in_channels, out_channels, 1, indice_key=indice_key, bias=False),
        nn.BatchNorm1d(out_channels),
        nn.ReLU()
    )
    return conv

class SparseBasicBlock(spconv.SparseModule):
    def __init__(self, in_channels, out_channels, indice_key):
        super(SparseBasicBlock, self).__init__()
        self.layers_in = spconv.SparseSequential(
            spconv.SubMConv3d(in_channels, out_channels, 1, indice_key=indice_key, bias=False),
            nn.BatchNorm1d(out_channels),
        )
        self.layers = spconv.SparseSequential(
            spconv.SubMConv3d(in_channels, out_channels, 3, indice_key=indice_key, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.LeakyReLU(0.1),
            spconv.SubMConv3d(out_channels, out_channels, 3, indice_key=indice_key, bias=False),
            nn.BatchNorm1d(out_channels),
        )

    def forward(self, x):
        identity = self.layers_in(x)
        output = self.layers(x)
        return output.replace_feature(F.leaky_relu(output.features + identity.features, 0.1))



class SparseAttenBlock(spconv.SparseModule):
    def __init__(self, in_channels, out_channels, indice_key):
        super(SparseAttenBlock, self).__init__()
        
        self.layer = SparseBasicBlock(in_channels, out_channels, indice_key)

        head_dim = 16
        num_heads = out_channels // head_dim
        self.atten = sptr.VarLengthMultiheadSA(
        out_channels, 
        num_heads, 
        indice_key='sptr_0', 
        window_size=6, 
        shift_win=False,
        )

    def forward(self, x):

        out = self.layer(x)

        feats, indices, spatial_shape = out.features, out.indices, out.spatial_shape
        # feats: [N, C], indices: [N, 4] with batch indices in the 0-th column
        input_tensor = sptr.SparseTrTensor(feats, indices, spatial_shape=spatial_shape, batch_size=None)
        output_tensor = self.atten(input_tensor)

        # Extract features from output tensor
        output_feats = output_tensor.query_feats

        return out.replace_feature(F.leaky_relu(output_feats, 0.1))
