import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential as Seq
from torchvision import models
from skimage import io, color, transform
from scipy import interpolate
import numpy as np
from tqdm import trange

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.helpers import load_pretrained
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model

from gcn_lib import Grapher, act_layer
from ssn.util import *
from graph_att import attention, construct_graph


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'patch_embed.proj', 'classifier': 'head',
        **kwargs
    }


default_cfgs = {
    'vig_224_gelu': _cfg(
        mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
    ),
    'vig_b_224_gelu': _cfg(
        crop_pct=0.95, mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
    ),
}


# feed-forward network
class FFN(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act='relu', drop_path=0.0):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Sequential(
            nn.Conv2d(in_features, hidden_features, 1, stride=1, padding=0),
            nn.BatchNorm2d(hidden_features),
        )
        self.act = act_layer(act)
        self.fc2 = nn.Sequential(
            nn.Conv2d(hidden_features, out_features, 1, stride=1, padding=0),
            nn.BatchNorm2d(out_features),
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        shortcut = x
        x = self.fc1(x)
        x = self.act(x)
        x = self.fc2(x)
        x = self.drop_path(x) + shortcut
        return x    # .reshape(B, C, N, 1)


# superpixels
class create_ssn_net(nn.Module):
    def __init__(self, num_spixels, num_iter, num_spixels_h, num_spixels_w):
        super(create_ssn_net, self).__init__()
        self.num_spixels = num_spixels
        self.num_iter = num_iter
        self.num_spixels_h = num_spixels_h
        self.num_spixels_w = num_spixels_w
        self.num_spixels = num_spixels_h * num_spixels_w

    def forward(self, x, p2sp_index, invisible, init_index, cir_index, spixel_h, spixel_w, device):
        trans_features = x
        self.num_spixels_h = spixel_h
        self.num_spixels_w = spixel_w
        self.num_spixels = spixel_h * spixel_w
        self.device = device
        # init spixel feature
        spixel_feature = SpixelFeature(trans_features, init_index, max_spixels=self.num_spixels)

        for i in range(self.num_iter):
            spixel_feature, _ = exec_iter(spixel_feature, trans_features, cir_index, p2sp_index,
                                                              invisible, self.num_spixels_h, self.num_spixels_w, self.device)

        final_pixel_assoc = compute_assignments(spixel_feature, trans_features, p2sp_index, invisible, device)  # (1,9,224,224)

        new_spixel_feat = SpixelFeature2(x, final_pixel_assoc, cir_index, invisible,
                                             self.num_spixels_h, self.num_spixels_w)
        new_spix_indices = compute_final_spixel_labels(final_pixel_assoc, p2sp_index,
                                                           self.num_spixels_h, self.num_spixels_w)
        recon_feat2 = Semar(new_spixel_feat, new_spix_indices)

        # recon_feat2, new_spix_indices, final_pixel_assoc
        return recon_feat2, new_spix_indices, final_pixel_assoc


def transform_and_get_spixel_init(max_spixels, out_size):

    out_height = out_size[0]
    out_width = out_size[1]

    spixel_init, feat_spixel_initmap, k_w, k_h = \
        get_spixel_init(max_spixels, out_width, out_height)
    spixel_init = spixel_init[None, None, :, :]
    feat_spixel_initmap = feat_spixel_initmap[None, None, :, :]

    return spixel_init, feat_spixel_initmap, k_h, k_w


def get_spixel_init(num_spixels, img_width, img_height):
    """
    :return each pixel belongs to which pixel
    """

    k = num_spixels
    k_w = int(np.floor(np.sqrt(k * img_width / img_height)))
    k_h = int(np.floor(np.sqrt(k * img_height / img_width)))

    spixel_height = img_height / (1. * k_h)
    spixel_width = img_width / (1. * k_w)

    h_coords = np.arange(-spixel_height / 2. - 1, img_height + spixel_height - 1,
                         spixel_height)
    w_coords = np.arange(-spixel_width / 2. - 1, img_width + spixel_width - 1,
                         spixel_width)
    spix_values = np.int32(np.arange(0, k_w * k_h).reshape((k_h, k_w)))
    spix_values = np.pad(spix_values, 1, 'symmetric')
    f = interpolate.RegularGridInterpolator((h_coords, w_coords), spix_values, method='nearest')

    all_h_coords = np.arange(0, img_height, 1)
    all_w_coords = np.arange(0, img_width, 1)
    all_grid = np.array(np.meshgrid(all_h_coords, all_w_coords, indexing = 'ij'))
    all_points = np.reshape(all_grid, (2, img_width * img_height)).transpose()

    spixel_initmap = f(all_points).reshape((img_height,img_width))

    feat_spixel_initmap = spixel_initmap
    return [spixel_initmap, feat_spixel_initmap, k_w, k_h]


def convert_index(num_spixel_w=10,  max_spixels=100, feat_spixel_init=None):
    '''
    :param num_spixel_w: the number of spixels of an row
    :param max_spixels: the number of spixels
    :param feat_spixel_init:  1*1*H*W each pixel with corresponding spixel ids
    :return:
    '''
    if feat_spixel_init is not None:
        length = []
        ind_x = []
        ind_y = []
        feat_spixel_init = feat_spixel_init[0, 0]
        for i in range(max_spixels):
            id_y, id_x = np.where(feat_spixel_init==i)
            l = len(id_y)
            ind_y.extend(id_y.tolist())
            ind_x.extend(id_x.tolist())
            length.append(l)
        length = np.array(length)
        init_x = np.array(ind_x)
        init_y = np.array(ind_y)
        init_cum = np.cumsum(length)

        p2sp_index_, invisible = Passoc_Nspixel(feat_spixel_init, num_spixel_w, max_spixels)  # H*W*9, H*W*9
        length = []
        ind_x = []
        ind_y = []
        ind_z = []
        for i in range(max_spixels):
            id_y, id_x, id_z = np.where(p2sp_index_ == i)
            l = len(id_y)
            ind_y.extend(id_y)
            ind_x.extend(id_x)
            ind_z.extend(id_z)
            length.append(l)
        cir_x = np.array(ind_x)
        cir_y = np.array(ind_y)
        cir_z = np.array(ind_z)
        cir_cum = np.cumsum(length)

        return [init_x, init_y, init_cum], [cir_x, cir_y, cir_z, cir_cum], p2sp_index_, invisible


class Stem(nn.Module):
    """ Image to Visual Embedding
    Overlap: https://arxiv.org/pdf/2106.13797.pdf
    """
    def __init__(self, img_size=224, in_dim=3, out_dim=768, act='relu'):
        super().__init__()        
        self.convs = nn.Sequential(
            nn.Conv2d(in_dim, out_dim//2, 3, stride=2, padding=1),
            nn.BatchNorm2d(out_dim//2),
            act_layer(act),
            nn.Conv2d(out_dim//2, out_dim, 3, stride=2, padding=1),
            nn.BatchNorm2d(out_dim),
            act_layer(act),
            nn.Conv2d(out_dim, out_dim, 3, stride=1, padding=1),
            nn.BatchNorm2d(out_dim),
        )

    def forward(self, x):
        x = self.convs(x)
        return x


class Downsample(nn.Module):
    """ Convolution-based downsample
    """
    def __init__(self, in_dim=3, out_dim=768):
        super().__init__()        
        self.conv = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, 3, stride=2, padding=1),
            nn.BatchNorm2d(out_dim),
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class DeepGCN(torch.nn.Module):
    def __init__(self, opt):
        super(DeepGCN, self).__init__()
        print(opt)
        k = opt.k
        act = opt.act
        norm = opt.norm
        bias = opt.bias
        epsilon = opt.epsilon
        stochastic = opt.use_stochastic
        conv = opt.conv
        emb_dims = opt.emb_dims
        drop_path = opt.drop_path
        
        blocks = opt.blocks
        self.n_blocks = sum(blocks)
        channels = opt.channels
        reduce_ratios = [4, 2, 1, 1]
        dpr = [x.item() for x in torch.linspace(0, drop_path, self.n_blocks)]  # stochastic depth decay rule 0
        num_knn = [int(x.item()) for x in torch.linspace(k, k, self.n_blocks)]  # number of knn's k 9
        max_dilation = 49 // max(num_knn)

        ####################################### superpixels ###################################################
        self.make_clusters = create_ssn_net(num_spixels=196, num_iter=5, num_spixels_h=16, num_spixels_w=16)
        self.device = torch.device('cuda')
        self.out_spixel_init, self.feat_spixel_init, self.spixels_h, self.spixels_w = \
            transform_and_get_spixel_init(196, [224, 224])
        self.init, self.cir, self.p2sp_index_, self.invisible = convert_index(self.spixels_w,
                                                                              self.spixels_w * self.spixels_h,
                                                                              self.feat_spixel_init)

        self.invisible = self.invisible.astype(np.float)
        ####################################### superpixels ###################################################

        self.stem = Stem(out_dim=channels[0], act=act)
        self.pos_embed = nn.Parameter(torch.zeros(1, channels[0], 224//4, 224//4))
        # self.lsh = LSHSelfAttention(dim=80, heads=8, bucket_size=32, n_hashes=8, causal=False)
        HW = 224 // 4 * 224 // 4

        self.backbone = nn.ModuleList([])
        idx = 0
        for i in range(len(blocks)):#4
            if i > 0:
                self.backbone.append(Downsample(channels[i-1], channels[i]))
                HW = HW // 4
            for j in range(blocks[i]):#2,2,6,2
                self.backbone += [
                    Seq(Grapher(channels[i], num_knn[idx], min(idx // 4 + 1, max_dilation), conv, act, norm,
                                    bias, stochastic, epsilon, reduce_ratios[i], n=HW, drop_path=dpr[idx],
                                    relative_pos=True),
                          FFN(channels[i], channels[i] * 4, act=act, drop_path=dpr[idx])
                         )] # blocks and channel
                idx += 1

        self.backbone = Seq(*self.backbone)

        self.prediction = Seq(nn.Conv2d(channels[-1], 1024, 1, bias=True),
                              nn.BatchNorm2d(1024),
                              act_layer(act),
                              nn.Dropout(opt.dropout),
                              nn.Conv2d(1024, opt.n_classes, 1, bias=True))
        self.model_init()

    def model_init(self):
        for m in self.modules():
            if isinstance(m, torch.nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
                m.weight.requires_grad = True
                if m.bias is not None:
                    m.bias.data.zero_()
                    m.bias.requires_grad = True

    def forward(self, inputs):
        # superpixel
        # superpixel_map = self.make_clusters(inputs)
        inputs = inputs.to(self.device)
        recon_feat2, new_spix_indices = self.make_clusters(inputs, self.p2sp_index_, self.invisible, self.init, self.cir,
                                                           self.spixels_h, self.spixels_w, self.device)

        # attention
        x = self.stem(recon_feat2) + self.pos_embed
        B, C, H, W = x.shape  # (1,80,56,56)

        # DeepGCN
        for i in range(len(self.backbone)):
            x = self.backbone[i](x)

        x = F.adaptive_avg_pool2d(x, 1)
        return self.prediction(x).squeeze(-1).squeeze(-1)
