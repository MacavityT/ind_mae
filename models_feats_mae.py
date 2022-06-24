import torch
import torch.nn as nn

from functools import partial

from timm.models.vision_transformer import PatchEmbed, Block
from util.pos_embed import get_2d_sincos_pos_embed

from ind_utils import RandomMasking
from ind_utils import HOGTarget


class MaskedFeatsAutoencoderViT(nn.Module):
    """ Masked Autoencoder with VisionTransformer backbone
    """

    def __init__(self,
                 img_size=224,
                 patch_size=16,
                 in_chans=3,
                 embed_dim=1024,
                 depth=24,
                 num_heads=16,
                 decoder_embed_dim=512,
                 mlp_ratio=4.,
                 masking=None,
                 feats=None,
                 norm_layer=nn.LayerNorm,
                 norm_pix_loss=False):
        super().__init__()

        # --------------------------------------------------------------------------
        # encoder specifics
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans,
                                      embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, embed_dim),
            requires_grad=False)  # fixed sin-cos embedding

        self.blocks = nn.ModuleList([
            Block(embed_dim,
                  num_heads,
                  mlp_ratio,
                  qkv_bias=True,
                  norm_layer=norm_layer) for i in range(depth)
        ])
        self.norm = norm_layer(embed_dim)
        # --------------------------------------------------------------------------

        self.feats = HOGTarget()
        self.masking = RandomMasking()
        self.norm_pix_loss = norm_pix_loss

        # --------------------------------------------------------------------------
        # decoder specifics
        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        self.decoder_pred = nn.Linear(decoder_embed_dim,
                                      patch_size**2,
                                      bias=True)  # decoder to patch
        # --------------------------------------------------------------------------

        self.initialize_weights()

    def initialize_weights(self):
        # initialization
        # initialize (and freeze) pos_embed by sin-cos embedding
        pos_embed = get_2d_sincos_pos_embed(
            self.pos_embed.shape[-1],
            int(self.patch_embed.num_patches**.5),
            cls_token=True)
        self.pos_embed.data.copy_(
            torch.from_numpy(pos_embed).float().unsqueeze(0))

        # initialize patch_embed like nn.Linear (instead of nn.Conv2d)
        w = self.patch_embed.proj.weight.data
        torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.cls_token, std=.02)
        torch.nn.init.normal_(self.mask_token, std=.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs):
        """
        imgs: (N, 1, H, W)
        x: (N, L, patch_size**2)
        """
        p = self.patch_embed.patch_size[0]
        assert imgs.shape[2] == imgs.shape[3] and imgs.shape[2] % p == 0

        h = w = imgs.shape[2] // p
        x = imgs.reshape(shape=(imgs.shape[0], 1, h, p, w, p))
        x = torch.einsum('nchpwq->nhwpqc', x)
        x = x.reshape(shape=(imgs.shape[0], h * w, p**2))
        return x

    def forward_encoder(self, x, mask_ratio):
        # embed patches
        x = self.patch_embed(x)

        # add pos embed w/o cls token
        x = x + self.pos_embed[:, 1:, :]

        # masking: length -> length * mask_ratio
        x, mask, ids_restore = self.masking(x, mask_ratio)

        # append cls token
        cls_token = self.cls_token + self.pos_embed[:, :1, :]
        cls_tokens = cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        return x, mask, ids_restore

    def forward_decoder(self, x, ids_restore):
        x = self.decoder_embed(x)
        # append mask tokens to sequence
        mask_tokens = self.mask_token.repeat(
            x.shape[0], ids_restore.shape[1] + 1 - x.shape[1], 1)
        x_ = torch.cat([x[:, 1:, :], mask_tokens], dim=1)  # no cls token
        x_ = torch.gather(x_,
                          dim=1,
                          index=ids_restore.unsqueeze(-1).repeat(
                              1, 1, x.shape[2]))  # unshuffle

        x = torch.cat([x[:, :1, :], x_], dim=1)  # append cls token

        # predictor projection
        x = self.decoder_pred(x)
        # remove cls token
        x = x[:, 1:, :]
        return x

    def forward_loss(self, imgs, pred, mask):
        """
        imgs: [N, 3, H, W]
        pred: [N, L, p*p]
        mask: [N, L], 0 is keep, 1 is remove, 
        """
        target = self.feats(imgs)  # [N, H, W]
        target = self.patchify(target)

        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.e-6)**.5

        loss = (pred - target)**2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss

    def forward(self, imgs, mask_ratio=0.75):
        latent, mask, ids_restore = self.forward_encoder(imgs, mask_ratio)
        pred = self.forward_decoder(latent, ids_restore)  # [N, L, p*p*3]
        loss = self.forward_loss(imgs, pred, mask)
        return loss, pred, mask


def mae_vit_base_patch16_dec512d8b(**kwargs):
    model = MaskedFeatsAutoencoderViT(patch_size=16,
                                      embed_dim=768,
                                      depth=12,
                                      num_heads=12,
                                      decoder_embed_dim=512,
                                      mlp_ratio=4,
                                      norm_layer=partial(nn.LayerNorm,
                                                         eps=1e-6),
                                      **kwargs)
    return model


def mae_vit_large_patch16_dec512d8b(**kwargs):
    model = MaskedFeatsAutoencoderViT(patch_size=16,
                                      embed_dim=1024,
                                      depth=24,
                                      num_heads=16,
                                      decoder_embed_dim=512,
                                      mlp_ratio=4,
                                      norm_layer=partial(nn.LayerNorm,
                                                         eps=1e-6),
                                      **kwargs)
    return model


def mae_vit_huge_patch14_dec512d8b(**kwargs):
    model = MaskedFeatsAutoencoderViT(patch_size=14,
                                      embed_dim=1280,
                                      depth=32,
                                      num_heads=16,
                                      decoder_embed_dim=512,
                                      mlp_ratio=4,
                                      norm_layer=partial(nn.LayerNorm,
                                                         eps=1e-6),
                                      **kwargs)
    return model


BASE_PARAM = dict(patch_size=16,
                  embed_dim=768,
                  depth=12,
                  num_heads=12,
                  decoder_embed_dim=512,
                  mlp_ratio=4,
                  norm_layer=partial(nn.LayerNorm, eps=1e-6))

LARGE_PARAM = dict(patch_size=16,
                   embed_dim=1024,
                   depth=24,
                   num_heads=16,
                   decoder_embed_dim=512,
                   mlp_ratio=4,
                   norm_layer=partial(nn.LayerNorm, eps=1e-6))

HUGE_PARAM = dict(patch_size=14,
                  embed_dim=1280,
                  depth=32,
                  num_heads=16,
                  decoder_embed_dim=512,
                  mlp_ratio=4,
                  norm_layer=partial(nn.LayerNorm, eps=1e-6))


# ind vit base models
def mae_vit_base_img512_patch16_dec512d8b(**kwargs):
    kwargs.update(BASE_PARAM)
    kwargs['img_size'] = 512
    model = MaskedFeatsAutoencoderViT(**kwargs)
    return model


def mae_vit_base_img640_patch16_dec512d8b(**kwargs):
    kwargs.update(BASE_PARAM)
    kwargs['img_size'] = 640
    model = MaskedFeatsAutoencoderViT(**kwargs)
    return model


def mae_vit_base_img768_patch16_dec512d8b(**kwargs):
    kwargs.update(BASE_PARAM)
    kwargs['img_size'] = 768
    model = MaskedFeatsAutoencoderViT(**kwargs)
    return model


def mae_vit_base_img1024_patch16_dec512d8b(**kwargs):
    kwargs.update(BASE_PARAM)
    kwargs['img_size'] = 1024
    model = MaskedFeatsAutoencoderViT(**kwargs)
    return model


# set recommended archs
mae_vit_base_patch16 = mae_vit_base_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
mae_vit_large_patch16 = mae_vit_large_patch16_dec512d8b  # decoder: 512 dim, 8 blocks
mae_vit_huge_patch14 = mae_vit_huge_patch14_dec512d8b  # decoder: 512 dim, 8 blocks

# set ind private archs
mae_vit_base_img512_patch16 = mae_vit_base_img512_patch16_dec512d8b
mae_vit_base_img640_patch16 = mae_vit_base_img640_patch16_dec512d8b
mae_vit_base_img768_patch16 = mae_vit_base_img768_patch16_dec512d8b
mae_vit_base_img1024_patch16 = mae_vit_base_img1024_patch16_dec512d8b
