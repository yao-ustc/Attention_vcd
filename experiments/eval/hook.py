import time
import numpy as np
import torch
from PIL import Image
import glob

import argparse
import datetime
import json
from pathlib import Path

from llava_api.hook import HookManager

def init_hookmanager(module):
    module.hook_manager = HookManager()

class MaskHookLogger(object):
    def __init__(self, model, device, noise_std=0.01, mod_ratio=0.5, mode='suppress_topk'):
        self.current_layer = 0
        self.device = device
        self.attns = []
        self.projected_attns = []
        self.image_embed_range = []
        self.index = []
        self.attn_weights = []
        self.model = model
        self.modified_attns = None
        self.noise_std = noise_std
        self.mod_ratio = mod_ratio
        self.mode = mode
        self.call_count = 0

    @torch.no_grad()
    def modify_attention(self, ret):
        st, ed = 35, 611
        self.call_count += 1

        # 偶数次才修改（负样本）或正样本处理
        if self.call_count % 2 != 0:
            if self.mode == 'suppress_low':
                attention = ret[:, :, -1, st:ed].detach()
                B, H, K = attention.shape
                avg_attention = attention.mean(dim=(0, 1))  # shape: [K]
                _, indices = torch.topk(avg_attention, k=num_tokens, largest=False)
                mask = torch.ones(K, device=ret.device)
                mask[indices] = 0

                mask = mask.view(1, 1, -1).expand_as(attention)
                modified = attention * mask
                self.attns.append(modified.mean(dim=1))
                ret[:, :, -1, st:ed] = modified                            
            return ret

        attention = ret[:, :, -1, st:ed].detach()
        B, H, K = attention.shape
        avg_attention = attention.mean(dim=(0, 1))  # shape: [K]
        num_tokens = int(K * self.mod_ratio)

        if self.mode == 'suppress_topk':
            _, indices = torch.topk(avg_attention, k=num_tokens)
            mask = torch.ones(K, device=ret.device)
            mask[indices] = 0

        elif self.mode == 'add_noise':
            noise = torch.randn_like(attention) * self.noise_std
            attention = attention + noise
            self.attns.append(attention.mean(dim=1))
            ret[:, :, -1, st:ed] = attention
            return ret

        elif self.mode == 'shuffle':
            attention_flat = attention.view(-1, K)
            for i in range(attention_flat.shape[0]):
                idx = torch.randperm(K)[:num_tokens]
                attention_flat[i, idx] = attention_flat[i, idx][torch.randperm(num_tokens)]
            attention = attention_flat.view(B, H, K)
            self.attns.append(attention.mean(dim=1))
            ret[:, :, -1, st:ed] = attention
            return ret

        elif self.mode == 'zeroout':
            idx = torch.randperm(K)[:num_tokens]
            mask = torch.ones(K, device=ret.device)
            mask[idx] = 0

        else:
            raise ValueError(f"Unknown attention modification mode: {self.mode}")

        mask = mask.view(1, 1, -1).expand_as(attention)
        modified = attention * mask
        self.attns.append(modified.mean(dim=1))
        ret[:, :, -1, st:ed] = modified
        return ret

    @torch.no_grad()
    def compute_attentions(self, ret):
        st, ed = 35, 611
        image_attention = ret[:, :, -1, st:ed].detach()
        image_attention = image_attention.mean(dim=1)
        self.attns.append(image_attention)
        return ret

    @torch.no_grad()
    def finalize(self):
        attns = torch.cat(self.attns, dim=0).to(self.device)
        return attns.mean(dim=0)

    def reinit(self):
        self.attns = []
        self.projected_attns = []
        self.image_embed_range = []
        self.index = []
        torch.cuda.empty_cache()

    def log_image_embeds_range(self, ret):
        self.image_embed_range.append(ret[0][0])
        return ret

def hook_logger(model, device, layer_index=0, noise_std=0.1, modify_attention=True, mod_ratio=0.5, mode='suppress_topk'):
    init_hookmanager(model.model.layers[layer_index].self_attn)
    prs = MaskHookLogger(model, device, noise_std=noise_std, mod_ratio=mod_ratio, mode=mode)

    if modify_attention:
        hook_fn = prs.modify_attention
    else:
        hook_fn = prs.compute_attentions

    model.model.layers[layer_index].self_attn.hook_manager.register(
        'after_attn_mask', hook_fn
    )
    model.hooklogger = prs
    return prs
