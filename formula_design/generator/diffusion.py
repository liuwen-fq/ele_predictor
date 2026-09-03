# Some modules used in this code are adapted from: https://github.com/hspark1212/chemeleon/blob/df4c67449265f5dfad66b1300ba438b35b9fcb7c/chemeleon/modules/cspnet.py
# Copyright (c) 2024 Hyunsoo Park
# Copyright (c) 2025 ByteDance Ltd. 
# SPDX-License-Identifier: MIT license
#
# This file has been modified by ByteDance Ltd. on 08/26/2025
#
# Original file was released under MIT license, with the full license text
# available at https://github.com/hspark1212/chemeleon/blob/df4c67449265f5dfad66b1300ba438b35b9fcb7c/LICENSE.
#
# This modified file is released under Apache License 2.0.

import math
import os
import time
from typing import Any, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from tqdm import tqdm

from formula_design.data import Data
from formula_design.utils.diff_utils import BetaScheduler, SigmaScheduler

from .decoder import FormulaDecoder
from .unet1d import Unet1D


### Model definition
class ScalarEmbed(nn.Module):

    def __init__(self, embed_dim=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(1, embed_dim), nn.SiLU(),
                                 nn.Linear(embed_dim, embed_dim))

    def forward(self, x):
        # x shape: [batch_size] or [batch_size, 1]
        return self.net(x).unsqueeze(1)  # 输出 [batch_size, 1, embed_dim]


class SinusoidalTimeEmbeddings(nn.Module):
    """ Attention is all you need. """

    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(
            torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings


class FormulaDiffusion(nn.Module):

    def __init__(self, unet1d_config: dict, beta_scheduler_config: dict,
                 sigma_scheduler_config: dict, diff_config: dict,
                 embedding_dim: int = 384) -> None:
        super().__init__()

        self.noise_predictor = Unet1D(**unet1d_config)
        self.prop_dim = int(unet1d_config.get("prop_dim", 2))
        if self.prop_dim not in (2, 4):
            raise ValueError(
                "FormulaDiffusion currently supports 2-D [sigma, anion_ratio] "
                "or 4-D [sigma, anion_ratio, I_sigma, I_r] conditions; "
                f"got prop_dim={self.prop_dim}")

        self.beta_scheduler = BetaScheduler(**beta_scheduler_config)
        self.sigma_scheduler = SigmaScheduler(**sigma_scheduler_config)
        self.time_dim = diff_config.get("time_dim")
        self.embedding_dim = int(embedding_dim)

        # ``mechanism_auxiliary`` is deliberately an *auxiliary supervision*
        # path, rather than another sampling condition.  This lets a model
        # learn formula--interaction correspondence while retaining the
        # deployable conditional interface p(formula | conductivity,
        # anion_ratio).  In particular, G2-MA must not need to guess an
        # interaction vector at inference time.
        mechanism_auxiliary = dict(diff_config.get("mechanism_auxiliary", {}))
        self.mechanism_auxiliary_enabled = bool(
            mechanism_auxiliary.get("enabled", False))
        self.mechanism_auxiliary_label_dim = int(
            mechanism_auxiliary.get("label_dim", 2))
        if self.mechanism_auxiliary_enabled:
            if self.mechanism_auxiliary_label_dim < 1:
                raise ValueError(
                    "mechanism_auxiliary.label_dim must be positive when "
                    "the auxiliary head is enabled")
            auxiliary_hidden_dim = int(
                mechanism_auxiliary.get("hidden_dim", self.embedding_dim // 2))
            if auxiliary_hidden_dim < 1:
                raise ValueError(
                    "mechanism_auxiliary.hidden_dim must be positive")
            # The head receives the denoiser's reconstructed clean formula
            # embedding, not the ground-truth embedding.  Its gradient can
            # therefore improve the denoising path instead of training an
            # unrelated post-hoc regressor.
            self.mechanism_auxiliary_head = nn.Sequential(
                nn.LayerNorm(self.embedding_dim),
                nn.Linear(self.embedding_dim, auxiliary_hidden_dim),
                nn.SiLU(),
                nn.Linear(auxiliary_hidden_dim,
                          self.mechanism_auxiliary_label_dim),
            )
        else:
            self.mechanism_auxiliary_head = None

        # self.prop_embedding = ScalarEmbed(diff_config.get("prop_embed_dim"))

    def load_ckpt(self, ckpt_path):
        self.load_state_dict(torch.load(ckpt_path)['model_state_dict'])

    def forward(self, data: Data):

        batch_size = data.frm_emb.size()[0]

        device = next(self.noise_predictor.parameters()).device

        times = self.beta_scheduler.uniform_sample_t(batch_size, device)

        alphas_cumprod = self.beta_scheduler.alphas_cumprod[times]

        c0 = torch.sqrt(alphas_cumprod)
        c1 = torch.sqrt(1. - alphas_cumprod)

        frm_emb = data["frm_emb"].unsqueeze(1)
        prop_parts = [
            data['conductivity'].unsqueeze(-1),
            data['anion_ratio'].unsqueeze(-1)
        ]
        if self.prop_dim == 4:
            if 'mechanism_emb' not in data:
                raise KeyError(
                    "4-D mechanism-conditioned diffusion requires "
                    "data['mechanism_emb'] = [I_sigma, I_r]")
            mechanism_emb = data['mechanism_emb']
            if mechanism_emb.ndim != 2 or mechanism_emb.shape[-1] != 2:
                raise ValueError(
                    "mechanism_emb must have shape [batch, 2], got "
                    f"{tuple(mechanism_emb.shape)}")
            prop_parts.append(mechanism_emb)
        prop = torch.cat(prop_parts, dim=-1)
        if prop.shape[-1] != self.prop_dim:
            raise RuntimeError(
                f"assembled condition has dim={prop.shape[-1]}, expected "
                f"prop_dim={self.prop_dim}")

        #rand_e, rand_p = torch.randn_like(frm_emb), torch.randn_like(p_embed)
        rand_e = torch.randn_like(frm_emb)

        input_e = c0[:, None, None] * frm_emb + c1[:, None, None] * rand_e

        pred_e = self.noise_predictor(times, input_e, prop)

        output = {"pred_e": pred_e, "rand_e": rand_e}
        if self.mechanism_auxiliary_head is not None:
            # x_0 = (x_t - sqrt(1 - alpha_bar_t) * eps_hat) /
            #       sqrt(alpha_bar_t).  The scheduler does not sample t=0,
            # so c0 is strictly positive here.
            pred_clean_emb = (input_e - c1[:, None, None] * pred_e) / (
                c0[:, None, None])
            output["mechanism_aux_pred"] = self.mechanism_auxiliary_head(
                pred_clean_emb.squeeze(1))
            # High-noise timesteps contain very little recoverable formula
            # information.  The loss uses this confidence weight to keep the
            # auxiliary task from overwhelming the base denoising objective.
            output["mechanism_aux_snr_weight"] = alphas_cumprod.detach()

        return output

    @torch.no_grad()
    def sample(self, prop):

        batch_size = prop.size()[0]
        device = next(self.noise_predictor.parameters()).device
        if prop.ndim != 2 or prop.shape[-1] != self.prop_dim:
            raise ValueError(
                "diffusion sampling condition must have shape "
                f"[batch, {self.prop_dim}], got {tuple(prop.shape)}")

        e_T = torch.randn([batch_size, 1, self.embedding_dim]).to(device)

        time_start = self.beta_scheduler.timesteps

        traj = {
            time_start: {
                'frm_emb': e_T,
            }
        }

        for t in tqdm(range(time_start, 0, -1)):

            times = torch.full((batch_size, ), t, device=device)

            alphas = self.beta_scheduler.alphas[t]
            alphas_cumprod = self.beta_scheduler.alphas_cumprod[t]

            sigmas = self.beta_scheduler.sigmas[t]

            c0 = 1.0 / torch.sqrt(alphas)
            c1 = (1 - alphas) / torch.sqrt(1 - alphas_cumprod)

            e_t = traj[t]['frm_emb']

            rand_e = torch.randn_like(e_T) if t > 1 else torch.zeros_like(e_T)

            pred_e = self.noise_predictor(times, e_t, prop)
            e_t_minus_1 = c0 * (e_t - c1 * pred_e) + sigmas * rand_e

            traj[t - 1] = {
                'frm_emb': e_t_minus_1,
            }

        traj_stack = {
            'all_frm_emb':
            torch.stack(
                [traj[i]['frm_emb'] for i in range(time_start, -1, -1)]),
        }

        return traj[0], traj_stack

    def compute_stats(self, output_dict, prefix):

        loss_emb = output_dict['loss_emb']
        loss_prop = output_dict['loss_prop']
        loss = output_dict['loss']

        log_dict = {
            f'{prefix}_loss': loss,
            f'{prefix}_emb_loss': loss_emb,
            f'{prefix}_prop_loss': loss_prop
        }

        return log_dict, loss
