"""Surface Action Model: frozen PI0.5 VLM, action expert and point DiT."""

import math

import torch
from torch import nn
from torch.nn import functional as F

from openpi.models_pytorch.pi0_pytorch import PI0Pytorch, make_att_2d_masks


class PointDiTBlock(nn.Module):
    """Time-adaptive transformer with self attention and context cross attention."""

    def __init__(self, width, heads):
        super().__init__()
        self.norm1 = nn.LayerNorm(width, elementwise_affine=False)
        self.norm2 = nn.LayerNorm(width)
        self.norm3 = nn.LayerNorm(width, elementwise_affine=False)
        self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(width, 6 * width))
        self.self_attention = nn.MultiheadAttention(width, heads, batch_first=True)
        self.cross_attention = nn.MultiheadAttention(width, heads, batch_first=True)
        self.mlp = nn.Sequential(nn.Linear(width, 4 * width), nn.GELU(), nn.Linear(4 * width, width))

    def forward(self, x, context, time, context_mask):
        shift1, scale1, gate1, shift2, scale2, gate2 = self.modulation(time).chunk(6, -1)
        z = self.norm1(x) * (1 + scale1[:, None]) + shift1[:, None]
        x = x + gate1[:, None].tanh() * self.self_attention(z, z, z, need_weights=False)[0]
        x = x + self.cross_attention(self.norm2(x), context, context, key_padding_mask=~context_mask, need_weights=False)[0]
        z = self.norm3(x) * (1 + scale2[:, None]) + shift2[:, None]
        return x + gate2[:, None].tanh() * self.mlp(z)


class SurfacePointDiT(nn.Module):
    def __init__(self, prefix_dim, horizon, action_dim=32, width=256, layers=4, heads=8):
        super().__init__()
        self.horizon = horizon
        self.noisy_projection = nn.Linear(3, width)
        self.surface_projection = nn.Sequential(nn.Linear(3, width), nn.SiLU(), nn.Linear(width, width))
        self.prefix_projection = nn.Linear(prefix_dim, width)
        # Condition on the same action chunk to align the two stochastic outputs.
        self.action_projection = nn.Linear(action_dim, width)
        self.horizon_embedding = nn.Parameter(torch.randn(horizon, width) * .02)
        self.time_projection = nn.Sequential(nn.Linear(width, width), nn.SiLU(), nn.Linear(width, width))
        self.blocks = nn.ModuleList(PointDiTBlock(width, heads) for _ in range(layers))
        self.point_decoder = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width), nn.SiLU(), nn.Linear(width, 3))

    def forward(self, noisy_offsets, time, current_surface, prefix, prefix_mask, action_chunk):
        b, h, k, _ = noisy_offsets.shape
        if h != self.horizon or current_surface.shape != (b, k, 3):
            raise ValueError("Surface/action horizon or current surface shape mismatch")
        width = self.noisy_projection.out_features
        frequencies = torch.exp(torch.linspace(0, -math.log(10000), width // 2, device=time.device))
        phase = time.float()[:, None] * frequencies[None] * 1000
        time_embedding = self.time_projection(torch.cat((phase.sin(), phase.cos()), -1))
        current_tokens = self.surface_projection(current_surface.float())
        x = self.noisy_projection(noisy_offsets.float()) + current_tokens[:, None] + self.horizon_embedding[None, :, None]
        context = torch.cat((self.prefix_projection(prefix.float()), current_tokens, self.action_projection(action_chunk.float())), 1)
        extra_mask = torch.ones((b, k + h), dtype=torch.bool, device=x.device)
        context_mask = torch.cat((prefix_mask.bool(), extra_mask), 1)
        x = x.reshape(b, h * k, width)
        for block in self.blocks:
            if self.training and torch.is_grad_enabled():
                x = torch.utils.checkpoint.checkpoint(block, x, context, time_embedding, context_mask, use_reentrant=False)
            else:
                x = block(x, context, time_embedding, context_mask)
        return self.point_decoder(x).reshape(b, h, k, 3)


class SurfaceActionModelPytorch(PI0Pytorch):
    schema_version = 3

    def __init__(self, config, *, joint_dim=12):
        if not config.pi05:
            raise ValueError("SurfaceActionModelPytorch requires PI0.5")
        super().__init__(config)
        self.surface_joint_dim = joint_dim
        prefix_width = self.paligemma_with_expert.paligemma.config.text_config.hidden_size
        self.surface_dit = SurfacePointDiT(prefix_width, config.action_horizon, config.action_dim)
        self.paligemma_with_expert.paligemma.requires_grad_(False)

    @torch.no_grad()
    def _surface_context(self, observation):
        images, masks, tokens, token_masks, _ = self._preprocess_observation(observation, train=False)
        prefix, padding, attention = self.embed_prefix(images, masks, tokens, token_masks)
        outputs, _ = self.paligemma_with_expert.forward(
            attention_mask=self._prepare_attention_masks_4d(make_att_2d_masks(padding, attention)),
            position_ids=torch.cumsum(padding, dim=1) - 1,
            inputs_embeds=[prefix, None], use_cache=False,
        )
        return outputs[0].detach().float(), padding

    def compute_losses(self, observation, actions, robot_points, joint_positions, target_point_offsets,
                       target_point_mask=None, *, point_loss_weight=1., noise=None, time=None,
                       point_noise=None, action_mask=None):
        if not all(torch.isfinite(x).all() for x in (robot_points, target_point_offsets, actions)):
            raise ValueError("Training targets must be finite; mask missing observations before loading")
        if time is None:
            time = self.sample_time(actions.shape[0], actions.device)
        action_error = super().forward(observation, actions, noise=noise, time=time)
        if action_mask is None:
            action_loss = action_error.mean()
        else:
            mask = action_mask.to(action_error).expand_as(action_error)
            action_loss = (action_error * mask).sum() / mask.sum().clamp_min(1)
        prefix, prefix_mask = self._surface_context(observation)
        if point_noise is None:
            point_noise = torch.randn_like(target_point_offsets)
        t = time[:, None, None, None]
        noisy = t * point_noise + (1 - t) * target_point_offsets
        velocity_target = point_noise - target_point_offsets
        velocity = self.surface_dit(noisy, time, robot_points, prefix, prefix_mask, actions)
        error = (velocity - velocity_target).square().mean(-1)
        mask = torch.ones_like(error) if target_point_mask is None else target_point_mask.to(error)
        point_loss = (error * mask).sum() / mask.sum().clamp_min(1)
        return {"loss": action_loss + point_loss_weight * point_loss,
                "action_loss": action_loss.detach(), "point_loss": point_loss.detach(), "point_velocity": velocity}

    def forward(self, *args, **kwargs):
        return self.compute_losses(*args, **kwargs)

    @torch.no_grad()
    def sample_actions_and_point_offsets(self, device, observation, robot_points, joint_positions, noise=None, num_steps=10):
        if num_steps < 1:
            raise ValueError("num_steps must be positive")
        actions = PI0Pytorch.sample_actions(self, device, observation, noise=noise, num_steps=num_steps)
        prefix, prefix_mask = self._surface_context(observation)
        offsets = self.sample_surface_offsets(robot_points, prefix, prefix_mask, actions, num_steps=num_steps)
        return actions, offsets

    @torch.no_grad()
    def sample_surface_offsets(self, current_surface, prefix, prefix_mask, action_chunk, *, num_steps=10):
        if num_steps < 1:
            raise ValueError("num_steps must be positive")
        device = current_surface.device
        robot_points = current_surface
        b, k, _ = robot_points.shape
        offsets = torch.randn((b, self.config.action_horizon, k, 3), device=device)
        dt = -1. / num_steps
        for step in range(num_steps):
            time = torch.full((b,), 1. - step / num_steps, device=device)
            offsets = offsets + dt * self.surface_dit(offsets, time, robot_points, prefix, prefix_mask, action_chunk)
        return offsets
