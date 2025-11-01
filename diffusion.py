from pathlib import Path
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformer_denoiser import TransformerDenoiser
from latentflow_denoiser import LatentFlowDenoiser

def make_beta_schedule(T=1000, beta_start=1e-4, beta_end=0.02):
    """Linear β schedule for diffusion (T steps)."""
    return torch.linspace(beta_start, beta_end, T)

class DiffusionUtils:
    def __init__(self, T=1000, beta_start=1e-4, beta_end=0.02, device="cpu"):
        self.device = device
        self.T = T
        self.betas = make_beta_schedule(T, beta_start, beta_end).to(device)
        self.alphas = 1.0 - self.betas
        self.alpha_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alpha_cumprod = torch.sqrt(self.alpha_cumprod)
        self.sqrt_one_minus_alpha_cumprod = torch.sqrt(1.0 - self.alpha_cumprod)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        
        self.alpha_cumprod_prev = F.pad(self.alpha_cumprod[:-1], (1, 0), value=1.0)
        self.posterior_variance = (
            self.betas * (1.0 - self.alpha_cumprod_prev) / (1.0 - self.alpha_cumprod)
        )

    def q_sample(self, x0, t, noise=None):
        """Forward diffusion: sample x_t ~ q(x_t | x_0)"""
        if noise is None:
            noise = torch.randn_like(x0)
        sqrt_ac = self._extract(self.sqrt_alpha_cumprod, t, x0.shape)
        sqrt_om = self._extract(self.sqrt_one_minus_alpha_cumprod, t, x0.shape)
        return sqrt_ac * x0 + sqrt_om * noise

    def p_sample(self, model, x_t, t, cond=None):
        """Reverse step: predict x_{t-1} from x_t"""
        betas_t = self._extract(self.betas, t, x_t.shape)
        sqrt_recip_alphas_t = self._extract(self.sqrt_recip_alphas, t, x_t.shape)
        sqrt_one_minus_ac_t = self._extract(self.sqrt_one_minus_alpha_cumprod, t, x_t.shape)

        pred_noise = model(x_t, t, cond=cond)
        mean = sqrt_recip_alphas_t * (x_t - betas_t / sqrt_one_minus_ac_t * pred_noise)

        if t[0] == 0:
            return mean
        noise = torch.randn_like(x_t)
        posterior_var = self._extract(self.posterior_variance, t, x_t.shape)
        return mean + torch.sqrt(posterior_var) * noise

    def sample_loop(self, model, shape, cond=None):
        """Generate a sample from pure noise by iterative denoising."""
        x = torch.randn(shape, device=self.device)
        for i in reversed(range(self.T)):
            t = torch.full((shape[0],), i, device=self.device, dtype=torch.long)
            x = self.p_sample(model, x, t, cond=cond)
        return x

    def _extract(self, a, t, x_shape):
        """Helper to match shapes for time-dependent coefficients."""
        out = a.gather(-1, t)
        while len(out.shape) < len(x_shape):
            out = out.unsqueeze(-1)
        return out

class TemporalUNet(nn.Module):
    def __init__(self, num_joints=22, coord_dim=3, base_channels=128, time_embed_dim=128):
        super().__init__()
        self.in_dim = num_joints * coord_dim

        self.time_mlp = nn.Sequential(
            nn.Linear(1, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

        self.cond_encoder = nn.Sequential(
            nn.Conv1d(self.in_dim, base_channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, base_channels),   
            nn.SiLU(),
            nn.Conv1d(base_channels, base_channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, base_channels),   
            nn.SiLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        self.cond_proj = nn.Linear(base_channels, base_channels * 2)

        self.encoder1 = nn.Sequential(               
            nn.Conv1d(self.in_dim, base_channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, base_channels),
            nn.SiLU()
        )
        self.encoder2 = nn.Sequential(
            nn.Conv1d(base_channels, base_channels * 2, kernel_size=3, padding=1),
            nn.GroupNorm(8, base_channels * 2),
            nn.SiLU()
        )

        self.time_proj1 = nn.Linear(time_embed_dim, base_channels)
        self.time_proj2 = nn.Linear(time_embed_dim, base_channels * 2)

        self.middle = nn.Sequential(
            nn.Conv1d(base_channels * 2, base_channels * 2, kernel_size=3, padding=1),
            nn.GroupNorm(8, base_channels * 2),
            nn.SiLU()
        )
        self.decoder1 = nn.Sequential(
            nn.Conv1d(base_channels * 4, base_channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, base_channels),
            nn.SiLU()
        )
        self.out_conv = nn.Conv1d(base_channels, self.in_dim, kernel_size=3, padding=1)
        
        nn.init.normal_(self.out_conv.weight, mean=0.0, std=0.1)   
        nn.init.zeros_(self.out_conv.bias)

    def forward(self, x, t, cond=None):
        B, T_future, J, C = x.shape
        x = x.reshape(B, T_future, J * C).permute(0, 2, 1)

        t_embed = self.time_mlp(t.float().unsqueeze(-1)) * 10.0     

        if cond is not None:
            T_past = cond.shape[1]
            cond_flat = cond.reshape(B, T_past, J * C).permute(0, 2, 1)
            cond_enc = self.cond_encoder(cond_flat).squeeze(-1)
            cond_emb = self.cond_proj(cond_enc).unsqueeze(-1) * 5.0   
        else:
            cond_emb = None

        h1 = self.encoder1(x)
        h1 = h1 + self.time_proj1(t_embed).unsqueeze(-1)
        h2 = self.encoder2(h1)
        h2 = h2 + self.time_proj2(t_embed).unsqueeze(-1)
        if cond_emb is not None:
            h2 = h2 + cond_emb
        h = self.middle(h2)

        h = torch.cat([h, h2], dim=1)
        h = self.decoder1(h)
        out = self.out_conv(h)
        out = out.permute(0, 2, 1).reshape(B, T_future, J, C)
        return out
    

def diffusion_loss(model, utils: DiffusionUtils, x0_future, x_past=None, device="cpu"):
    """
    Args:
        x0_future: (B, T_future, J, 3) - clean future poses
        x_past: (B, T_past, J, 3) - clean past poses for conditioning
    """
    B = x0_future.shape[0]
    t = torch.randint(0, utils.T, (B,), device=device).long()
    noise = torch.randn_like(x0_future)
    x_t = utils.q_sample(x0_future, t, noise)
    noise_pred = model(x_t, t, cond=x_past)
    return F.mse_loss(noise_pred, noise)



class PoseDiffusion(nn.Module):
    def __init__(self, num_joints=22, coord_dim=3, T=1000, device="cpu"):
        super().__init__()
        self.device = device
        self.utils = DiffusionUtils(T=T, device=device)
        self.model = LatentFlowDenoiser(num_joints=num_joints, coord_dim=coord_dim, vae_pretrained_path="runs/vae/vae_pretrained.pt", freeze_vae=True).to(device)
        #self.model = TransformerDenoiser(num_joints=num_joints, coord_dim=coord_dim).to(device)
        #self.model = TemporalUNet(num_joints=num_joints, coord_dim=coord_dim).to(device)

    def training_loss(self, x0_future, x_past=None):
        """
        Args:
            x0_future: (B, T_future, J, 3) - future poses to predict
            x_past: (B, T_past, J, 3) - past observed poses
        """
        return diffusion_loss(self.model, self.utils, x0_future, x_past, device=self.device)

    @torch.no_grad()
    def sample(self, batch_size, seq_len, cond=None, num_joints=22, coord_dim=3):
        """
        Args:
            cond: (B, T_past, J, 3) - past poses for conditioning
        """
        shape = (batch_size, seq_len, num_joints, coord_dim)
        return self.utils.sample_loop(self.model, shape, cond=cond)