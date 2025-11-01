import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformerBlock(nn.Module):
    """Efficient transformer block for temporal modeling."""
    def __init__(self, dim, num_heads=4, mlp_ratio=2.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        
        mlp_hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden),
            nn.GELU(),
            nn.Linear(mlp_hidden, dim)
        )

    def forward(self, x):
        # x: (B, T, D)
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.mlp(self.norm2(x))
        return x

class TransformerDenoiser(nn.Module):
    def __init__(self, num_joints=22, coord_dim=3, hidden_dim=128, num_layers=4, num_heads=4):
        super().__init__()
        self.num_joints = num_joints
        self.coord_dim = coord_dim
        self.in_dim = num_joints * coord_dim
        self.hidden_dim = hidden_dim

        self.time_mlp = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.cond_encoder = nn.Sequential(
            nn.Conv1d(self.in_dim, hidden_dim, kernel_size=3, padding=1),
            nn.GroupNorm(8, hidden_dim),
            nn.SiLU(),
            nn.AdaptiveAvgPool1d(1)
        )
        self.cond_proj = nn.Linear(hidden_dim, hidden_dim)

        self.input_proj = nn.Linear(self.in_dim, hidden_dim)

        self.blocks = nn.ModuleList([
            TransformerBlock(hidden_dim, num_heads) for _ in range(num_layers)
        ])

        self.output_proj = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, self.in_dim)
        )

        nn.init.normal_(self.output_proj[1].weight, mean=0.0, std=0.1)
        nn.init.zeros_(self.output_proj[1].bias)

    def forward(self, x, t, cond=None):
        B, T_future, J, C = x.shape

        x_flat = x.reshape(B, T_future, J * C)  # (B, T_future, in_dim)

        t_embed = self.time_mlp(t.float().unsqueeze(-1)) * 10.0  # (B, hidden_dim)

        if cond is not None:
            T_past = cond.shape[1]
            cond_flat = cond.reshape(B, T_past, J * C).permute(0, 2, 1)  # (B, in_dim, T_past)
            cond_enc = self.cond_encoder(cond_flat).squeeze(-1)  # (B, hidden_dim)
            cond_emb = self.cond_proj(cond_enc) * 5.0  # (B, hidden_dim)
        else:
            cond_emb = None

        h = self.input_proj(x_flat)  # (B, T_future, hidden_dim)

        h = h + t_embed.unsqueeze(1)  # Broadcast time to all frames

        if cond_emb is not None:
            h = h + cond_emb.unsqueeze(1)

        for block in self.blocks:
            h = block(h)

        out = self.output_proj(h)  # (B, T_future, in_dim)

        out = out.reshape(B, T_future, J, C)

        return out

if __name__ == "__main__":
    print("Testing Transformer Denoiser...")
    model = TransformerDenoiser().to("cpu")
    
    x = torch.randn(2, 16, 22, 3)
    t = torch.randint(0, 1000, (2,))
    cond = torch.randn(2, 8, 22, 3)
    
    y = model(x, t, cond=cond)
    print(f"Output shape: {y.shape}")
    print(f"Expected: torch.Size([2, 16, 22, 3])")
    assert y.shape == torch.Size([2, 16, 22, 3]), "Shape mismatch!"
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    print("All tests passed!")