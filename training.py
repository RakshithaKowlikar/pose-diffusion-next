# from pathlib import Path
# import torch
# import torch.nn.functional as F
# from torch.utils.data import DataLoader
# from torch.optim import Adam
# from tqdm import tqdm

# from data_loader import AMASSForecastDataset
# from diffusion import PoseDiffusion

# # ============================================================
# # CONFIG
# # ============================================================

# SPLIT_FILE = Path("data/AMASS/train.txt")
# VAL_SPLIT_FILE = Path("data/AMASS/val.txt")
# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# BATCH_SIZE = 32
# EPOCHS = 100
# LR = 1e-4
# SAVE_DIR = Path("runs/baseline")
# SAVE_DIR.mkdir(parents=True, exist_ok=True)

# # ============================================================
# # DATASET
# # ============================================================

# train_dataset = AMASSForecastDataset(split_file=SPLIT_FILE)
# val_dataset = AMASSForecastDataset(split_file=VAL_SPLIT_FILE)

# train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
# val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

# # ============================================================
# # MODEL + OPTIMIZER
# # ============================================================

# model = PoseDiffusion(num_joints=22, coord_dim=3, T=1000, device=DEVICE).to(DEVICE)
# optimizer = Adam(model.parameters(), lr=LR)

# # ============================================================
# # METRICS
# # ============================================================

# def mpjpe(pred, target):
#     """Mean per-joint position error."""
#     return torch.norm(pred - target, dim=-1).mean()

# def velocity_error(pred, target):
#     """Mean velocity difference error."""
#     vel_pred = pred[:, 1:] - pred[:, :-1]
#     vel_target = target[:, 1:] - target[:, :-1]
#     return torch.norm(vel_pred - vel_target, dim=-1).mean()

# # ============================================================
# # TRAINING LOOP
# # ============================================================

# def train_one_epoch(epoch):
#     model.train()
#     total_loss = 0.0
#     for past_poses, future_poses, _, _ in tqdm(train_loader, desc=f"Train {epoch}"):
#         future_poses = future_poses.to(DEVICE)  # [B, T, 22, 3]
#         loss = model.training_loss(future_poses)

#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

#         total_loss += loss.item()

#     avg_loss = total_loss / len(train_loader)
#     print(f"[Epoch {epoch}] Train Loss: {avg_loss:.6f}")
#     return avg_loss

# @torch.no_grad()
# def validate(epoch):
#     model.eval()
#     total_mpjpe = 0.0
#     total_vel = 0.0
#     n_batches = 0

#     for past_poses, future_poses, _, _ in tqdm(val_loader, desc=f"Val {epoch}"):
#         future_poses = future_poses.to(DEVICE)
#         # Sample future poses from diffusion model
#         B, T, J, C = future_poses.shape
#         samples = model.sample(B, T, J, C)

#         total_mpjpe += mpjpe(samples, future_poses).item()
#         total_vel += velocity_error(samples, future_poses).item()
#         n_batches += 1

#     avg_mpjpe = total_mpjpe / n_batches
#     avg_vel = total_vel / n_batches
#     print(f"[Epoch {epoch}] Val MPJPE: {avg_mpjpe:.6f}, Velocity Error: {avg_vel:.6f}")
#     return avg_mpjpe, avg_vel

# # ============================================================
# # CHECKPOINTING
# # ============================================================

# def save_checkpoint(epoch, model, optimizer, train_loss, val_mpjpe, val_vel):
#     ckpt = {
#         "epoch": epoch,
#         "model_state_dict": model.state_dict(),
#         "optimizer_state_dict": optimizer.state_dict(),
#         "train_loss": train_loss,
#         "val_mpjpe": val_mpjpe,
#         "val_vel": val_vel
#     }
#     path = SAVE_DIR / f"epoch_{epoch:03d}.pt"
#     torch.save(ckpt, path)
#     print(f"[Checkpoint] Saved: {path}")

# # ============================================================
# # MAIN TRAINING SCRIPT
# # ============================================================

# def main():
#     best_mpjpe = float("inf")

#     for epoch in range(1, EPOCHS + 1):
#         train_loss = train_one_epoch(epoch)
#         val_mpjpe, val_vel = validate(epoch)

#         save_checkpoint(epoch, model, optimizer, train_loss, val_mpjpe, val_vel)

#         if val_mpjpe < best_mpjpe:
#             best_mpjpe = val_mpjpe
#             torch.save(model.state_dict(), SAVE_DIR / "best_model.pt")
#             print(f"[Best Model] Epoch {epoch} with MPJPE: {best_mpjpe:.6f}")

#     print("✅ Training complete.")

# if __name__ == "__main__":
#     main()

# from pathlib import Path
# import torch
# import torch.nn.functional as F
# from torch.utils.data import DataLoader
# from torch.optim import Adam
# from torch.optim.lr_scheduler import CosineAnnealingLR
# from tqdm import tqdm

# from data_loader import AMASSForecastDataset
# from diffusion import PoseDiffusion

# # ============================================================
# # CONFIG
# # ============================================================

# SPLIT_FILE = Path("data/AMASS/train.txt")
# VAL_SPLIT_FILE = Path("data/AMASS/val.txt")
# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# BATCH_SIZE = 32
# N_EPOCHS = 100
# LR = 1e-4
# VAL_DIFFUSION_STEPS = 50  # Faster validation (vs 1000 for training)
# SAVE_DIR = Path("runs/baseline")
# SAVE_DIR.mkdir(parents=True, exist_ok=True)

# # ============================================================
# # DATASET
# # ============================================================

# train_dataset = AMASSForecastDataset(split_file=SPLIT_FILE)
# val_dataset = AMASSForecastDataset(split_file=VAL_SPLIT_FILE)

# train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
# val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

# # ============================================================
# # MODEL + OPTIMIZER
# # ============================================================

# model = PoseDiffusion(num_joints=22, coord_dim=3, T=1000, device=DEVICE).to(DEVICE)
# optimizer = Adam(model.parameters(), lr=LR)
# scheduler = CosineAnnealingLR(optimizer, T_max=N_EPOCHS)

# # ============================================================
# # METRICS
# # ============================================================

# def mpjpe(pred, target):
#     """Mean per-joint position error."""
#     return torch.norm(pred - target, dim=-1).mean()

# def velocity_error(pred, target):
#     """Mean velocity difference error."""
#     vel_pred = pred[:, 1:] - pred[:, :-1]
#     vel_target = target[:, 1:] - target[:, :-1]
#     return torch.norm(vel_pred - vel_target, dim=-1).mean()

# # ============================================================
# # TRAINING LOOP
# # ============================================================

# # def train_one_epoch(epoch):
# #     model.train()
# #     total_loss = 0.0
# #     for past_poses, future_poses, _, _ in tqdm(train_loader, desc=f"Train {epoch}"):
# #         past_poses = past_poses.to(DEVICE)
# #         future_poses = future_poses.to(DEVICE)
        
# #         loss = model.training_loss(future_poses, x_past=past_poses)

# #         optimizer.zero_grad()
# #         loss.backward()
# #         optimizer.step()

# #         total_loss += loss.item()

# #     avg_loss = total_loss / len(train_loader)
# #     print(f"[Epoch {epoch}] Train Loss: {avg_loss:.6f}")
# #     return avg_loss

# def train_one_epoch(epoch):
#     model.train()
#     total_loss = 0.0
    
#     # ✅ Diagnostic: Check first batch only
#     first_batch = True
    
#     for past_poses, future_poses, _, _ in tqdm(train_loader, desc=f"Train {epoch}"):
#         past_poses = past_poses.to(DEVICE)
#         future_poses = future_poses.to(DEVICE)
        
#         # ✅ DIAGNOSTIC BLOCK (only runs once per epoch)
#         if first_batch and epoch == 1:
#             print(f"\n=== DIAGNOSTICS ===")
#             print(f"Past shape: {past_poses.shape}")
#             print(f"Future shape: {future_poses.shape}")
#             print(f"Future mean: {future_poses.mean().item():.4f}")
#             print(f"Future std: {future_poses.std().item():.4f}")
            
#             # Check model output
#             with torch.no_grad():
#                 t = torch.zeros(future_poses.shape[0], device=DEVICE).long()
#                 noise = torch.randn_like(future_poses)
#                 x_t = model.utils.q_sample(future_poses, t, noise)
#                 pred_noise = model.model(x_t, t, cond=past_poses)
#                 print(f"Pred noise mean: {pred_noise.mean().item():.4f}")
#                 print(f"Pred noise std: {pred_noise.std().item():.4f}")
#                 print(f"True noise mean: {noise.mean().item():.4f}")
#                 print(f"True noise std: {noise.std().item():.4f}")
#             print(f"===================\n")
#             first_batch = False
        
#         loss = model.training_loss(future_poses, x_past=past_poses)

#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

#         total_loss += loss.item()

#     avg_loss = total_loss / len(train_loader)
#     print(f"[Epoch {epoch}] Train Loss: {avg_loss:.6f}")
#     return avg_loss

# @torch.no_grad()
# def validate(epoch):
#     model.eval()
#     total_mpjpe = 0.0
#     total_vel = 0.0
#     n_batches = 0

#     # Temporarily reduce diffusion steps for faster validation
#     original_T = model.utils.T
#     model.utils.T = VAL_DIFFUSION_STEPS

#     for past_poses, future_poses, _, _ in tqdm(val_loader, desc=f"Val {epoch}"):
#         past_poses = past_poses.to(DEVICE)
#         future_poses = future_poses.to(DEVICE)
        
#         B, T, J, C = future_poses.shape
#         samples = model.sample(B, T, cond=past_poses, num_joints=J, coord_dim=C)

#         total_mpjpe += mpjpe(samples, future_poses).item()
#         total_vel += velocity_error(samples, future_poses).item()
#         n_batches += 1

#     # Restore original diffusion steps
#     model.utils.T = original_T

#     avg_mpjpe = total_mpjpe / n_batches
#     avg_vel = total_vel / n_batches
#     print(f"[Epoch {epoch}] Val MPJPE: {avg_mpjpe:.6f}, Velocity Error: {avg_vel:.6f}")
#     return avg_mpjpe, avg_vel

# # ============================================================
# # CHECKPOINTING
# # ============================================================

# def save_checkpoint(epoch, model, optimizer, train_loss, val_mpjpe, val_vel):
#     ckpt = {
#         "epoch": epoch,
#         "model_state_dict": model.state_dict(),
#         "optimizer_state_dict": optimizer.state_dict(),
#         "train_loss": train_loss,
#         "val_mpjpe": val_mpjpe,
#         "val_vel": val_vel
#     }
#     path = SAVE_DIR / f"epoch_{epoch:03d}.pt"
#     torch.save(ckpt, path)
#     print(f"[Checkpoint] Saved: {path}")

# # ============================================================
# # MAIN TRAINING SCRIPT
# # ============================================================

# def main():
#     best_mpjpe = float("inf")

#     for epoch in range(1, N_EPOCHS + 1):
#         train_loss = train_one_epoch(epoch)
#         val_mpjpe, val_vel = validate(epoch)

#         save_checkpoint(epoch, model, optimizer, train_loss, val_mpjpe, val_vel)

#         if val_mpjpe < best_mpjpe:
#             best_mpjpe = val_mpjpe
#             torch.save(model.state_dict(), SAVE_DIR / "best_model.pt")
#             print(f"[Best Model] Epoch {epoch} with MPJPE: {best_mpjpe:.6f}")

#         scheduler.step()

#     print("✅ Training complete.")

# if __name__ == "__main__":
#     main()

# # from pathlib import Path
# # import torch
# # import torch.nn.functional as F
# # from torch.utils.data import DataLoader
# # from torch.optim import Adam
# # from torch.optim.lr_scheduler import CosineAnnealingLR
# # from tqdm import tqdm

# # from data_loader import AMASSForecastDataset
# # from diffusion import PoseDiffusion
# # # ============================================================
# # # CONFIG
# # # ============================================================

# # SPLIT_FILE = Path("data/AMASS/train.txt")
# # VAL_SPLIT_FILE = Path("data/AMASS/val.txt")
# # DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# # BATCH_SIZE = 32
# # N_EPOCHS = 50  # Reduced from 100 for ~5.5 hour training
# # LR = 1e-4
# # VAL_DIFFUSION_STEPS = 30  # Faster validation (was 50)
# # SAVE_DIR = Path("runs/transformer")
# # SAVE_DIR.mkdir(parents=True, exist_ok=True)

# # # ============================================================
# # # DATASET
# # # ============================================================

# # train_dataset = AMASSForecastDataset(split_file=SPLIT_FILE)
# # val_dataset = AMASSForecastDataset(split_file=VAL_SPLIT_FILE)

# # train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
# # val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

# # # ============================================================
# # # MODEL + OPTIMIZER
# # # ============================================================

# # model = PoseDiffusion(num_joints=22, coord_dim=3, T=1000, device=DEVICE).to(DEVICE) 
# # optimizer = Adam(model.parameters(), lr=LR)
# # scheduler = CosineAnnealingLR(optimizer, T_max=N_EPOCHS)

# # # ============================================================
# # # METRICS
# # # ============================================================

# # def mpjpe(pred, target):
# #     """Mean per-joint position error."""
# #     return torch.norm(pred - target, dim=-1).mean()

# # def velocity_error(pred, target):
# #     """Mean velocity difference error."""
# #     vel_pred = pred[:, 1:] - pred[:, :-1]
# #     vel_target = target[:, 1:] - target[:, :-1]
# #     return torch.norm(vel_pred - vel_target, dim=-1).mean()

# # # ============================================================
# # # TRAINING LOOP
# # # ============================================================

# # def train_one_epoch(epoch):
# #     model.train()
# #     total_loss = 0.0
# #     for past_poses, future_poses, _, _ in tqdm(train_loader, desc=f"Train {epoch}"):
# #         past_poses = past_poses.to(DEVICE)
# #         future_poses = future_poses.to(DEVICE)
        
# #         loss = model.training_loss(future_poses, x_past=past_poses)

# #         optimizer.zero_grad()
# #         loss.backward()
# #         optimizer.step()

# #         total_loss += loss.item()

# #     avg_loss = total_loss / len(train_loader)
# #     print(f"[Epoch {epoch}] Train Loss: {avg_loss:.6f}")
# #     return avg_loss

# # @torch.no_grad()
# # def validate(epoch):
# #     model.eval()
# #     total_mpjpe = 0.0
# #     total_vel = 0.0
# #     n_batches = 0

# #     # Temporarily reduce diffusion steps for faster validation
# #     original_T = model.utils.T
# #     model.utils.T = VAL_DIFFUSION_STEPS

# #     for past_poses, future_poses, _, _ in tqdm(val_loader, desc=f"Val {epoch}"):
# #         past_poses = past_poses.to(DEVICE)
# #         future_poses = future_poses.to(DEVICE)
        
# #         B, T, J, C = future_poses.shape
# #         samples = model.sample(B, T, cond=past_poses, num_joints=J, coord_dim=C)

# #         total_mpjpe += mpjpe(samples, future_poses).item()
# #         total_vel += velocity_error(samples, future_poses).item()
# #         n_batches += 1

# #     # Restore original diffusion steps
# #     model.utils.T = original_T

# #     avg_mpjpe = total_mpjpe / n_batches
# #     avg_vel = total_vel / n_batches
# #     print(f"[Epoch {epoch}] Val MPJPE: {avg_mpjpe:.6f}, Velocity Error: {avg_vel:.6f}")
# #     return avg_mpjpe, avg_vel

# # # ============================================================
# # # CHECKPOINTING
# # # ============================================================

# # def save_checkpoint(epoch, model, optimizer, train_loss, val_mpjpe, val_vel):
# #     ckpt = {
# #         "epoch": epoch,
# #         "model_state_dict": model.state_dict(),
# #         "optimizer_state_dict": optimizer.state_dict(),
# #         "train_loss": train_loss,
# #         "val_mpjpe": val_mpjpe,
# #         "val_vel": val_vel
# #     }
# #     path = SAVE_DIR / f"epoch_{epoch:03d}.pt"
# #     torch.save(ckpt, path)
# #     print(f"[Checkpoint] Saved: {path}")

# # # ============================================================
# # # MAIN TRAINING SCRIPT
# # # ============================================================

# # def main():
# #     best_mpjpe = float("inf")

# #     for epoch in range(1, N_EPOCHS + 1):
# #         train_loss = train_one_epoch(epoch)
        
# #         # Validate every 2 epochs (except always validate epoch 1)
# #         if epoch % 2 == 0 or epoch == 1:
# #             val_mpjpe, val_vel = validate(epoch)
# #             save_checkpoint(epoch, model, optimizer, train_loss, val_mpjpe, val_vel)
            
# #             if val_mpjpe < best_mpjpe:
# #                 best_mpjpe = val_mpjpe
# #                 torch.save(model.state_dict(), SAVE_DIR / "best_model.pt")
# #                 print(f"[Best Model] Epoch {epoch} with MPJPE: {best_mpjpe:.6f}")
# #         else:
# #             # Save training checkpoint without validation
# #             save_checkpoint(epoch, model, optimizer, train_loss, 0.0, 0.0)

# #         scheduler.step()

# #     print("✅ Training complete.")

# # if __name__ == "__main__":
# #     main()


# from pathlib import Path
# import torch
# import torch.nn.functional as F
# from torch.utils.data import DataLoader
# from torch.optim import Adam
# from torch.optim.lr_scheduler import CosineAnnealingLR
# from torch.cuda.amp import autocast, GradScaler
# from tqdm import tqdm

# from data_loader import AMASSForecastDataset
# from diffusion_parallel import PoseDiffusion

# # ============================================================
# # CONFIG
# # ============================================================

# SPLIT_FILE = Path("data/AMASS/train.txt")
# VAL_SPLIT_FILE = Path("data/AMASS/val.txt")
# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# BATCH_SIZE = 32
# N_EPOCHS = 50
# LR = 1e-4
# VAL_DIFFUSION_STEPS = 30
# SAVE_DIR = Path("runs/latent_flow")
# SAVE_DIR.mkdir(parents=True, exist_ok=True)

# # ============================================================
# # DATASET
# # ============================================================

# train_dataset = AMASSForecastDataset(split_file=SPLIT_FILE)
# val_dataset = AMASSForecastDataset(split_file=VAL_SPLIT_FILE)

# train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
# val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

# # ============================================================
# # MODEL + OPTIMIZER + MIXED PRECISION
# # ============================================================

# model = PoseDiffusion(num_joints=22, coord_dim=3, T=1000, device=DEVICE).to(DEVICE) 
# optimizer = Adam(model.parameters(), lr=LR)
# scheduler = CosineAnnealingLR(optimizer, T_max=N_EPOCHS)
# scaler = GradScaler()

# # ============================================================
# # METRICS
# # ============================================================

# def mpjpe(pred, target):
#     return torch.norm(pred - target, dim=-1).mean()

# def velocity_error(pred, target):
#     vel_pred = pred[:, 1:] - pred[:, :-1]
#     vel_target = target[:, 1:] - target[:, :-1]
#     return torch.norm(vel_pred - vel_target, dim=-1).mean()

# # ============================================================
# # TRAINING LOOP
# # ============================================================

# def train_one_epoch(epoch):
#     model.train()
#     total_loss = 0.0
#     nan_count = 0
    
#     for past_poses, future_poses, _, _ in tqdm(train_loader, desc=f"Train {epoch}"):
#         past_poses = past_poses.to(DEVICE)
#         future_poses = future_poses.to(DEVICE)
        
#         optimizer.zero_grad()
        
#         with autocast():
#             loss = model.training_loss(future_poses, x_past=past_poses)
        
#         if torch.isnan(loss):
#             print(f"[WARNING] NaN loss detected, skipping batch")
#             nan_count += 1
#             continue
        
#         scaler.scale(loss).backward()
#         scaler.unscale_(optimizer)
#         torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
#         scaler.step(optimizer)
#         scaler.update()

#         total_loss += loss.item()

#     avg_loss = total_loss / max(1, len(train_loader) - nan_count)
#     print(f"[Epoch {epoch}] Train Loss: {avg_loss:.6f}, NaN batches: {nan_count}")
#     return avg_loss

# @torch.no_grad()
# def validate(epoch):
#     model.eval()
#     total_mpjpe = 0.0
#     total_vel = 0.0
#     n_batches = 0

#     original_T = model.utils.T
#     model.utils.T = VAL_DIFFUSION_STEPS

#     for past_poses, future_poses, _, _ in tqdm(val_loader, desc=f"Val {epoch}"):
#         past_poses = past_poses.to(DEVICE)
#         future_poses = future_poses.to(DEVICE)
        
#         B, T, J, C = future_poses.shape
        
#         with autocast():
#             samples = model.sample(B, T, cond=past_poses, num_joints=J, coord_dim=C)

#         total_mpjpe += mpjpe(samples, future_poses).item()
#         total_vel += velocity_error(samples, future_poses).item()
#         n_batches += 1

#     model.utils.T = original_T

#     avg_mpjpe = total_mpjpe / n_batches
#     avg_vel = total_vel / n_batches
#     print(f"[Epoch {epoch}] Val MPJPE: {avg_mpjpe:.6f}, Velocity Error: {avg_vel:.6f}")
#     return avg_mpjpe, avg_vel

# # ============================================================
# # CHECKPOINTING
# # ============================================================

# def save_checkpoint(epoch, model, optimizer, train_loss, val_mpjpe, val_vel):
#     ckpt = {
#         "epoch": epoch,
#         "model_state_dict": model.state_dict(),
#         "optimizer_state_dict": optimizer.state_dict(),
#         "train_loss": train_loss,
#         "val_mpjpe": val_mpjpe,
#         "val_vel": val_vel
#     }
#     path = SAVE_DIR / f"epoch_{epoch:03d}.pt"
#     torch.save(ckpt, path)
#     print(f"[Checkpoint] Saved: {path}")

# # ============================================================
# # MAIN TRAINING SCRIPT
# # ============================================================

# def main():
#     best_mpjpe = float("inf")

#     for epoch in range(1, N_EPOCHS + 1):
#         train_loss = train_one_epoch(epoch)
        
#         if epoch % 2 == 0 or epoch == 1:
#             val_mpjpe, val_vel = validate(epoch)
#             save_checkpoint(epoch, model, optimizer, train_loss, val_mpjpe, val_vel)
            
#             if val_mpjpe < best_mpjpe:
#                 best_mpjpe = val_mpjpe
#                 torch.save(model.state_dict(), SAVE_DIR / "best_model.pt")
#                 print(f"[Best Model] Epoch {epoch} with MPJPE: {best_mpjpe:.6f}")
#         else:
#             save_checkpoint(epoch, model, optimizer, train_loss, 0.0, 0.0)

#         scheduler.step()

#     print("✅ Training complete.")

# if __name__ == "__main__":
#     main()


from pathlib import Path
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import autocast, GradScaler
from tqdm import tqdm

from data_loader import AMASSForecastDataset
from diffusion_parallel import PoseDiffusion

# ============================================================
# CONFIG
# ============================================================

SPLIT_FILE = Path("data/AMASS/train.txt")
VAL_SPLIT_FILE = Path("data/AMASS/val.txt")
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32
N_EPOCHS = 50
LR = 1e-4
VAL_DIFFUSION_STEPS = 30
SAVE_DIR = Path("runs/latent_flow")
SAVE_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# DATASET
# ============================================================

train_dataset = AMASSForecastDataset(split_file=SPLIT_FILE)
val_dataset = AMASSForecastDataset(split_file=VAL_SPLIT_FILE)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

# ============================================================
# MODEL + OPTIMIZER + MIXED PRECISION
# ============================================================

model = PoseDiffusion(num_joints=22, coord_dim=3, T=1000, device=DEVICE).to(DEVICE) 
optimizer = Adam(model.parameters(), lr=LR)
scheduler = CosineAnnealingLR(optimizer, T_max=N_EPOCHS)
scaler = GradScaler()

# ============================================================
# METRICS
# ============================================================

def mpjpe(pred, target):
    return torch.norm(pred - target, dim=-1).mean()

def velocity_error(pred, target):
    vel_pred = pred[:, 1:] - pred[:, :-1]
    vel_target = target[:, 1:] - target[:, :-1]
    return torch.norm(vel_pred - vel_target, dim=-1).mean()

# ============================================================
# TRAINING LOOP
# ============================================================

def train_one_epoch(epoch):
    model.train()
    total_loss = 0.0
    nan_count = 0
    
    for past_poses, future_poses, _, _ in tqdm(train_loader, desc=f"Train {epoch}"):
        past_poses = past_poses.to(DEVICE)
        future_poses = future_poses.to(DEVICE)
        
        optimizer.zero_grad()
        
        with autocast():
            loss = model.training_loss(future_poses, x_past=past_poses)
        
        if torch.isnan(loss):
            print(f"[WARNING] NaN loss detected, skipping batch")
            nan_count += 1
            continue
        
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        total_loss += loss.item()

    avg_loss = total_loss / max(1, len(train_loader) - nan_count)
    print(f"[Epoch {epoch}] Train Loss: {avg_loss:.6f}, NaN batches: {nan_count}")
    return avg_loss

@torch.no_grad()
def validate(epoch):
    model.eval()
    total_mpjpe = 0.0
    total_vel = 0.0
    n_batches = 0

    original_T = model.utils.T
    model.utils.T = VAL_DIFFUSION_STEPS

    for past_poses, future_poses, _, _ in tqdm(val_loader, desc=f"Val {epoch}"):
        past_poses = past_poses.to(DEVICE)
        future_poses = future_poses.to(DEVICE)
        
        B, T, J, C = future_poses.shape
        
        with autocast():
            samples = model.sample(B, T, cond=past_poses, num_joints=J, coord_dim=C)

        total_mpjpe += mpjpe(samples, future_poses).item()
        total_vel += velocity_error(samples, future_poses).item()
        n_batches += 1

    model.utils.T = original_T

    avg_mpjpe = total_mpjpe / n_batches
    avg_vel = total_vel / n_batches
    print(f"[Epoch {epoch}] Val MPJPE: {avg_mpjpe:.6f}, Velocity Error: {avg_vel:.6f}")
    return avg_mpjpe, avg_vel

# ============================================================
# CHECKPOINTING
# ============================================================

def save_checkpoint(epoch, model, optimizer, train_loss, val_mpjpe, val_vel):
    ckpt = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "train_loss": train_loss,
        "val_mpjpe": val_mpjpe,
        "val_vel": val_vel
    }
    path = SAVE_DIR / f"epoch_{epoch:03d}.pt"
    torch.save(ckpt, path)
    print(f"[Checkpoint] Saved: {path}")

# ============================================================
# MAIN TRAINING SCRIPT
# ============================================================

def main():
    best_mpjpe = float("inf")

    for epoch in range(1, N_EPOCHS + 1):
        train_loss = train_one_epoch(epoch)
        
        if epoch % 2 == 0 or epoch == 1:
            val_mpjpe, val_vel = validate(epoch)
            save_checkpoint(epoch, model, optimizer, train_loss, val_mpjpe, val_vel)
            
            if val_mpjpe < best_mpjpe:
                best_mpjpe = val_mpjpe
                torch.save(model.state_dict(), SAVE_DIR / "best_model.pt")
                print(f"[Best Model] Epoch {epoch} with MPJPE: {best_mpjpe:.6f}")
        else:
            save_checkpoint(epoch, model, optimizer, train_loss, 0.0, 0.0)

        scheduler.step()

    print("✅ Training complete.")

if __name__ == "__main__":
    main()