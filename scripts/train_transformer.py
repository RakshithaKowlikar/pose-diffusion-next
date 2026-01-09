import sys
import time
from pathlib import Path
import numpy as np
import torch
torch.set_float32_matmul_precision("high")
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent))

from data_loader import AMASSForecastDataset, collate_fn
from models.transformer import STTransformer, rot6d_to_rotmat, rotmat_to_rot6d


SMPL_PARENTS = [
    -1,  
    0, 0, 0,  
    1, 2, 3,  
    4, 5, 6, 
    7, 8, 9,  
    9, 9,     
    12,       
    13, 14,   
    16, 17,   
    18, 19,   
]

BONE_LENS = torch.tensor([
    0.0,    
    0.1, 0.1, 0.1,
    0.4, 0.4, 0.15,
    0.4, 0.4, 0.15,
    0.05, 0.05, 0.15,
    0.1, 0.1,
    0.1,
    0.15, 0.15,
    0.25, 0.25,
    0.2, 0.2,
], dtype=torch.float32)


def fk(rotmats, root_pos, parents=SMPL_PARENTS, bone_lens=BONE_LENS):
    B, T, J = rotmats.shape[:3]
    dev = rotmats.device
    
    pos = torch.zeros(B, T, J, 3, device=dev)
    glob_rot = torch.zeros(B, T, J, 3, 3, device=dev)
    
    pos[:, :, 0, :] = root_pos
    glob_rot[:, :, 0] = rotmats[:, :, 0]
    
    bone_lens = bone_lens.to(dev)
    
    for j in range(1, J):
        p = parents[j]
        if p >= 0:
            glob_rot[:, :, j] = glob_rot[:, :, p] @ rotmats[:, :, j]
            bone_vec = torch.tensor([0, bone_lens[j], 0], device=dev)
            bone_world = (glob_rot[:, :, p] @ bone_vec.unsqueeze(-1)).squeeze(-1)
            pos[:, :, j, :] = pos[:, :, p, :] + bone_world
    
    return pos


cfg = {
    "input_frames": 25,
    "output_frames": 5, 
    "num_joints": 22,
    "stride_train": 10,
    "stride_val": 10,

    "d_model": 128,
    "num_layers": 8,
    "num_heads": 8,
    "dim_feedforward": 256,
    "dropout": 0.1,

    "batch_size": 24, 
    "lr": 1e-4,
    "epochs": 20, 
    "warmup_epochs": 2, 
    "weight_decay": 1e-4,
    
    "sched_sampling": True,
    "ss_start": 6,
    "ss_ramp": 8,    
    "ss_min": 0.1,

    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "ckpt_dir": "checkpoints/st_transformer",
    "save_every": 10,
    "val_every": 2,
    "val_batches": 5,
    "amp": True,
    "workers": 2, 
    "pin_mem": True,
    "grad_ckpt": True,
    
    "val_gt_root": False,
}


def get_lr_schedule(opt, warmup, total):
    dim = cfg["d_model"]
    
    def lr_fn(step):
        step = max(1, step)
        a = step ** -0.5
        b = step * (warmup ** -1.5)
        return (dim ** -0.5) * min(a, b)
    
    return torch.optim.lr_scheduler.LambdaLR(opt, lr_fn)


def prep_batch(batch, dev):
    past_p = batch["past_pose"].to(dev, non_blocking=True)
    fut_p = batch["future_pose"].to(dev, non_blocking=True)
    past_r = batch["past_root"].to(dev, non_blocking=True)
    fut_r = batch["future_root"].to(dev, non_blocking=True)
    
    B, T_in, J = past_p.shape[:3]
    T_out = fut_p.shape[1]
    
    past_3x3 = past_p.view(B, T_in, J, 3, 3)
    fut_3x3 = fut_p.view(B, T_out, J, 3, 3)
    
    past_6d = rotmat_to_rot6d(past_3x3)
    fut_6d = rotmat_to_rot6d(fut_3x3)
    
    return past_6d, fut_6d, past_r, fut_r, past_3x3, fut_3x3


def train_epoch(model, loader, opt, sched, scaler, dev, ep):
    model.train()
    total = 0.0
    
    if cfg["sched_sampling"] and ep >= cfg["ss_start"]:
        prog = min(1.0, (ep - cfg["ss_start"]) / cfg["ss_ramp"])
        tf_prob = max(cfg["ss_min"], 1.0 - prog)
    else:
        tf_prob = 1.0

    pbar = tqdm(loader, desc=f"Epoch {ep} [TF={tf_prob:.2f}]")
    for batch in pbar:
        past_6d, fut_6d, past_r, fut_r, _, _ = prep_batch(batch, dev)
        
        B, T_in, J = past_6d.shape[:3]
        T_out = fut_6d.shape[1]
        
        opt.zero_grad(set_to_none=True)
        
        losses = []
        inp_p = past_6d.clone()
        inp_r = past_r.clone()
        
        with torch.amp.autocast("cuda", enabled=cfg["amp"]):
            for t in range(T_out):
                pred_p, pred_r = model.predict_next(inp_p, inp_r)
                
                tgt_p = fut_6d[:, t, :, :]
                tgt_r = fut_r[:, t, :]
                
                l_p = F.mse_loss(pred_p, tgt_p)
                l_r = F.mse_loss(pred_r, tgt_r)
                losses.append(l_p + 0.1 * l_r)
                
                use_gt = torch.rand(B, device=dev) < tf_prob 
                
                next_p = torch.where(use_gt[:, None, None], tgt_p, pred_p.detach())
                next_r = torch.where(use_gt[:, None], tgt_r, pred_r.detach())
                
                inp_p = torch.cat([inp_p, next_p.unsqueeze(1)], dim=1)
                inp_r = torch.cat([inp_r, next_r.unsqueeze(1)], dim=1)
            
            loss = torch.stack(losses).mean()
        
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        
        if sched:
            sched.step()
        
        total += loss.item()
        pbar.set_postfix(loss=f"{loss.item():.4f}", lr=f"{opt.param_groups[0]['lr']:.6f}")
    
    return total / len(loader)


@torch.no_grad()
def validate(model, loader, dev, max_b=20):
    model.eval()
    mpjpe_vals, fde_vals = [], []
    
    gt_root = cfg["val_gt_root"]
    metric = "Pose-only MPJPE" if gt_root else "Full motion MPJPE"
    
    for i, batch in enumerate(tqdm(loader, desc=f"Val [{metric}]", total=max_b)):
        if i >= max_b:
            break
        
        past_6d, fut_6d, past_r, fut_r, _, fut_3x3 = prep_batch(batch, dev)
        
        B, T_in, J = past_6d.shape[:3]
        T_out = fut_6d.shape[1]
        
        pred_6d, pred_r = model.forward(past_6d, past_r, T_out)
        pred_3x3 = rot6d_to_rotmat(pred_6d)
        
        if gt_root:
            pred_pos = fk(pred_3x3, fut_r)
            gt_pos = fk(fut_3x3, fut_r)
        else:
            pred_pos = fk(pred_3x3, pred_r)
            gt_pos = fk(fut_3x3, fut_r)
        
        mpjpe = torch.norm(pred_pos - gt_pos, dim=-1).mean() * 1000
        mpjpe_vals.append(mpjpe.item())
        
        fde = torch.norm(pred_pos[:, -1] - gt_pos[:, -1], dim=-1).mean() * 1000
        fde_vals.append(fde.item())
    
    return {
        "mpjpe": float(np.mean(mpjpe_vals)),
        "fde": float(np.mean(fde_vals)),
        "type": metric,
    }


def main():
    dev = torch.device(cfg["device"])
    ckpt_dir = Path(cfg["ckpt_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    
    train_ds = AMASSForecastDataset(
        split_file="data/AMASS/train.txt",
        input_frames=cfg["input_frames"],
        output_frames=cfg["output_frames"],
        stride=cfg["stride_train"],
        normalize_orientation=True,
    )
    val_ds = AMASSForecastDataset(
        split_file="data/AMASS/val.txt",
        input_frames=cfg["input_frames"],
        output_frames=cfg["output_frames"],
        stride=cfg["stride_val"],
        normalize_orientation=True,
    )
    
    train_loader = DataLoader(
        train_ds, batch_size=cfg["batch_size"], shuffle=True,
        num_workers=cfg["workers"], pin_memory=cfg["pin_mem"],
        collate_fn=collate_fn, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg["batch_size"], shuffle=False,
        num_workers=cfg["workers"], pin_memory=cfg["pin_mem"],
        collate_fn=collate_fn, drop_last=False,
    )
    
    model = STTransformer(
        in_frames=cfg["input_frames"],
        nj=cfg["num_joints"],
        dim=cfg["d_model"],
        nlayers=cfg["num_layers"],
        nhead=cfg["num_heads"],
        dff=cfg["dim_feedforward"],
        dropout=cfg["dropout"],
        grad_ckpt=cfg["grad_ckpt"],
    ).to(dev)
    
    opt = torch.optim.AdamW(
        model.parameters(), lr=cfg["lr"],
        weight_decay=cfg["weight_decay"],
        betas=(0.9, 0.98), eps=1e-9,
    )
    
    steps_per_ep = len(train_loader)
    total_steps = steps_per_ep * cfg["epochs"]
    warmup = steps_per_ep * cfg["warmup_epochs"]
    
    sched = get_lr_schedule(opt, warmup, total_steps)
    scaler = torch.amp.GradScaler(enabled=cfg["amp"])
    
    best = float("inf")
    t0 = time.time()
    
    for ep in range(1, cfg["epochs"] + 1):
        t_ep = time.time()
        
        tr_loss = train_epoch(model, train_loader, opt, sched, scaler, dev, ep)
        
        elapsed = time.time() - t_ep
        total_t = time.time() - t0
        eta = (cfg["epochs"] - ep) * (elapsed / 60.0)
        
        print(f"\nEpoch {ep}/{cfg['epochs']} | loss={tr_loss:.4f}")
        print(f"Time: {elapsed/60:.1f}m | Total: {total_t/3600:.2f}h | ETA: {eta:.0f}m")
        
        if ep % cfg["val_every"] == 0 or ep == cfg["epochs"]:
            metrics = validate(model, val_loader, dev, cfg["val_batches"])
            print(f"{metrics['type']}: {metrics['mpjpe']:.2f}mm | FDE: {metrics['fde']:.2f}mm")
            
            if metrics["mpjpe"] < best:
                best = metrics["mpjpe"]
                torch.save({
                    "epoch": ep,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": opt.state_dict(),
                    "scheduler_state_dict": sched.state_dict(),
                    "config": cfg,
                }, ckpt_dir / "best_model.pth")
                print("Saved best_model.pth")
        
        if ep % cfg["save_every"] == 0:
            torch.save({
                "epoch": ep,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": opt.state_dict(),
                "scheduler_state_dict": sched.state_dict(),
                "config": cfg,
            }, ckpt_dir / f"checkpoint_epoch_{ep}.pth")
            print(f"Saved checkpoint_epoch_{ep}.pth") 
    
    print(f"\nBest {metrics['type']}: {best:.2f}mm")
    print(f"Checkpoints: {ckpt_dir}")

if __name__ == "__main__":
    main()