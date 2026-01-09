import sys
from pathlib import Path
import numpy as np
import torch
torch.set_float32_matmul_precision("high")
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent))

from scripts.data_loader import AMASSForecastDataset, collate_fn
from models.baseline import PoseGRU
from models.transformer import rot6d_to_rotmat, rotmat_to_rot6d


SMPL_PARENTS = [
    -1, 0, 0, 0,
    1, 2, 3,
    4, 5, 6,
    7, 8, 9,
    9, 9,
    12,
    13, 14,
    16, 17,
    18, 19,
]

BONE_LENGTHS = torch.tensor([
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


def forward_kinematics(rotmats, root_pos):
    B, T, J, _, _ = rotmats.shape
    device = rotmats.device

    positions = torch.zeros(B, T, J, 3, device=device)
    global_rot = torch.zeros(B, T, J, 3, 3, device=device)

    positions[:, :, 0] = root_pos
    global_rot[:, :, 0] = rotmats[:, :, 0]

    bone_lengths = BONE_LENGTHS.to(device)

    for j in range(1, J):
        p = SMPL_PARENTS[j]
        global_rot[:, :, j] = global_rot[:, :, p] @ rotmats[:, :, j]
        bone = torch.tensor([0, bone_lengths[j], 0], device=device)
        bone_world = global_rot[:, :, p] @ bone
        positions[:, :, j] = positions[:, :, p] + bone_world

    return positions


CONFIG = {
    "input_frames": 25,
    "output_frames": 5,
    "num_joints": 22,
    "stride_train": 10,
    "stride_val": 10,

    "hidden_dim": 256,
    "num_layers": 2,
    "dropout": 0.1,

    "batch_size": 24,
    "learning_rate": 1e-4,
    "num_epochs": 20,
    "weight_decay": 1e-4,

    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "checkpoint_dir": "checkpoints/baseline_gru",
    "validate_every": 2,
    "val_max_batches": 5,
    "mixed_precision": True,
    "num_workers": 2,
    "pin_memory": True,

    "val_use_gt_root": False,
}


def prepare_batch(batch, device):
    past_pose = batch["past_pose"].to(device)
    future_pose = batch["future_pose"].to(device)
    past_root = batch["past_root"].to(device)
    future_root = batch["future_root"].to(device)

    B, T_in, J, _ = past_pose.shape
    T_out = future_pose.shape[1]

    past_pose_6d = rotmat_to_rot6d(
        past_pose.view(B, T_in, J, 3, 3)
    )
    future_pose_6d = rotmat_to_rot6d(
        future_pose.view(B, T_out, J, 3, 3)
    )

    return past_pose_6d, future_pose_6d, past_root, future_root

def train_epoch(model, loader, optimizer, scaler, device, epoch):
    model.train()
    total = 0.0

    pbar = tqdm(loader, desc=f"Epoch {epoch}")
    for batch in pbar:
        past_pose, future_pose, past_root, future_root = prepare_batch(batch, device)

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast("cuda", enabled=CONFIG["mixed_precision"]):
            pred_pose, pred_root = model(past_pose, past_root)
            loss = (
                F.mse_loss(pred_pose, future_pose)
                + 0.1 * F.mse_loss(pred_root, future_root)
            )

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        total += loss.item()
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    return total / len(loader)


@torch.no_grad()
def validate(model, loader, device):
    model.eval()
    mpjpe_list = []

    for i, batch in enumerate(loader):
        if i >= CONFIG["val_max_batches"]:
            break

        past_pose, future_pose, past_root, future_root = prepare_batch(batch, device)
        pred_pose, pred_root = model(past_pose, past_root)

        pred_rot = rot6d_to_rotmat(pred_pose)
        gt_rot = rot6d_to_rotmat(future_pose)

        if CONFIG["val_use_gt_root"]:
            pred_pos = forward_kinematics(pred_rot, future_root)
            gt_pos = forward_kinematics(gt_rot, future_root)
        else:
            pred_pos = forward_kinematics(pred_rot, pred_root)
            gt_pos = forward_kinematics(gt_rot, future_root)

        mpjpe = torch.norm(pred_pos - gt_pos, dim=-1).mean() * 1000
        mpjpe_list.append(mpjpe.item())

    return float(np.mean(mpjpe_list))


def main():
    device = torch.device(CONFIG["device"])
    ckpt_dir = Path(CONFIG["checkpoint_dir"])
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    train_ds = AMASSForecastDataset(
        "data/AMASS/train.txt",
        CONFIG["input_frames"],
        CONFIG["output_frames"],
        CONFIG["stride_train"],
        normalize_orientation=True,
    )
    val_ds = AMASSForecastDataset(
        "data/AMASS/val.txt",
        CONFIG["input_frames"],
        CONFIG["output_frames"],
        CONFIG["stride_val"],
        normalize_orientation=True,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=CONFIG["batch_size"],
        shuffle=True,
        num_workers=CONFIG["num_workers"],
        collate_fn=collate_fn,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=CONFIG["batch_size"],
        shuffle=False,
        num_workers=CONFIG["num_workers"],
        collate_fn=collate_fn,
    )

    model = PoseGRU(
        num_joints=CONFIG["num_joints"],
        input_dim=CONFIG["num_joints"] * 6,
        hidden_dim=CONFIG["hidden_dim"],
        num_layers=CONFIG["num_layers"],
        output_frames=CONFIG["output_frames"],
        dropout=CONFIG["dropout"],
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=CONFIG["learning_rate"],
        weight_decay=CONFIG["weight_decay"],
    )
    scaler = torch.amp.GradScaler(enabled=CONFIG["mixed_precision"])

    best = float("inf")

    for epoch in range(1, CONFIG["num_epochs"] + 1):
        train_epoch(model, train_loader, optimizer, scaler, device, epoch)

        if epoch % CONFIG["validate_every"] == 0:
            mpjpe = validate(model, val_loader, device)
            print(f"Epoch {epoch} | MPJPE: {mpjpe:.2f} mm")

            if mpjpe < best:
                best = mpjpe
                torch.save(model.state_dict(), ckpt_dir / "best_model.pth")

    print(f"\nBest GRU MPJPE: {best:.2f} mm")


if __name__ == "__main__":
    main()
