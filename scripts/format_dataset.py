from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm
import smplx

RAW_DIR = Path("data/raw")
OUT_DIR = Path("data/AMASS/npz")
OUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_FPS = 25
JOINTS_TO_KEEP = 22
DEVICE = torch.device("cpu")

MODEL_DIR = Path("smplh1/smplh")


def downsample(x, factor):
    return x[::factor] if factor > 0 else x


def load_smplh_model(gender: str):
    gender = gender.lower()
    if gender.startswith("m"):
        model_file = MODEL_DIR / "SMPLH_MALE.pkl"
        gender = "male"
    elif gender.startswith("f"):
        model_file = MODEL_DIR / "SMPLH_FEMALE.pkl"
        gender = "female"
    else:
        model_file = MODEL_DIR / "SMPLH_MALE.pkl"
        gender = "male"

    if not model_file.exists():
        raise FileNotFoundError(f"Missing SMPL-H model file: {model_file}")

    return smplx.create(
        MODEL_DIR.parent,
        model_type="smplh",
        gender=gender,
        use_pca=False,
        batch_size=1
    ).to(DEVICE)


def process_file(npz_path: Path):
    try:
        data = np.load(npz_path, allow_pickle=True)
        gender = data["gender"].item().lower() if "gender" in data else "male"
        fps = int(data["mocap_framerate"].item())

        poses = torch.from_numpy(data["poses"]).float()      #[T, 156]
        trans = torch.from_numpy(data["trans"]).float()      # [T, 3]

        pose_body = poses[:, :66]

        factor = round(fps / TARGET_FPS)
        pose_body = downsample(pose_body, factor)
        trans = downsample(trans, factor)

        T = min(pose_body.shape[0], trans.shape[0])
        pose_body = pose_body[:T]
        trans = trans[:T]

        global_orient = pose_body[:, :3]
        body_pose = pose_body[:, 3:]

        left_hand_pose = torch.zeros(T, 45, device=DEVICE)
        right_hand_pose = torch.zeros(T, 45, device=DEVICE)

        model = load_smplh_model(gender)
        num_betas = model.num_betas if hasattr(model, "num_betas") else 10

        betas_np = data["betas"] if "betas" in data else np.zeros(num_betas, dtype=np.float32)
        betas_np = np.asarray(betas_np, dtype=np.float32)
        if betas_np.shape[0] > num_betas:
            betas_np = betas_np[:num_betas]
        elif betas_np.shape[0] < num_betas:
            betas_np = np.pad(betas_np, (0, num_betas - betas_np.shape[0]))
        betas = torch.from_numpy(betas_np).float().unsqueeze(0).expand(T, -1).to(DEVICE)

        with torch.no_grad():
            output = model(
                transl=trans,
                global_orient=global_orient,
                body_pose=body_pose,
                left_hand_pose=left_hand_pose,
                right_hand_pose=right_hand_pose,
                betas=betas
            )
            joints = output.joints[:, :JOINTS_TO_KEEP, :]

        return {
            "poses": joints.cpu().numpy(),   # [T, 22, 3]
            "trans": trans.cpu().numpy(),    # [T, 3]
            "betas": betas_np,
            "gender": gender
        }

    except Exception as e:
        print(f"Skipping {npz_path.name}: {e}")
        return None


def save_npz(out_path: Path, data_dict: dict):
    np.savez_compressed(out_path, **data_dict)


pose_files = list(RAW_DIR.rglob("*_poses.npz"))
print(f"Found {len(pose_files)} pose files")

for path in tqdm(pose_files):
    result = process_file(path)
    if result is not None:
        dataset = path.parts[2]
        stem = path.stem.replace("_poses", "")
        out_path = OUT_DIR / f"{dataset}_{stem}.npz"
        save_npz(out_path, result)

print(f"Finished. Processed files saved to {OUT_DIR}")
