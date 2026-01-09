from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm
import smplx
from scipy.spatial.transform import Rotation as R

IN_DIR = Path("data/raw")
OUT_DIR = Path("data/AMASS/npz")
OUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_FPS = 25
JOINTS_TO_KEEP = 22
DEVICE = torch.device("cpu")
MODEL_DIR = Path("smplh1/smplh")

SMPL_MODELS = {}

MAX_BATCH_SIZE = 512

R_Z_TO_Y = np.array([
    [1,  0,  0],
    [0,  0, -1],
    [0,  1,  0]
], dtype=np.float32)


def downsample(x, factor):
    return x[::factor] if factor > 0 else x


def axis_angle_to_rot_mat(axis_angle):
    original_shape = axis_angle.shape
    
    if axis_angle.ndim == 2 and axis_angle.shape[1] == 3:
        axis_angle_flat = axis_angle
    elif axis_angle.ndim == 2:
        T, dim = axis_angle.shape
        J = dim // 3
        axis_angle_flat = axis_angle.reshape(T * J, 3)
    else:
        raise ValueError(f"Unexpected axis_angle shape: {axis_angle.shape}")
    
    rotmats = R.from_rotvec(axis_angle_flat).as_matrix()  # (N, 3, 3)
    
    if axis_angle.ndim == 2 and axis_angle.shape[1] > 3:
        T = original_shape[0]
        J = original_shape[1] // 3
        rotmats = rotmats.reshape(T, J, 3, 3)
    
    return rotmats.astype(np.float32)


def rotate_z_to_y_rotmat(rotmat):
    T, J = rotmat.shape[:2]
    rotmat_flat = rotmat.reshape(-1, 3, 3)
    
    rotated = R_Z_TO_Y @ rotmat_flat @ R_Z_TO_Y.T
    
    return rotated.reshape(T, J, 3, 3)


def load_smplh_model(gender: str):
    gender = gender.lower()
    if gender.startswith("m"):
        gender = "male"
    elif gender.startswith("f"):
        gender = "female"
    else:
        gender = "male"
    
    if gender not in SMPL_MODELS:
        model_file = MODEL_DIR / f"SMPLH_{gender.upper()}.pkl"
        if not model_file.exists():
            raise FileNotFoundError(f"Missing SMPL-H model file: {model_file}")
        
        SMPL_MODELS[gender] = smplx.create(
            MODEL_DIR.parent,
            model_type="smplh",
            gender=gender,
            use_pca=False,
            batch_size=MAX_BATCH_SIZE
        ).to(DEVICE)
    
    return SMPL_MODELS[gender]


def process_smpl_chunked(model, trans, global_orient, body_pose, betas, chunk_size=MAX_BATCH_SIZE):
    T = trans.shape[0]
    all_joints = []
    
    betas_expanded = betas.expand(T, -1)
    
    for start_idx in range(0, T, chunk_size):
        end_idx = min(start_idx + chunk_size, T)
        chunk_T = end_idx - start_idx
        
        trans_chunk = trans[start_idx:end_idx]
        global_orient_chunk = global_orient[start_idx:end_idx]
        body_pose_chunk = body_pose[start_idx:end_idx]
        betas_chunk = betas_expanded[start_idx:end_idx]
        
        left_hand_pose = torch.zeros(chunk_T, 45, device=DEVICE)
        right_hand_pose = torch.zeros(chunk_T, 45, device=DEVICE)
        
        with torch.no_grad():
            output = model(
                transl=trans_chunk,
                global_orient=global_orient_chunk,
                body_pose=body_pose_chunk,
                left_hand_pose=left_hand_pose,
                right_hand_pose=right_hand_pose,
                betas=betas_chunk
            )
            joints_chunk = output.joints[:, :JOINTS_TO_KEEP, :]
            all_joints.append(joints_chunk)
    
    return torch.cat(all_joints, dim=0)


def process_file(npz_path: Path):
    try:
        data = np.load(npz_path, allow_pickle=True)
        gender = data["gender"].item().lower() if "gender" in data else "male"
        fps = int(data["mocap_framerate"].item())

        poses = torch.from_numpy(data["poses"]).float()
        trans = torch.from_numpy(data["trans"]).float()
        pose_body = poses[:, :66]

        factor = int(round(fps / TARGET_FPS))
        factor = max(1, factor)
        pose_body = downsample(pose_body, factor)
        trans = downsample(trans, factor)

        T = min(pose_body.shape[0], trans.shape[0])
        pose_body = pose_body[:T]
        trans = trans[:T]

        global_orient = pose_body[:, :3]
        body_pose = pose_body[:, 3:]

        global_orient_rotmat = axis_angle_to_rot_mat(
            global_orient.cpu().numpy()
        )  
        
        body_pose_rotmat = axis_angle_to_rot_mat(
            body_pose.cpu().numpy()
        )  
        
        joint_rotmats = np.concatenate([
            global_orient_rotmat[:, None, :, :],  
            body_pose_rotmat                      
        ], axis=1)
        
        joint_rotmats = rotate_z_to_y_rotmat(joint_rotmats)
        
        joint_rotmats_flat = joint_rotmats.reshape(T, JOINTS_TO_KEEP, 9)

        model = load_smplh_model(gender)
        num_betas = model.num_betas if hasattr(model, "num_betas") else 10

        betas_np = data["betas"] if "betas" in data else np.zeros(num_betas, dtype=np.float32)
        betas_np = np.asarray(betas_np, dtype=np.float32)
        if betas_np.shape[0] > num_betas:
            betas_np = betas_np[:num_betas]
        elif betas_np.shape[0] < num_betas:
            betas_np = np.pad(betas_np, (0, num_betas - betas_np.shape[0]))
        
        betas = torch.from_numpy(betas_np).float().unsqueeze(0).to(DEVICE)
        trans = trans.to(DEVICE)
        global_orient = global_orient.to(DEVICE)
        body_pose = body_pose.to(DEVICE)
        
        joints = process_smpl_chunked(
            model, trans, global_orient, body_pose, betas, chunk_size=MAX_BATCH_SIZE
        )
        joints = joints.cpu().numpy()
        
        joints = joints @ R_Z_TO_Y.T
        root_pos = joints[:, 0, :]

        return {
            "poses": joint_rotmats_flat,   
            "trans": root_pos,             
            "betas": betas_np,
            "gender": gender
        }

    except Exception as e:
        print(f"Skipping {npz_path.name}: {e}")
        return None


def save_npz(out_path: Path, data_dict: dict):
    np.savez_compressed(out_path, **data_dict)


pose_files = list(IN_DIR.rglob("*_poses.npz"))
print(f"Total: {len(pose_files)} pose files")

for path in tqdm(pose_files):
    result = process_file(path)
    if result is not None:
        dataset = path.parents[1].name
        stem = path.stem.replace("_poses", "")
        out_path = OUT_DIR / f"{dataset}_{stem}.npz"
        save_npz(out_path, result)

print(f"Files saved to {OUT_DIR}")