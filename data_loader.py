from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from collections import OrderedDict

def _safe_norm(v, eps=1e-8):
    n = np.linalg.norm(v)
    return n if n > eps else eps

def _yaw_R_from_frame(frame):
    pelvis = frame[0]
    left_hip = frame[1]
    right_hip = frame[2]
    up_ref = frame[12] if frame.shape[0] > 12 else frame[3]

    right = right_hip - left_hip
    up = up_ref - pelvis
    fwd = np.cross(up, right)
    fwd[1] = 0.0
    n = _safe_norm(fwd)
    fwd = fwd / n

    if n < 1e-6:
        return np.eye(3, dtype=np.float32)

    yaw = np.arctan2(fwd[0], fwd[2])
    cy, sy = np.cos(-yaw), np.sin(-yaw)
    R = np.array([[cy, 0.0, sy],
                  [0.0, 1.0, 0.0],
                  [-sy, 0.0, cy]], dtype=np.float32)
    return R

class AMASSForecastDataset(Dataset):
    def __init__(self,
                 split_file,
                 input_frames=8,
                 output_frames=16,
                 stride=1,
                 normalize=True,
                 min_length=None,
                 normalize_orientation=True,
                 trans_mode="root",
                 cache_max_items=256):
        """
        Args:
            split_file (str): path to train.txt / val.txt / test.txt
            input_frames (int): number of past frames
            output_frames (int): number of future frames
            stride (int): sliding window step
            normalize (bool): if True, root-center poses per frame
            min_length (int): override minimum T
            normalize_orientation (bool): if True, rotate so facing +Z
            trans_mode (str): 'root' = trajectory relative to t0, 'global' = rotated global
            cache_max_items (int): max npz files to cache in memory
        """
        self.paths = [Path(line.strip()) for line in Path(split_file).read_text().splitlines()]
        self.input_len = input_frames
        self.output_len = output_frames
        self.total_len = input_frames + output_frames
        self.stride = stride
        self.normalize = normalize
        self.min_length = min_length or self.total_len
        self.normalize_orientation = normalize_orientation
        self.trans_mode = trans_mode
        self.cache_max_items = cache_max_items

        self._cache = OrderedDict()
        self._warned = False
        self.samples = []

        for path in self.paths:
            data = np.load(path, allow_pickle=True)
            poses = data["poses"]
            T = poses.shape[0]
            if T < self.min_length:
                continue
            for t in range(0, T - self.total_len + 1, self.stride):
                self.samples.append((path, t))

    def __len__(self):
        return len(self.samples)

    def _load_cached(self, path: Path):
        key = str(path)
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        data = np.load(path, allow_pickle=True)
        poses = data["poses"].astype(np.float32)
        trans = data["trans"].astype(np.float32)
        self._cache[key] = (poses, trans)
        if len(self._cache) > self.cache_max_items:
            self._cache.popitem(last=False)
        return poses, trans

    def __getitem__(self, idx):
        path, t = self.samples[idx]
        poses, trans = self._load_cached(path)

        pose_clip = poses[t:t + self.total_len].copy()    # [L, 22, 3]
        trans_clip = trans[t:t + self.total_len].copy()   # [L, 3]

        if not self._warned:
            self._warned = True
            dist = np.linalg.norm(pose_clip[0, 0] - trans_clip[0])
            print(f"Pelvis index=0, distance to trans: {dist:.3f} m")
            if dist > 0.5:
                print(f"[WARNING] Pelvis and trans offset large ({dist:.2f} m) in {path.name}")

        if self.normalize_orientation:
            R = _yaw_R_from_frame(pose_clip[0])
            pose_clip = np.einsum('tjc,cd->tjd', pose_clip, R)
            trans_clip = trans_clip @ R  
        root_global = pose_clip[:, 0, :]  # [L, 3]

        if self.trans_mode == "root":
            out_trans = root_global - root_global[0:1, :]
        else:
            out_trans = trans_clip

        if self.normalize:
            pose_clip = pose_clip - root_global[:, None, :]

        past_poses = pose_clip[:self.input_len]
        future_poses = pose_clip[self.input_len:]
        past_trans = out_trans[:self.input_len]
        future_trans = out_trans[self.input_len:]

        return (
            torch.from_numpy(past_poses),
            torch.from_numpy(future_poses),
            torch.from_numpy(past_trans),
            torch.from_numpy(future_trans)
        )
