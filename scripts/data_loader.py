from pathlib import Path
import numpy as np
import torch
from torch.utils.data import Dataset
from collections import OrderedDict

class AMASSForecastDataset(Dataset):
    def __init__(self, split_file, input_frames=8, output_frames=16, stride=1, min_length=None, normalize_orientation=True, cache_max_items=256):
        
        self.paths = [Path(line.strip()) for line in Path(split_file).read_text().splitlines()]
        self.input_len = input_frames
        self.output_len = output_frames
        self.total_len = input_frames + output_frames
        self.stride = stride
        self.min_length = min_length or self.total_len
        self.normalize_orientation = normalize_orientation
        self.cache_max_items = cache_max_items

        self._cache = OrderedDict()
        self.samples = []

        for path in self.paths:
            data = np.load(path, allow_pickle=True)
            poses = data["poses"]  
            T = poses.shape[0]
            if T < self.min_length:
                continue
            assert poses.shape[1:] == (22, 9), f"Expected (T, 22, 9), got {poses.shape}"
            
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
        poses = data["poses"].astype(np.float32, copy=False)  
        trans = data["trans"].astype(np.float32, copy=False)  
        self._cache[key] = (poses, trans)
        if len(self._cache) > self.cache_max_items:
            self._cache.popitem(last=False)
        return poses, trans

    def __getitem__(self, idx):
        path, t = self.samples[idx]
        poses, trans = self._load_cached(path)

        pose_clip = poses[t:t + self.total_len].copy()   
        trans_clip = trans[t:t + self.total_len].copy()   

        if self.normalize_orientation:
            root_R = pose_clip[0, 0].reshape(3, 3)
            
            yaw = np.arctan2(root_R[0, 2], root_R[2, 2])
            cy, sy = np.cos(-yaw), np.sin(-yaw)
            R_yaw = np.array([[cy, 0, sy],
                              [0,  1, 0 ],
                              [-sy, 0, cy]], dtype=np.float32)
            
            trans_clip = trans_clip @ R_yaw.T
            
            trans_clip = trans_clip - trans_clip[0:1]
            
            pose_clip_3x3 = pose_clip.reshape(-1, 22, 3, 3)
            R_yaw_exp = R_yaw[None, None, :, :]  
            pose_clip_3x3 = R_yaw_exp @ pose_clip_3x3
            pose_clip = pose_clip_3x3.reshape(-1, 22, 9)

        root_traj = trans_clip 

        past_pose = pose_clip[:self.input_len]      
        future_pose = pose_clip[self.input_len:]    
        past_root = root_traj[:self.input_len]      
        future_root = root_traj[self.input_len:]    

        return {
            "past_pose": torch.from_numpy(past_pose),
            "future_pose": torch.from_numpy(future_pose),
            "past_root": torch.from_numpy(past_root),
            "future_root": torch.from_numpy(future_root)
        }


def collate_fn(batch):
    keys = batch[0].keys()
    return {key: torch.stack([item[key] for item in batch]) for key in keys}