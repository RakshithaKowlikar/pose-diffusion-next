from pathlib import Path
import random
from collections import defaultdict

NPZ_DIR = Path("data/AMASS/npz")
OUTPUT_DIR = Path("data/AMASS")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
random.seed(42)

split_ratio = {"train": 0.7, "val": 0.15, "test": 0.15}
subject_to_files = defaultdict(list)

def extract_subject_id(path: Path) -> str:
    name = path.name.replace("_poses", "").replace(".npz", "")
    parent = path.parent.name  
    if parent.isdigit():
        return parent

    if name.startswith("CMU_"):
        parts = name.split("_")
        if len(parts) > 1 and parts[1].isdigit():
            return parts[1]

    if name.startswith("KIT_"):
        parts = name.split("_")
        for part in parts:
            if part.isdigit():
                return part

    if name.startswith("Transitions_mocap_"):
        parts = name.split("_")
        if len(parts) > 2:
            return parts[2]

    if "subject" in name:
        chunks = name.split("_")
        for chunk in chunks:
            if "subject" in chunk and any(ch.isdigit() for ch in chunk):
                return "".join(ch for ch in chunk if ch.isdigit())

    parts = name.split("_")
    if parts[-1].isdigit():
        return "_".join(parts[:-1])

    return name

for npz_path in NPZ_DIR.glob("*.npz"):
    try:
        subject = extract_subject_id(npz_path)
        subject_to_files[subject].append(npz_path)
    except Exception as e:
        print(f"[ERROR] Failed on {npz_path.name}: {e}")

subjects = sorted(subject_to_files.keys())
random.shuffle(subjects)

n_total = len(subjects)
n_train = int(split_ratio["train"] * n_total)
n_val = int(split_ratio["val"] * n_total)

train_subjects = subjects[:n_train]
val_subjects = subjects[n_train:n_train + n_val]
test_subjects = subjects[n_train + n_val:]

splits = {"train": [], "val": [], "test": []}
for split_name, subject_list in zip(["train", "val", "test"], [train_subjects, val_subjects, test_subjects]):
    for subj in subject_list:
        splits[split_name].extend(subject_to_files[subj])

for split_name, paths in splits.items():
    out_file = OUTPUT_DIR / f"{split_name}.txt"
    with out_file.open("w") as f:
        for p in sorted(paths):
            f.write(str(p) + "\n")
    print(f"{len(paths)} written to {out_file}")

print("\nSplit summary by subject count:")
for split_name, paths in splits.items():
    unique_subjects = {extract_subject_id(p) for p in paths}
    print(f"  {split_name.upper()}: {len(unique_subjects)} subjects, {len(paths)} files")
