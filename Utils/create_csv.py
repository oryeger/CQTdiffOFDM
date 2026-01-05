import os
import csv
import random
import shutil
from pathlib import Path

ROOT = Path(r"C:\Projects\CQTdiffOFDM\examples\costum44100")  # dset_args.path
PATTERN = "ofdm_maestro_like_Nfft2048_16QAM_clean*.wav"
CSV_NAME = "costum.csv"

SEED = 17
N_TRAIN, N_VAL, N_TEST = 80, 10, 10

train_year, val_year, test_year = 2003, 2007, 2010

# 1) Collect files (only in ROOT, not recursively)
files = sorted(ROOT.glob(PATTERN))
if len(files) != (N_TRAIN + N_VAL + N_TEST):
    raise ValueError(f"Expected {N_TRAIN+N_VAL+N_TEST} files, found {len(files)}")

# 2) Shuffle
random.seed(SEED)
random.shuffle(files)

train_files = files[:N_TRAIN]
val_files   = files[N_TRAIN:N_TRAIN+N_VAL]
test_files  = files[N_TRAIN+N_VAL:]

# 3) Ensure year folders exist
(ROOT / str(train_year)).mkdir(exist_ok=True)
(ROOT / str(val_year)).mkdir(exist_ok=True)
(ROOT / str(test_year)).mkdir(exist_ok=True)

rows = []

# def move_and_add(file_path: Path, year: int, split: str):
#     dest = ROOT / str(year) / file_path.name
#     # Move file into year folder
#     shutil.move(str(file_path), str(dest))
#
#     # audio_filename must be relative to ROOT
#     rel = dest.relative_to(ROOT).as_posix()  # forward slashes
#     rows.append({"audio_filename": rel, "year": year, "split": split})
def move_and_add(file_path: Path, year: int, split: str):
    dest = ROOT / str(year) / file_path.name
    shutil.move(str(file_path), str(dest))

    # audio_filename relative to ROOT, with backslashes
    rel = str(dest.relative_to(ROOT))
    rows.append({"audio_filename": rel, "year": year, "split": split})

for f in train_files:
    move_and_add(f, train_year, "train")
for f in val_files:
    move_and_add(f, val_year, "validation")
for f in test_files:
    move_and_add(f, test_year, "test")

# 4) Write CSV
csv_path = ROOT / CSV_NAME
with open(csv_path, "w", newline="", encoding="utf-8") as fp:
    writer = csv.DictWriter(fp, fieldnames=["audio_filename", "year", "split"])
    writer.writeheader()
    writer.writerows(rows)

print(f"Wrote {csv_path} with {len(rows)} rows.")
print(f"train={N_TRAIN}, validation={N_VAL}, test={N_TEST}")
