import argparse
import glob
from pathlib import Path
import os
import shutil


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt-dir", type=str)
    parser.add_argument(
        "--comfy-dir", type=str, default="/home/ostamand/git/ComfyUI/models/embeddings"
    )
    parser.add_argument("--name", type=str, default="test")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    return args


def main(ckpt_dir: str, comfy_dir: str, name: str, dry_run: bool = False):
    files = glob.glob(str(Path(ckpt_dir) / "training" / "*.pt"))

    def process_file_path(f: str):
        return {
            "step": int(os.path.basename(f).split("-")[-1].replace(".pt", "")),
            "path": f,
        }

    data_pt = sorted(map(process_file_path, files), key=lambda x: x["step"])

    for pt in data_pt:
        out_path = Path(comfy_dir) / f"{name}-{pt['step']}.pt"
        print(f"From: {pt['path']}, To: {out_path}")
        if not dry_run:
            shutil.copyfile(pt["path"], out_path)


"""
python scripts/move_to_comfy.py --ckpt-dir out/gabby-0.1 --name gabby01 --dry-run
"""
if __name__ == "__main__":
    args = parse_args()
    main(args.ckpt_dir, args.comfy_dir, args.name, dry_run=args.dry_run)
