import argparse
import os
import shutil

from huggingface_hub import hf_hub_download, login

ACCESS_TOKEN = "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"

def main(out_folder: str | None, clear_cache: bool):
    login(ACCESS_TOKEN)
    # check if folder already exists
    if out_folder is not None:
        if os.path.exists(out_folder):
            raise Exception("Output folder already exist")
        os.makedirs(out_folder)

    files = [
        {"repo_id": "comfyanonymous/flux_text_encoders", "filename": "t5xxl_fp16.safetensors"},  
        {"repo_id": "comfyanonymous/flux_text_encoders", "filename": "clip_l.safetensors"},
        {"repo_id": "black-forest-labs/FLUX.1-dev", "filename": "ae.safetensors"},
        {"repo_id": "black-forest-labs/FLUX.1-dev", "filename": "flux1-dev.safetensors"}
    ]

    for file in files:
        file_path = hf_hub_download(repo_id=file["repo_id"], filename=file["filename"])
        if out_folder is not None:
            shutil.copy(file_path, os.path.join(os.curdir, out_folder, file["filename"]))

    # clear cache
    if clear_cache:
        cache_dir = os.path.expanduser("~/.cache/huggingface")
        shutil.rmtree(cache_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_folder", type=str, default=None, help="Output folder to save the downloaded files")
    parser.add_argument("--clear_cache", action="store_true", help="Clear the cache after downloading files")
    
    args = parser.parse_args()
    main(args.out_folder, args.clear_cache)