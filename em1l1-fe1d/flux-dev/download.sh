#!/bin/bash
python -m venv venv
source venv/bin/activate
pip install huggingface_hub
python download_flux.py --out_folder models --clear_cache
deactivate