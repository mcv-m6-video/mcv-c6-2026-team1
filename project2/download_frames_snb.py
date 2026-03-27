import argparse
from huggingface_hub import snapshot_download

parser = argparse.ArgumentParser(description="Download SoccerNet dataset to a specific directory.")
parser.add_argument("--local_dir", default="SoccerNet/SN-BAS-2025", help="Local directory to download the dataset to.")
args = parser.parse_args()

snapshot_download(repo_id="SoccerNet/SN-BAS-2025",
                  repo_type="dataset", revision="main",
                  local_dir=args.local_dir)