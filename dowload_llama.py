"""
download_llama.py - download Llama 3.2 weights from Hugging Face.

Usage:
    pip install huggingface_hub
    python download_llama.py                      # Llama 3.2 3B Instruct (~6.5 GB)
    python download_llama.py --repo meta-llama/Llama-3.2-1B-Instruct --out models/llama-3.2-1b-instruct

The Llama repos are GATED: before this works you must (once)
  1. have a free huggingface.co account,
  2. open the model page in the browser and accept Meta's license,
  3. create an access token (Settings -> Access Tokens -> Read) and give it to this script
     via the HF_TOKEN environment variable, or paste it when asked.
"""
import argparse
import getpass
import os
import shutil
import sys

try:
    from huggingface_hub import snapshot_download
except ImportError:
    sys.exit("huggingface_hub is not installed. Run:  pip install huggingface_hub")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="meta-llama/Llama-3.2-3B-Instruct", help="Hugging Face repo id")
    ap.add_argument("--out", default="models/llama-3.2-3b-instruct", help="folder to put the files in")
    args = ap.parse_args()

    # 1) Token: from the environment if set, otherwise ask (getpass hides what you type).
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        token = getpass.getpass("Paste your Hugging Face token (starts with hf_, input is hidden): ").strip()
    if not token:
        sys.exit("No token given.")

    # 2) Disk check: bf16 weights are ~6.5 GB for 3B, and the converted file needs ~2.5 GB more.
    os.makedirs(args.out, exist_ok=True)
    free_gb = shutil.disk_usage(args.out).free / 1e9
    print(f"Free disk space: {free_gb:.1f} GB (need roughly 10 GB for the 3B model + conversion)")
    if free_gb < 10:
        print("WARNING: low disk space, the download may fail.")

    # 3) Download only what inference needs. The repo also contains an 'original/' folder with
    #    a second copy of the weights in Meta's own format - we skip that (it would double the download).
    patterns = [
        "config.json",
        "generation_config.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "model.safetensors.index.json",
        "*.safetensors",
    ]
    try:
        path = snapshot_download(
            repo_id=args.repo,
            local_dir=args.out,
            allow_patterns=patterns,
            token=token,
        )
    except Exception as e:  # noqa: BLE001 - we want to print a friendly hint for any failure
        name = type(e).__name__
        msg = str(e)
        print(f"\nDownload failed: {name}: {msg}\n", file=sys.stderr)
        if "Gated" in name or "gated" in msg or "403" in msg or "401" in msg:
            print(
                "This usually means access was not granted yet. Check:\n"
                f"  - Did you accept the license at https://huggingface.co/{args.repo} ?\n"
                "    (the page must say you have been granted access; approval can take a while)\n"
                "  - Is the token correct and with 'Read' permission?\n",
                file=sys.stderr,
            )
        sys.exit(1)

    print(f"\nDone. Files are in: {path}")
    for fn in sorted(os.listdir(path)):
        full = os.path.join(path, fn)
        if os.path.isfile(full):
            print(f"  {fn:45s} {os.path.getsize(full) / 1e6:10.1f} MB")
    print("\nNext step:  python convert_to_q4.py", args.out, "model.q4.bin")


if __name__ == "__main__":
    main()
