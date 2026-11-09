"""
chat.py - talk to Llama 3.x through engine.cpp.

Usage:
    pip install tokenizers
    python chat.py --model model.q4.bin --tokenizer models/llama-3.2-3b-instruct/tokenizer.json --engine ./engine
    (on Windows: --engine engine.exe)

What this script does (the C++ engine only knows numbers):
    your text -> chat template -> tokenizer -> token ids --> engine.cpp --> token ids -> tokenizer -> text
"""
import argparse
import subprocess
import sys

try:
    from tokenizers import Tokenizer
except ImportError:
    sys.exit("tokenizers is not installed. Run:  pip install tokenizers")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="model.q4.bin made by convert_to_q4.py")
    ap.add_argument("--tokenizer", required=True, help="tokenizer.json from the Hugging Face folder")
    ap.add_argument("--engine", default="./engine", help="path to the compiled engine")
    ap.add_argument("--system", default="You are a helpful assistant.")
    ap.add_argument("--ctx", type=int, default=2048, help="context window (more = more RAM)")
    ap.add_argument("--maxnew", type=int, default=512, help="max tokens per answer")
    ap.add_argument("--temp", type=float, default=0.6)
    ap.add_argument("--topp", type=float, default=0.9)
    ap.add_argument("--threads", type=int, default=0, help="0 = let OpenMP decide; try the number of physical cores")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--prompt", default=None, help="ask one question and exit")
    ap.add_argument("--raw", action="store_true",
                    help="no chat template: plain text continuation (for base models / testing)")
    ap.add_argument("--stop-ids", default=None, help="comma separated token ids that end an answer")
    args = ap.parse_args()

    tok = Tokenizer.from_file(args.tokenizer)

    # Special tokens that end an answer in Llama 3: <|eot_id|> (end of turn), <|end_of_text|>, <|eom_id|>.
    if args.stop_ids:
        stop_ids = [int(x) for x in args.stop_ids.split(",") if x]
    else:
        stop_ids = [i for i in (tok.token_to_id(t) for t in ("<|eot_id|>", "<|end_of_text|>", "<|eom_id|>")) if i is not None]
    eot_id = tok.token_to_id("<|eot_id|>")

    cmd = [args.engine, args.model, "--ctx", str(args.ctx), "--maxnew", str(args.maxnew),
           "--temp", str(args.temp), "--topp", str(args.topp), "--threads", str(args.threads)]
    if stop_ids:
        cmd += ["--stop", ",".join(map(str, stop_ids))]
    if args.seed is not None:
        cmd += ["--seed", str(args.seed)]

    # stdin/stdout are pipes; stderr is inherited so you see the engine's speed statistics.
    eng = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True,
                           encoding="utf-8", bufsize=1)

    first_turn = True      # does the engine already have the conversation start (BOS + system prompt)?
    pending_eot = False    # did the last answer end WITHOUT <|eot_id|> in the engine's cache?

    def ask(user_text: str) -> None:
        nonlocal first_turn, pending_eot
        if args.raw:
            text = user_text
        else:
            text = ""
            if pending_eot:
                text += "<|eot_id|>"
            if first_turn:
                text += ("<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
                         f"{args.system}<|eot_id|>")
            text += ("<|start_header_id|>user<|end_header_id|>\n\n"
                     f"{user_text}<|eot_id|>"
                     "<|start_header_id|>assistant<|end_header_id|>\n\n")
        first_turn = False
        pending_eot = False

        ids = tok.encode(text, add_special_tokens=False).ids
        eng.stdin.write(" ".join(map(str, ids)) + "\n")
        eng.stdin.flush()

        out_ids, printed = [], 0
        while True:
            line = eng.stdout.readline()
            if not line:
                sys.exit("engine exited unexpectedly")
            line = line.strip()
            if line.startswith("T "):
                tid = int(line[2:])
                if tid in stop_ids:
                    continue                      # do not print <|eot_id|> and friends
                out_ids.append(tid)
                # Decode everything so far and print only the new part. A character can be split over
                # several tokens; while it is incomplete the text ends with U+FFFD, so we wait.
                s = tok.decode(out_ids, skip_special_tokens=False)
                end = len(s) - 1 if s.endswith("�") else len(s)
                if end > printed:
                    print(s[printed:end], end="", flush=True)
                    printed = end
            elif line.startswith("END"):
                pending_eot = line.endswith("length")
                break
            elif line.startswith("ERR"):
                print(f"\n[engine] {line}")
                break
        print()

    try:
        if args.prompt is not None:
            ask(args.prompt)
            return
        print("Chat started. Commands: /reset (new conversation), /exit")
        while True:
            try:
                user = input("\nYou: ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not user:
                continue
            if user == "/exit":
                break
            if user == "/reset":
                eng.stdin.write("RESET\n")
                eng.stdin.flush()
                eng.stdout.readline()
                first_turn, pending_eot = True, False
                print("(conversation cleared)")
                continue
            print("Llama: ", end="", flush=True)
            ask(user)
    finally:
        try:
            eng.stdin.write("QUIT\n")
            eng.stdin.flush()
        except Exception:  # noqa: BLE001
            pass
        eng.terminate()


if __name__ == "__main__":
    main()
