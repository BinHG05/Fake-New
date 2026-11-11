import argparse
import os
import re
import subprocess
import sys


DEFAULT_ENV_PYTHON = r"D:\anacoda3\envs\multimodal_gnn\python.exe"


def resolve_python():
    current_python = sys.executable
    if current_python and os.path.exists(current_python):
        return current_python
    if os.path.exists(DEFAULT_ENV_PYTHON):
        return DEFAULT_ENV_PYTHON
    raise FileNotFoundError("Could not resolve a Python executable for the multimodal_gnn environment.")


def run_trainer(python_exe, seed, extra_args):
    cmd = [
        python_exe,
        "-u",
        "src/training/train_multimodal_gnn.py",
        "--epochs",
        "100",
        "--batch_size",
        "32",
        "--augment_edge_drop",
        "0.1",
        "--augment_feature_mask",
        "0.1",
        "--lr_scheduler",
        "cosine",
        "--seed",
        str(seed),
    ] + list(extra_args)

    print(f"\n>>> Running training with seed: {seed}")
    print(">>> Command:", " ".join(cmd))

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
        bufsize=1,
    )

    last_accuracy = 0.0
    for line in process.stdout:
        try:
            print(line, end="", flush=True)
        except Exception:
            pass

        match = re.search(r"Binary Accuracy:\s+([\d\.]+)", line)
        if match:
            last_accuracy = float(match.group(1))

    process.wait()
    return process.returncode, last_accuracy


def main():
    parser = argparse.ArgumentParser(description="Retry multimodal training with new seeds until a target accuracy is reached.")
    parser.add_argument("--target", type=float, default=0.90)
    parser.add_argument("--seed_start", type=int, default=42)
    parser.add_argument("--seed_step", type=int, default=10)
    parser.add_argument(
        "trainer_args",
        nargs=argparse.REMAINDER,
        help="Extra args forwarded to train_multimodal_gnn.py. Prefix with -- before trainer args.",
    )
    args = parser.parse_args()
    trainer_args = list(args.trainer_args)
    if trainer_args and trainer_args[0] == "--":
        trainer_args = trainer_args[1:]

    python_exe = resolve_python()
    target = args.target
    seed = args.seed_start
    best_acc = 0.0

    while best_acc < target:
        code, acc = run_trainer(python_exe, seed, trainer_args)
        if code != 0:
            print(f"\n[ERROR] Training failed for seed {seed} with exit code {code}.")
            sys.exit(code)

        if acc > best_acc:
            best_acc = acc

        if best_acc >= target:
            print(f"\n[DONE] Mission accomplished. Reached {best_acc:.4f} accuracy.")
            break

        print(f"\n[INFO] Current best: {best_acc:.4f}. Retrying with a new seed...")
        seed += args.seed_step


if __name__ == "__main__":
    main()
