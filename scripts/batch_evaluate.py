import argparse
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed


def find_jobs(root_dir):
    """
    Search for last_checkpoint.txt and config.yml in each directory under train_results,
    return tuples of (checkpoint_path, train_config_path)
    """
    print(root_dir)
    for name in os.listdir(root_dir):
        dirpath = os.path.join(root_dir, name)
        if not os.path.isdir(dirpath):
            continue
        last_ckpt_path = os.path.join(dirpath, "last_checkpoint.txt")
        train_config_path = os.path.join(dirpath, "config.yaml")
        if os.path.exists(last_ckpt_path) and os.path.exists(train_config_path):
            try:
                with open(last_ckpt_path, "r") as f:
                    ckpt_rel = f.read().strip()
                ckpt_path = ckpt_rel
                if not os.path.isabs(ckpt_rel):
                    ckpt_path = os.path.join(dirpath, ckpt_rel)
                if os.path.exists(ckpt_path):
                    yield (ckpt_path, train_config_path)
            except Exception as e:
                print(f"WARN: failed to read {last_ckpt_path}: {e}")


def run_evaluate(
    ckpt_path, train_config, config, script_path, overwrite=False, cuda_device=None
):
    exp_dir = os.path.dirname(os.path.abspath(ckpt_path))
    stdout_path = os.path.join(exp_dir, "evaluate.stdout")
    stderr_path = os.path.join(exp_dir, "evaluate.stderr")
    cmd = [
        sys.executable,
        script_path,
        "--config",
        config,
        "--checkpoint-path",
        ckpt_path,
        "--train-config",
        train_config,
    ]
    env = os.environ.copy()
    if cuda_device is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(cuda_device)
    with open(stdout_path, "w") as out, open(stderr_path, "w") as err:
        try:
            subprocess.run(cmd, stdout=out, stderr=err, check=True, env=env)
            return f"OK: {ckpt_path} (CUDA_VISIBLE_DEVICES={cuda_device})"
        except subprocess.CalledProcessError:
            return f"FAIL: {ckpt_path} (see {stderr_path}) (CUDA_VISIBLE_DEVICES={cuda_device})"


def main():
    parser = argparse.ArgumentParser(
        description="Batch evaluate checkpoints in train_results using last_checkpoint and config.yml."
    )
    parser.add_argument(
        "--config", required=True, help="Path to evaluation config file"
    )
    parser.add_argument(
        "--max-workers", type=int, default=4, help="Number of parallel workers"
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="Overwrite existing evaluate.json"
    )
    parser.add_argument(
        "--root",
        default="train_results",
        help="Root directory to search for checkpoints",
    )
    parser.add_argument(
        "--script", default="scripts/evaluate.py", help="Path to evaluate.py script"
    )
    args = parser.parse_args()

    jobs = list(find_jobs(args.root))
    if not jobs:
        print(f"No jobs found (last_checkpoint & config.yml) in {args.root}")
        return
    print(f"Found {len(jobs)} jobs. Start evaluation...")

    results = []
    num_gpus = args.max_workers  # assume same number of GPUs as max_workers
    with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
        futures = []
        for idx, (ckpt, train_config) in enumerate(jobs):
            cuda_device = idx % num_gpus
            futures.append(
                executor.submit(
                    run_evaluate,
                    ckpt,
                    train_config,
                    args.config,
                    args.script,
                    args.overwrite,
                    cuda_device,
                )
            )
        for f in as_completed(futures):
            res = f.result()
            print(res)
            results.append(res)
    print("Batch evaluation finished.")


if __name__ == "__main__":
    main()
