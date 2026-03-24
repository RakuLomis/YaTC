import subprocess
import sys
from pathlib import Path


# ===== Edit these configs directly =====
PROJECT_ROOT = Path(__file__).resolve().parent
PNG_INPUT_ROOT = PROJECT_ROOT / "TrafficData_png"
SPLIT_OUTPUT_ROOT = PROJECT_ROOT / "data"
RUN_OUTPUT_ROOT = PROJECT_ROOT / "runs_yatc"
PRETRAINED_CKPT = PROJECT_ROOT / "output_dir" / "pretrained-model.pth"

# Only these datasets will be processed.
DATASETS_TO_RUN = ["ISCX-VPN", "tls1.3", "CipherSpectrum"]

# Number of classes for each dataset.
NB_CLASSES = {
    "ISCX-VPN": 7,
    "tls1.3": 2,
    "CipherSpectrum": 6,
}

# Split ratios
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1
SPLIT_SEED = 42

# Training hyperparameters
EPOCHS = 100
BATCH_SIZE = 64
NUM_WORKERS = 4
BASE_LR = 2e-3


def run_cmd(args):
    print("Running:", " ".join(str(a) for a in args))
    subprocess.run(args, check=True)


def main():
    py = sys.executable
    PROJECT_ROOT.mkdir(parents=True, exist_ok=True)
    RUN_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    if not PRETRAINED_CKPT.exists():
        raise FileNotFoundError(f"Pretrained checkpoint not found: {PRETRAINED_CKPT}")

    # 1) Split datasets to train/val/test.
    run_cmd(
        [
            py,
            str(PROJECT_ROOT / "split_png_dataset.py"),
            "--input_root",
            str(PNG_INPUT_ROOT),
            "--output_root",
            str(SPLIT_OUTPUT_ROOT),
            "--dataset_names",
            *DATASETS_TO_RUN,
            "--train_ratio",
            str(TRAIN_RATIO),
            "--val_ratio",
            str(VAL_RATIO),
            "--test_ratio",
            str(TEST_RATIO),
            "--seed",
            str(SPLIT_SEED),
            "--overwrite",
        ]
    )

    # 2) Train + 3) Test eval with saved metrics/cm.
    for dataset_name in DATASETS_TO_RUN:
        if dataset_name not in NB_CLASSES:
            raise KeyError(f"Missing nb_classes config for dataset: {dataset_name}")

        data_path = SPLIT_OUTPUT_ROOT / dataset_name
        run_dir = RUN_OUTPUT_ROOT / dataset_name
        eval_dir = run_dir / "eval"
        run_dir.mkdir(parents=True, exist_ok=True)
        eval_dir.mkdir(parents=True, exist_ok=True)

        # Train with validation monitoring.
        run_cmd(
            [
                py,
                str(PROJECT_ROOT / "fine-tune.py"),
                "--data_path",
                str(data_path),
                "--nb_classes",
                str(NB_CLASSES[dataset_name]),
                "--finetune",
                str(PRETRAINED_CKPT),
                "--epochs",
                str(EPOCHS),
                "--batch_size",
                str(BATCH_SIZE),
                "--num_workers",
                str(NUM_WORKERS),
                "--blr",
                str(BASE_LR),
                "--output_dir",
                str(run_dir),
                "--log_dir",
                str(run_dir),
                "--train_split",
                "train",
                "--val_split",
                "val",
                "--test_split",
                "test",
            ]
        )

        best_ckpt = run_dir / "checkpoint-best.pth"
        if not best_ckpt.exists():
            raise FileNotFoundError(f"Best checkpoint not found after training: {best_ckpt}")

        # Final test evaluation and save artifacts.
        run_cmd(
            [
                py,
                str(PROJECT_ROOT / "fine-tune.py"),
                "--data_path",
                str(data_path),
                "--nb_classes",
                str(NB_CLASSES[dataset_name]),
                "--eval",
                "--resume",
                str(best_ckpt),
                "--eval_split",
                "test",
                "--save_eval_dir",
                str(eval_dir),
                "--save_eval_prefix",
                "test_final",
                "--batch_size",
                str(BATCH_SIZE),
                "--num_workers",
                str(NUM_WORKERS),
            ]
        )

    print("All datasets finished.")
    print(f"Run outputs: {RUN_OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
