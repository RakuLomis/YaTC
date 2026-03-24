import subprocess
import sys
import importlib.util
import importlib.metadata
from pathlib import Path


# ===== Edit these configs directly =====
PROJECT_ROOT = Path(__file__).resolve().parent
# Use paths relative to this script's directory.
PNG_INPUT_ROOT = (PROJECT_ROOT / "../../TrafficData/data_png").resolve()
SPLIT_OUTPUT_ROOT = (PROJECT_ROOT / "../../TrafficData/data_png_split").resolve()
RUN_OUTPUT_ROOT = (PROJECT_ROOT / "../../Res/YaTC").resolve()
PRETRAINED_CKPT = (PROJECT_ROOT / "../../OtherModels/PretrainedModels/YaTC_pretrained_model.pth").resolve()

# Only these datasets will be processed.
DATASETS_TO_RUN = ["cstnet_tls_1.3", "CipherSpectrum"]

# Number of classes for each dataset.
NB_CLASSES = {
    # "ISCX-VPN": 7,
    "cstnet_tls_1.3": 26,
    "CipherSpectrum": 41,
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
FORCE_RESPLIT = False

# Dependency guard
AUTO_INSTALL_MISSING = False
REQUIRED_PACKAGES = {
    "timm": "timm",
    "torch": "torch",
    "torchvision": "torchvision",
    "numpy": "numpy",
    "sklearn": "scikit-learn",
    "PIL": "pillow",
    "tqdm": "tqdm",
}
MIN_TIMM_VERSION = (0, 9, 0)


def run_cmd(args):
    print("Running:", " ".join(str(a) for a in args))
    subprocess.run(args, check=True)


def _split_has_png(split_dir: Path) -> bool:
    return split_dir.exists() and any(split_dir.rglob("*.png"))


def is_dataset_split_ready(dataset_dir: Path) -> bool:
    train_dir = dataset_dir / "train"
    val_dir = dataset_dir / "val"
    test_dir = dataset_dir / "test"
    return _split_has_png(train_dir) and _split_has_png(val_dir) and _split_has_png(test_dir)


def ensure_dependencies(py_exec: str):
    missing = []
    for module_name, pip_name in REQUIRED_PACKAGES.items():
        if importlib.util.find_spec(module_name) is None:
            missing.append((module_name, pip_name))

    if not missing:
        # timm old versions (e.g., 0.3.2) are incompatible with torch 2.x.
        timm_version = importlib.metadata.version("timm")
        version_tuple = tuple(int(x) for x in timm_version.split(".")[:3])
        if version_tuple < MIN_TIMM_VERSION:
            raise RuntimeError(
                f"Detected timm=={timm_version}, which is too old for torch 2.x.\n"
                f"Please upgrade timm first, e.g.:\n{py_exec} -m pip install -U timm\n"
                "Then rerun run_yatc_pipeline.py"
            )
        return

    install_specs = [pkg for _, pkg in missing]
    install_cmd = [py_exec, "-m", "pip", "install", *install_specs]
    print("Missing python packages:")
    for mod, pkg in missing:
        print(f"  - module '{mod}' (pip: {pkg})")

    if AUTO_INSTALL_MISSING:
        run_cmd(install_cmd)
        return

    cmd_str = " ".join(install_cmd)
    raise RuntimeError(
        "Missing required dependencies. Install them in this environment first:\n"
        f"{cmd_str}\n"
        "Then rerun run_yatc_pipeline.py"
    )


def main():
    py = sys.executable
    ensure_dependencies(py)
    PROJECT_ROOT.mkdir(parents=True, exist_ok=True)
    RUN_OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    if not PRETRAINED_CKPT.exists():
        raise FileNotFoundError(f"Pretrained checkpoint not found: {PRETRAINED_CKPT}")

    # 1) Split datasets to train/val/test only when needed.
    if FORCE_RESPLIT:
        datasets_to_split = list(DATASETS_TO_RUN)
    else:
        datasets_to_split = []
        for dataset_name in DATASETS_TO_RUN:
            dataset_split_dir = SPLIT_OUTPUT_ROOT / dataset_name
            if is_dataset_split_ready(dataset_split_dir):
                print(f"[Skip split] {dataset_name} already has train/val/test with PNG files.")
            else:
                datasets_to_split.append(dataset_name)

    if datasets_to_split:
        split_cmd = [
            py,
            str(PROJECT_ROOT / "split_png_dataset.py"),
            "--input_root",
            str(PNG_INPUT_ROOT),
            "--output_root",
            str(SPLIT_OUTPUT_ROOT),
            "--dataset_names",
            *datasets_to_split,
            "--train_ratio",
            str(TRAIN_RATIO),
            "--val_ratio",
            str(VAL_RATIO),
            "--test_ratio",
            str(TEST_RATIO),
            "--seed",
            str(SPLIT_SEED),
        ]
        if FORCE_RESPLIT:
            split_cmd.append("--overwrite")
        run_cmd(split_cmd)
    else:
        print("All requested datasets are already split. Start training/evaluation directly.")

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
