import argparse
import random
import shutil
from pathlib import Path
from typing import Dict, List, Tuple

from tqdm import tqdm


DEFAULT_INPUT_ROOT = "./TrafficData_png"
DEFAULT_OUTPUT_ROOT = "./data"
DEFAULT_DATASET_NAMES: List[str] = []
DEFAULT_TRAIN_RATIO = 0.7
DEFAULT_VAL_RATIO = 0.15
DEFAULT_TEST_RATIO = 0.15
DEFAULT_SEED = 42


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("Split PNG dataset into train/val/test by label")
    parser.add_argument("--input_root", type=str, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output_root", type=str, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--dataset_names", nargs="*", default=DEFAULT_DATASET_NAMES)
    parser.add_argument("--train_ratio", type=float, default=DEFAULT_TRAIN_RATIO)
    parser.add_argument("--val_ratio", type=float, default=DEFAULT_VAL_RATIO)
    parser.add_argument("--test_ratio", type=float, default=DEFAULT_TEST_RATIO)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--overwrite", action="store_true", help="Delete target split folders before writing.")
    return parser.parse_args()


def split_counts(n: int, train_ratio: float, val_ratio: float, test_ratio: float) -> Tuple[int, int, int]:
    train_n = int(round(n * train_ratio))
    val_n = int(round(n * val_ratio))
    test_n = n - train_n - val_n

    if n >= 3:
        if train_n == 0:
            train_n = 1
        if val_n == 0:
            val_n = 1
        test_n = n - train_n - val_n
        if test_n <= 0:
            test_n = 1
            if train_n > val_n:
                train_n -= 1
            else:
                val_n -= 1
    return train_n, val_n, test_n


def get_dataset_names(input_root: Path, dataset_names: List[str]) -> List[str]:
    if dataset_names:
        return dataset_names
    return sorted([p.name for p in input_root.iterdir() if p.is_dir()])


def copy_split(files: List[Path], target_dir: Path) -> int:
    target_dir.mkdir(parents=True, exist_ok=True)
    for f in files:
        shutil.copy2(f, target_dir / f.name)
    return len(files)


def split_one_dataset(
    dataset_dir: Path,
    output_root: Path,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    rng: random.Random,
    overwrite: bool,
) -> Dict[str, int]:
    out_dataset_dir = output_root / dataset_dir.name
    if overwrite:
        for split in ["train", "val", "test"]:
            split_dir = out_dataset_dir / split
            if split_dir.exists():
                shutil.rmtree(split_dir)

    counts = {"train": 0, "val": 0, "test": 0}
    label_dirs = sorted([p for p in dataset_dir.iterdir() if p.is_dir()])

    for label_dir in label_dirs:
        imgs = sorted([p for p in label_dir.iterdir() if p.is_file() and p.suffix.lower() == ".png"])
        if not imgs:
            continue
        rng.shuffle(imgs)
        train_n, val_n, test_n = split_counts(len(imgs), train_ratio, val_ratio, test_ratio)

        train_files = imgs[:train_n]
        val_files = imgs[train_n:train_n + val_n]
        test_files = imgs[train_n + val_n:train_n + val_n + test_n]

        counts["train"] += copy_split(train_files, out_dataset_dir / "train" / label_dir.name)
        counts["val"] += copy_split(val_files, out_dataset_dir / "val" / label_dir.name)
        counts["test"] += copy_split(test_files, out_dataset_dir / "test" / label_dir.name)

    return counts


def main() -> None:
    args = parse_args()
    input_root = Path(args.input_root).resolve()
    output_root = Path(args.output_root).resolve()

    ratio_sum = args.train_ratio + args.val_ratio + args.test_ratio
    if abs(ratio_sum - 1.0) > 1e-6:
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")

    if not input_root.exists():
        raise FileNotFoundError(f"input_root not found: {input_root}")

    dataset_names = get_dataset_names(input_root, args.dataset_names)
    if not dataset_names:
        raise ValueError(f"No dataset folders found under: {input_root}")

    rng = random.Random(args.seed)
    summary = {}

    for name in tqdm(dataset_names, desc="Splitting datasets", unit="dataset"):
        dataset_dir = input_root / name
        if not dataset_dir.is_dir():
            print(f"Skip missing dataset folder: {dataset_dir}")
            continue
        summary[name] = split_one_dataset(
            dataset_dir=dataset_dir,
            output_root=output_root,
            train_ratio=args.train_ratio,
            val_ratio=args.val_ratio,
            test_ratio=args.test_ratio,
            rng=rng,
            overwrite=args.overwrite,
        )

    print("Done.")
    print(f"Input root : {input_root}")
    print(f"Output root: {output_root}")
    for name, counts in summary.items():
        total = counts["train"] + counts["val"] + counts["test"]
        print(
            f"{name}: train={counts['train']}, val={counts['val']}, test={counts['test']}, total={total}"
        )


if __name__ == "__main__":
    main()
