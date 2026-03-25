import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from scapy.all import PcapReader
from scapy.layers.inet import IP, TCP, UDP
from scapy.layers.inet6 import IPv6
from tqdm import tqdm


# ===== Edit these configs directly =====
INPUT_ROOT = Path("./TrafficData")
# If empty, scan all datasets under INPUT_ROOT.
DATASET_NAMES: List[str] = []
# Grouping scope:
# - "dataset_label": duplicates checked within each dataset+label group
# - "label_global": duplicates checked within same label across all datasets
GROUP_SCOPE = "dataset_label"
# Stop early when reading one pcap after first flow key is found.
FIRST_FLOW_ONLY = True
# Save summary and details json files.
SAVE_REPORT = True
REPORT_DIR = Path("./flow_dup_report")


FlowKey = Tuple[str, str, str, int, int]  # (ip_ver+proto, src, dst, sport, dport)


def iter_pcap_files(input_root: Path, dataset_names: List[str]) -> List[Path]:
    exts = {".pcap", ".cap"}
    files: List[Path] = []

    if dataset_names:
        datasets = [input_root / n for n in dataset_names]
    else:
        datasets = [p for p in input_root.iterdir() if p.is_dir()]

    for ds in datasets:
        if not ds.exists():
            print(f"Skip missing dataset folder: {ds}")
            continue
        for p in ds.rglob("*"):
            if p.is_file() and p.suffix.lower() in exts:
                files.append(p)
    return sorted(files)


def parse_group_keys(p: Path, input_root: Path) -> Tuple[str, str]:
    rel = p.relative_to(input_root)
    parts = rel.parts
    if len(parts) < 3:
        # fallback for unexpected structures
        dataset = parts[0] if len(parts) >= 1 else "unknown_dataset"
        label = parts[-2] if len(parts) >= 2 else "unknown_label"
    else:
        dataset = parts[0]
        label = parts[1]
    return dataset, label


def flow_keys_from_pcap(path: Path, first_only: bool = True) -> List[FlowKey]:
    keys: List[FlowKey] = []
    seen = set()
    try:
        with PcapReader(str(path)) as reader:
            for pkt in reader:
                key: Optional[FlowKey] = None
                if IP in pkt:
                    ip = pkt[IP]
                    if TCP in pkt:
                        t = pkt[TCP]
                        key = (f"ipv4:{ip.proto}", ip.src, ip.dst, int(t.sport), int(t.dport))
                    elif UDP in pkt:
                        u = pkt[UDP]
                        key = (f"ipv4:{ip.proto}", ip.src, ip.dst, int(u.sport), int(u.dport))
                elif IPv6 in pkt:
                    ip6 = pkt[IPv6]
                    if TCP in pkt:
                        t = pkt[TCP]
                        key = (f"ipv6:{ip6.nh}", ip6.src, ip6.dst, int(t.sport), int(t.dport))
                    elif UDP in pkt:
                        u = pkt[UDP]
                        key = (f"ipv6:{ip6.nh}", ip6.src, ip6.dst, int(u.sport), int(u.dport))

                if key is None:
                    continue
                if key in seen:
                    continue
                seen.add(key)
                keys.append(key)
                if first_only:
                    break
    except Exception as e:
        print(f"Failed to read {path}: {e}")
    return keys


def main() -> None:
    input_root = INPUT_ROOT.resolve()
    if not input_root.exists():
        raise FileNotFoundError(f"INPUT_ROOT not found: {input_root}")

    pcap_files = iter_pcap_files(input_root, DATASET_NAMES)
    if not pcap_files:
        raise ValueError(f"No pcap/cap files found under: {input_root}")

    # group -> flow_key -> list[pcap_path]
    groups: Dict[str, Dict[FlowKey, List[str]]] = defaultdict(lambda: defaultdict(list))

    for pcap in tqdm(pcap_files, desc="Scanning pcaps", unit="file"):
        dataset, label = parse_group_keys(pcap, input_root)
        if GROUP_SCOPE == "label_global":
            group_id = label
        else:
            group_id = f"{dataset}/{label}"

        keys = flow_keys_from_pcap(pcap, first_only=FIRST_FLOW_ONLY)
        for k in keys:
            groups[group_id][k].append(str(pcap))

    repeated_flow_count = 0
    repeated_pair_count = 0
    repeated_file_count = 0
    details = []
    files_involved = set()

    for group_id, flow_map in groups.items():
        for flow_key, paths in flow_map.items():
            uniq_paths = sorted(set(paths))
            if len(uniq_paths) < 2:
                continue
            repeated_flow_count += 1
            repeated_pair_count += (len(uniq_paths) * (len(uniq_paths) - 1)) // 2
            repeated_file_count += len(uniq_paths)
            files_involved.update(uniq_paths)
            details.append(
                {
                    "group": group_id,
                    "flow_key": {
                        "proto": flow_key[0],
                        "src_ip": flow_key[1],
                        "dst_ip": flow_key[2],
                        "src_port": flow_key[3],
                        "dst_port": flow_key[4],
                    },
                    "pcap_count": len(uniq_paths),
                    "pcap_files": uniq_paths,
                }
            )

    summary = {
        "input_root": str(input_root),
        "dataset_names": DATASET_NAMES,
        "group_scope": GROUP_SCOPE,
        "first_flow_only": FIRST_FLOW_ONLY,
        "total_pcap_files_scanned": len(pcap_files),
        "duplicate_flow_key_count": repeated_flow_count,
        "duplicate_pair_count": repeated_pair_count,
        "duplicate_files_total_count": repeated_file_count,
        "duplicate_files_unique_count": len(files_involved),
    }

    print("\n=== Duplicate Flow Check Summary ===")
    for k, v in summary.items():
        print(f"{k}: {v}")

    if SAVE_REPORT:
        report_dir = REPORT_DIR.resolve()
        report_dir.mkdir(parents=True, exist_ok=True)
        summary_path = report_dir / "duplicate_flow_summary.json"
        details_path = report_dir / "duplicate_flow_details.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        with open(details_path, "w", encoding="utf-8") as f:
            json.dump(details, f, indent=2, ensure_ascii=False)
        print(f"\nSaved summary: {summary_path}")
        print(f"Saved details: {details_path}")


if __name__ == "__main__":
    main()
