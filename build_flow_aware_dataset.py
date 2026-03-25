import hashlib
import json
import random
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
from scapy.all import PcapReader
from scapy.layers.inet import IP, TCP, UDP
from scapy.layers.inet6 import IPv6
from scapy.layers.l2 import Ether
from tqdm import tqdm

try:
    from scapy.layers.tls.extensions import TLS_Ext_ServerName
except Exception:
    TLS_Ext_ServerName = None


# =========================
# Edit Configs Here
# =========================
INPUT_ROOT = Path(r"../../TrafficData/datasets_raw_add2")
OUTPUT_ROOT = Path(r"../../TrafficData/data_png_split_flowaware")

# Leave empty to process all datasets under INPUT_ROOT.
DATASET_NAMES: List[str] = ["ISCX-VPN", "cstnet_tls_1.3", "CipherSpectrum"]

# Dataset mode:
# - "multi_flow_per_pcap": one pcap may contain many flows; same 5-tuple within SAME pcap = one flow sample.
# - "single_flow_per_pcap": each pcap is one flow sample.
DATASET_MODES: Dict[str, str] = {
    "ISCX-VPN": "multi_flow_per_pcap",
    "cstnet_tls_1.3": "single_flow_per_pcap",
    "CipherSpectrum": "single_flow_per_pcap",
}

TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1
RANDOM_SEED = 42

MAX_PACKETS_PER_SAMPLE = 5
IMAGE_SIZE = 40

MASK_IP_PORT = True
MASK_TLS_SNI = True

OVERWRITE_OUTPUT_DATASET = True
SAVE_MANIFEST = True
MANIFEST_NAME = "flow_manifest.json"
SHOW_DETAILED_PROGRESS = True


FlowKey = Tuple[str, str, str, int, int]  # (proto_tag, src, dst, sport, dport)


def ensure_ratio_valid() -> None:
    s = TRAIN_RATIO + VAL_RATIO + TEST_RATIO
    if abs(s - 1.0) > 1e-6:
        raise ValueError("TRAIN_RATIO + VAL_RATIO + TEST_RATIO must equal 1.0")


def normalize_hex(s: str, target_len: int) -> str:
    if len(s) > target_len:
        return s[:target_len]
    return s + ("0" * (target_len - len(s)))


def flow_key_from_pkt(pkt) -> Optional[FlowKey]:
    if IP in pkt:
        ip = pkt[IP]
        if TCP in pkt:
            t = pkt[TCP]
            return (f"ipv4:{ip.proto}", ip.src, ip.dst, int(t.sport), int(t.dport))
        if UDP in pkt:
            u = pkt[UDP]
            return (f"ipv4:{ip.proto}", ip.src, ip.dst, int(u.sport), int(u.dport))
    elif IPv6 in pkt:
        ip6 = pkt[IPv6]
        if TCP in pkt:
            t = pkt[TCP]
            return (f"ipv6:{ip6.nh}", ip6.src, ip6.dst, int(t.sport), int(t.dport))
        if UDP in pkt:
            u = pkt[UDP]
            return (f"ipv6:{ip6.nh}", ip6.src, ip6.dst, int(u.sport), int(u.dport))
    return None


def _mask_ip_and_port(pkt) -> None:
    if IP in pkt:
        pkt[IP].src = "0.0.0.0"
        pkt[IP].dst = "0.0.0.0"
        pkt[IP].len = None
        pkt[IP].chksum = None
    if IPv6 in pkt:
        pkt[IPv6].src = "::"
        pkt[IPv6].dst = "::"
        pkt[IPv6].plen = None
    if TCP in pkt:
        pkt[TCP].sport = 0
        pkt[TCP].dport = 0
        pkt[TCP].chksum = None
    if UDP in pkt:
        pkt[UDP].sport = 0
        pkt[UDP].dport = 0
        pkt[UDP].len = None
        pkt[UDP].chksum = None


def _mask_or_drop_tls_sni(pkt) -> None:
    if TLS_Ext_ServerName is None or TLS_Ext_ServerName not in pkt:
        return

    done = False
    ext = pkt.getlayer(TLS_Ext_ServerName)
    if ext is not None and hasattr(ext, "servernames"):
        try:
            for sn in ext.servernames:
                if hasattr(sn, "servername"):
                    sn.servername = b"masked.local" if isinstance(sn.servername, bytes) else "masked.local"
            done = True
        except Exception:
            done = False

    if not done and TCP in pkt:
        pkt[TCP].remove_payload()
        pkt[TCP].chksum = None
        if IP in pkt:
            pkt[IP].len = None
            pkt[IP].chksum = None


def sanitize_packet_to_yatc_bytes(pkt) -> Optional[bytes]:
    p = pkt.copy()
    if MASK_IP_PORT:
        _mask_ip_and_port(p)
    if MASK_TLS_SNI:
        _mask_or_drop_tls_sni(p)

    if Ether in p:
        p = p[Ether].payload

    if not (IP in p or IPv6 in p):
        return None

    # Follow YaTC style: header (80B) + payload (240B), max 5 packets.
    if IP in p:
        ip_layer = p[IP]
    else:
        ip_layer = p[IPv6]

    header_hex = bytes(ip_layer).hex()
    payload_hex = ""
    if TCP in p:
        payload_hex = bytes(p[TCP].payload).hex()
    elif UDP in p:
        payload_hex = bytes(p[UDP].payload).hex()

    if payload_hex:
        if header_hex.endswith(payload_hex):
            header_hex = header_hex[: -len(payload_hex)]
        else:
            header_hex = header_hex.replace(payload_hex, "", 1)

    header_hex = normalize_hex(header_hex, 160)
    payload_hex = normalize_hex(payload_hex, 480)
    return bytes.fromhex(header_hex + payload_hex)  # 320 bytes


def sample_to_image_vec(packet_bytes_list: List[bytes]) -> np.ndarray:
    sample = list(packet_bytes_list[:MAX_PACKETS_PER_SAMPLE])
    while len(sample) < MAX_PACKETS_PER_SAMPLE:
        sample.append(bytes.fromhex(("0" * 160) + ("0" * 480)))
    blob = b"".join(sample)  # 5 * 320 = 1600 bytes
    vec = np.frombuffer(blob, dtype=np.uint8)
    target = IMAGE_SIZE * IMAGE_SIZE
    if vec.size > target:
        vec = vec[:target]
    elif vec.size < target:
        vec = np.pad(vec, (0, target - vec.size), mode="constant", constant_values=0)
    return vec


def split_counts(n: int) -> Tuple[int, int, int]:
    tr = int(round(n * TRAIN_RATIO))
    va = int(round(n * VAL_RATIO))
    te = n - tr - va
    if n >= 3:
        if tr == 0:
            tr = 1
        if va == 0:
            va = 1
        te = n - tr - va
        if te <= 0:
            te = 1
            if tr > va:
                tr -= 1
            else:
                va -= 1
    return tr, va, te


def get_dataset_names(input_root: Path) -> List[str]:
    if DATASET_NAMES:
        return DATASET_NAMES
    return sorted([p.name for p in input_root.iterdir() if p.is_dir()])


def collect_pcaps(dataset_dir: Path, label_dir: Path) -> List[Path]:
    exts = {".pcap", ".cap"}
    return sorted([p for p in label_dir.rglob("*") if p.is_file() and p.suffix.lower() in exts])


def build_samples_for_pcap_multi_flow(pcap_path: Path) -> List[Dict]:
    # Key point: same 5-tuple within SAME pcap is one flow sample.
    flow_packets: Dict[FlowKey, List[bytes]] = defaultdict(list)

    with PcapReader(str(pcap_path)) as rd:
        for pkt in rd:
            k = flow_key_from_pkt(pkt)
            if k is None:
                continue
            if len(flow_packets[k]) >= MAX_PACKETS_PER_SAMPLE:
                continue
            b = sanitize_packet_to_yatc_bytes(pkt)
            if b is None:
                continue
            flow_packets[k].append(b)

    samples = []
    for k, pkt_bytes in flow_packets.items():
        samples.append(
            {
                "source_pcap": str(pcap_path),
                "flow_key": {
                    "proto": k[0],
                    "src_ip": k[1],
                    "dst_ip": k[2],
                    "src_port": k[3],
                    "dst_port": k[4],
                },
                "packet_repr_count": len(pkt_bytes),
                "packet_repr_bytes": pkt_bytes,
            }
        )
    return samples


def build_sample_for_pcap_single_flow(pcap_path: Path) -> Optional[Dict]:
    pkt_bytes: List[bytes] = []
    first_key = None

    with PcapReader(str(pcap_path)) as rd:
        for pkt in rd:
            if first_key is None:
                k = flow_key_from_pkt(pkt)
                if k is not None:
                    first_key = {
                        "proto": k[0],
                        "src_ip": k[1],
                        "dst_ip": k[2],
                        "src_port": k[3],
                        "dst_port": k[4],
                    }
            b = sanitize_packet_to_yatc_bytes(pkt)
            if b is None:
                continue
            pkt_bytes.append(b)
            if len(pkt_bytes) >= MAX_PACKETS_PER_SAMPLE:
                break

    if not pkt_bytes:
        return None

    return {
        "source_pcap": str(pcap_path),
        "flow_key": first_key,  # informational only
        "packet_repr_count": len(pkt_bytes),
        "packet_repr_bytes": pkt_bytes,
    }


def save_png(vec: np.ndarray, out_path: Path) -> None:
    img = Image.fromarray(vec.reshape(IMAGE_SIZE, IMAGE_SIZE))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)


def safe_id(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()[:12]


def process_dataset(input_root: Path, output_root: Path, dataset_name: str, rng: random.Random) -> Dict:
    dataset_dir = input_root / dataset_name
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Dataset folder not found: {dataset_dir}")

    mode = DATASET_MODES.get(dataset_name, "single_flow_per_pcap")
    if mode not in {"multi_flow_per_pcap", "single_flow_per_pcap"}:
        raise ValueError(f"Unsupported mode for {dataset_name}: {mode}")

    out_dataset = output_root / dataset_name
    if OVERWRITE_OUTPUT_DATASET and out_dataset.exists():
        shutil.rmtree(out_dataset)

    labels = sorted([p for p in dataset_dir.iterdir() if p.is_dir()])
    manifest = []
    summary = {
        "dataset": dataset_name,
        "mode": mode,
        "labels": {},
        "total_samples": 0,
    }

    label_iter = tqdm(labels, desc=f"Building {dataset_name}", unit="label")
    for label_dir in label_iter:
        label = label_dir.name
        pcaps = collect_pcaps(dataset_dir, label_dir)
        label_samples = []

        pcap_iter = pcaps
        if SHOW_DETAILED_PROGRESS:
            pcap_iter = tqdm(
                pcaps,
                desc=f"{dataset_name}/{label} read",
                unit="pcap",
                leave=False,
            )

        for pcap in pcap_iter:
            if mode == "multi_flow_per_pcap":
                samples = build_samples_for_pcap_multi_flow(pcap)
                for s in samples:
                    label_samples.append(s)
            else:
                sample = build_sample_for_pcap_single_flow(pcap)
                if sample is not None:
                    label_samples.append(sample)

        rng.shuffle(label_samples)
        tr, va, te = split_counts(len(label_samples))
        train_samples = label_samples[:tr]
        val_samples = label_samples[tr:tr + va]
        test_samples = label_samples[tr + va:tr + va + te]

        split_map = {
            "train": train_samples,
            "val": val_samples,
            "test": test_samples,
        }

        idx = 0
        total_to_write = len(train_samples) + len(val_samples) + len(test_samples)
        write_pbar = None
        if SHOW_DETAILED_PROGRESS:
            write_pbar = tqdm(
                total=total_to_write,
                desc=f"{dataset_name}/{label} write",
                unit="png",
                leave=False,
            )

        for split, items in split_map.items():
            for s in items:
                vec = sample_to_image_vec(s["packet_repr_bytes"])
                source = s["source_pcap"]
                flow_part = ""
                if s.get("flow_key") is not None:
                    fk = s["flow_key"]
                    flow_part = f"_{safe_id(json.dumps(fk, sort_keys=True))}"
                name = f"{Path(source).stem}{flow_part}_{idx}.png"
                out_path = out_dataset / split / label / name
                save_png(vec, out_path)

                if SAVE_MANIFEST:
                    manifest.append(
                        {
                            "dataset": dataset_name,
                            "label": label,
                            "split": split,
                            "output_png": str(out_path),
                            "source_pcap": source,
                            "flow_key": s.get("flow_key"),
                            "packet_repr_count": s["packet_repr_count"],
                        }
                    )
                idx += 1
                if write_pbar is not None:
                    write_pbar.update(1)

        if write_pbar is not None:
            write_pbar.close()

        summary["labels"][label] = {
            "pcap_count": len(pcaps),
            "sample_count": len(label_samples),
            "train": len(train_samples),
            "val": len(val_samples),
            "test": len(test_samples),
        }
        summary["total_samples"] += len(label_samples)
        label_iter.set_postfix(
            label=label,
            samples=len(label_samples),
            train=len(train_samples),
            val=len(val_samples),
            test=len(test_samples),
        )

    if SAVE_MANIFEST:
        manifest_path = out_dataset / MANIFEST_NAME
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, ensure_ascii=False)
        summary["manifest_path"] = str(manifest_path)

    return summary


def main() -> None:
    ensure_ratio_valid()
    input_root = INPUT_ROOT.resolve()
    output_root = OUTPUT_ROOT.resolve()

    if not input_root.exists():
        raise FileNotFoundError(f"INPUT_ROOT not found: {input_root}")
    output_root.mkdir(parents=True, exist_ok=True)

    datasets = get_dataset_names(input_root)
    rng = random.Random(RANDOM_SEED)
    all_summary = []

    for d in datasets:
        print(f"\n=== Processing dataset: {d} ===")
        summary = process_dataset(input_root, output_root, d, rng)
        all_summary.append(summary)
        print(
            f"Done {d}: total_samples={summary['total_samples']}, "
            f"mode={summary['mode']}"
        )

    summary_path = output_root / "build_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(all_summary, f, indent=2, ensure_ascii=False)
    print(f"\nAll done. Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
