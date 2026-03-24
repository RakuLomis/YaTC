import argparse
from pathlib import Path
from typing import Iterable, List

import numpy as np
from PIL import Image
from scapy.all import PcapReader
from scapy.layers.inet import IP, TCP, UDP
from scapy.layers.inet6 import IPv6
from scapy.layers.l2 import Ether

try:
    from scapy.layers.tls.extensions import TLS_Ext_ServerName
except Exception:
    TLS_Ext_ServerName = None

# ===== Local defaults (edit these in Python directly) =====
DEFAULT_INPUT_ROOT = "../../TrafficData/datasets_raw_add2/"
DEFAULT_OUTPUT_ROOT = "../../TrafficData/data_png/"
DEFAULT_MAX_PACKETS = 5
DEFAULT_IMAGE_SIZE = 40
DEFAULT_MASK_IP_PORT = True
DEFAULT_MASK_TLS_SNI = True
DEFAULT_VERBOSE = True
# Only process these dataset folders under input_root.
# Example: input_root/ISCX-VPN, input_root/tls1.3, input_root/CipherSpectrum
DEFAULT_DATASET_NAMES = ["ISCX-VPN", "cstnet_tls_1.3", "CipherSpectrum"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert pcap/cap files to 40x40 PNG with sensitive-field masking."
    )
    parser.add_argument(
        "--input_root",
        type=str,
        default=DEFAULT_INPUT_ROOT,
        help="Root folder of pcaps, e.g. ./TrafficData",
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default=DEFAULT_OUTPUT_ROOT,
        help="Root output folder for generated pngs.",
    )
    parser.add_argument(
        "--max_packets",
        type=int,
        default=DEFAULT_MAX_PACKETS,
        help="Use first N packets of each flow/file (default: 5).",
    )
    parser.add_argument(
        "--image_size",
        type=int,
        default=DEFAULT_IMAGE_SIZE,
        help="Output image side length (default: 40).",
    )
    parser.add_argument(
        "--mask_ip_port",
        action="store_true",
        default=DEFAULT_MASK_IP_PORT,
        help="Mask IP addresses and ports (default: enabled).",
    )
    parser.add_argument(
        "--no_mask_ip_port",
        action="store_false",
        dest="mask_ip_port",
        help="Disable IP/port masking.",
    )
    parser.add_argument(
        "--mask_tls_sni",
        action="store_true",
        default=DEFAULT_MASK_TLS_SNI,
        help="Mask/remove TLS SNI if present (default: enabled).",
    )
    parser.add_argument(
        "--no_mask_tls_sni",
        action="store_false",
        dest="mask_tls_sni",
        help="Disable TLS SNI handling.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=DEFAULT_VERBOSE,
        help="Print progress details.",
    )
    parser.add_argument(
        "--dataset_names",
        nargs="*",
        default=DEFAULT_DATASET_NAMES,
        help="Only process these dataset folder names under input_root.",
    )
    return parser.parse_args()


def iter_capture_files(root: Path, dataset_names: List[str]) -> Iterable[Path]:
    exts = {".pcap", ".cap"}
    if dataset_names:
        dataset_set = set(dataset_names)
        for dataset_dir in sorted(root.iterdir()):
            if not dataset_dir.is_dir():
                continue
            if dataset_dir.name not in dataset_set:
                continue
            for p in dataset_dir.rglob("*"):
                if p.is_file() and p.suffix.lower() in exts:
                    yield p
    else:
        for p in root.rglob("*"):
            if p.is_file() and p.suffix.lower() in exts:
                yield p


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
    if TLS_Ext_ServerName is None:
        return
    if TLS_Ext_ServerName not in pkt:
        return

    handled = False
    layer = pkt.getlayer(TLS_Ext_ServerName)
    if layer is not None and hasattr(layer, "servernames"):
        try:
            for sn in layer.servernames:
                if hasattr(sn, "servername"):
                    if isinstance(sn.servername, bytes):
                        sn.servername = b"masked.local"
                    else:
                        sn.servername = "masked.local"
            handled = True
        except Exception:
            handled = False

    # Fallback: if direct masking fails, remove TCP payload to ensure SNI is gone.
    if not handled and TCP in pkt:
        pkt[TCP].remove_payload()
        pkt[TCP].chksum = None
        if IP in pkt:
            pkt[IP].len = None
            pkt[IP].chksum = None


def sanitize_packet_bytes(pkt, mask_ip_port: bool, mask_tls_sni: bool) -> bytes:
    p = pkt.copy()

    if mask_ip_port:
        _mask_ip_and_port(p)
    if mask_tls_sni:
        _mask_or_drop_tls_sni(p)

    # Remove whole ETH layer by taking payload beneath Ether.
    if Ether in p:
        return bytes(p[Ether].payload)
    if IP in p:
        return bytes(p[IP])
    if IPv6 in p:
        return bytes(p[IPv6])
    return bytes(p)


def packet_bytes_to_header_payload_hex(raw_pkt: bytes) -> str:
    hex_str = raw_pkt.hex()
    header = hex_str[:160].ljust(160, "0")
    payload = hex_str[160:160 + 480].ljust(480, "0")
    return header + payload


def file_to_mfr_vector(
    cap_path: Path, max_packets: int, mask_ip_port: bool, mask_tls_sni: bool
) -> np.ndarray:
    pieces: List[str] = []

    with PcapReader(str(cap_path)) as reader:
        for pkt in reader:
            raw_pkt = sanitize_packet_bytes(pkt, mask_ip_port=mask_ip_port, mask_tls_sni=mask_tls_sni)
            pieces.append(packet_bytes_to_header_payload_hex(raw_pkt))
            if len(pieces) >= max_packets:
                break

    while len(pieces) < max_packets:
        pieces.append(("0" * 160) + ("0" * 480))

    all_hex = "".join(pieces)
    vec = np.array([int(all_hex[i:i + 2], 16) for i in range(0, len(all_hex), 2)], dtype=np.uint8)
    return vec


def convert_one(
    cap_path: Path,
    input_root: Path,
    output_root: Path,
    image_size: int,
    max_packets: int,
    mask_ip_port: bool,
    mask_tls_sni: bool,
) -> Path:
    vec = file_to_mfr_vector(
        cap_path=cap_path,
        max_packets=max_packets,
        mask_ip_port=mask_ip_port,
        mask_tls_sni=mask_tls_sni,
    )

    target_len = image_size * image_size
    if vec.size < target_len:
        vec = np.pad(vec, (0, target_len - vec.size), mode="constant", constant_values=0)
    elif vec.size > target_len:
        vec = vec[:target_len]

    img = Image.fromarray(vec.reshape(image_size, image_size))

    rel = cap_path.relative_to(input_root)
    out_path = (output_root / rel).with_suffix(".png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)
    return out_path


def main() -> None:
    args = parse_args()
    input_root = Path(args.input_root).resolve()
    output_root = Path(args.output_root).resolve()

    files = sorted(iter_capture_files(input_root, args.dataset_names))
    if not files:
        raise FileNotFoundError(f"No .pcap/.cap files found under: {input_root}")

    count = 0
    for cap_file in files:
        out_file = convert_one(
            cap_path=cap_file,
            input_root=input_root,
            output_root=output_root,
            image_size=args.image_size,
            max_packets=args.max_packets,
            mask_ip_port=args.mask_ip_port,
            mask_tls_sni=args.mask_tls_sni,
        )
        count += 1
        if args.verbose:
            print(f"[{count}/{len(files)}] {cap_file} -> {out_file}")

    print(f"Done. Converted {count} files.")
    print(f"Input : {input_root}")
    print(f"Output: {output_root}")
    if args.dataset_names:
        print(f"Datasets: {args.dataset_names}")


if __name__ == "__main__":
    main()
