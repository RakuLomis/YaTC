import json
import hashlib
import difflib
from pathlib import Path
from statistics import mean
from typing import Any, Dict, List, Optional, Tuple

from scapy.all import PcapReader, raw
from scapy.packet import Raw
from scapy.layers.inet import IP, TCP, UDP
from scapy.layers.inet6 import IPv6

try:
    from scapy.layers.tls.handshake import TLSClientHello
except Exception:
    TLSClientHello = None

try:
    from scapy.layers.tls.extensions import TLS_Ext_ServerName
except Exception:
    TLS_Ext_ServerName = None


# ===== Edit here =====
PCAP_A = Path(
    r"E:\Coding\TrafficData\datasets_raw_add2\CipherSpectrum\adblockplus.org\traffic_2024-01-22_adblockplus.org_chacha20_firefox_7.pcap.TCP_10-0-2-15_44910_148-251-232-132_443.pcap"
)
PCAP_B = Path(
    r"E:\Coding\TrafficData\datasets_raw_add2\CipherSpectrum\adblockplus.org\traffic_2024-02-14_adblockplus.org_aes-128_firefox_34.pcap.TCP_10-0-2-15_44910_148-251-232-132_443.pcap"
)
REPORT_PATH = Path("./pcap_compare_report.json")
MAX_STREAM_BYTES = 2_000_000
YATC_NUM_PACKETS = 5
YATC_HEADER_HEX_LEN = 160   # 80 bytes
YATC_PAYLOAD_HEX_LEN = 480  # 240 bytes
YATC_COMPARE_FIRST_N_BYTES = 1600  # full YaTC representation by default


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def normalize_hex(s: str, target_len: int) -> str:
    if len(s) > target_len:
        return s[:target_len]
    return s + ("0" * (target_len - len(s)))


def yatc_packet_bytes(pkt) -> Optional[bytes]:
    # Match YaTC's original extraction logic:
    # header = IP bytes without payload; payload = Raw bytes.
    if IP in pkt:
        ip_layer = pkt[IP]
    elif IPv6 in pkt:
        ip_layer = pkt[IPv6]
    else:
        return None

    header_hex = bytes(ip_layer).hex()
    payload_hex = bytes(pkt[Raw]).hex() if Raw in pkt else ""

    if payload_hex:
        if header_hex.endswith(payload_hex):
            header_hex = header_hex[: -len(payload_hex)]
        else:
            header_hex = header_hex.replace(payload_hex, "", 1)

    header_hex = normalize_hex(header_hex, YATC_HEADER_HEX_LEN)
    payload_hex = normalize_hex(payload_hex, YATC_PAYLOAD_HEX_LEN)
    return bytes.fromhex(header_hex + payload_hex)


def yatc_representation_from_pcap(path: Path) -> Dict[str, Any]:
    pkt_bytes: List[bytes] = []

    with PcapReader(str(path)) as rd:
        for pkt in rd:
            b = yatc_packet_bytes(pkt)
            if b is None:
                continue
            pkt_bytes.append(b)
            if len(pkt_bytes) >= YATC_NUM_PACKETS:
                break

    while len(pkt_bytes) < YATC_NUM_PACKETS:
        pkt_bytes.append(bytes.fromhex(("0" * YATC_HEADER_HEX_LEN) + ("0" * YATC_PAYLOAD_HEX_LEN)))

    stream = b"".join(pkt_bytes)  # default 5 * 320 = 1600 bytes
    return {
        "packet_repr_bytes": pkt_bytes,
        "stream_repr_bytes": stream,
        "stream_repr_sha256": hashlib.sha256(stream).hexdigest(),
    }


def extract_tls_info(pkt) -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "has_client_hello": False,
        "sni": None,
        "tls_version": None,
        "cipher_suites_count": None,
        "cipher_suites_head": [],
    }

    if TLSClientHello is not None and TLSClientHello in pkt:
        ch = pkt[TLSClientHello]
        info["has_client_hello"] = True
        info["tls_version"] = int(getattr(ch, "version", 0)) if getattr(ch, "version", None) is not None else None
        ciphers = getattr(ch, "ciphers", None)
        if ciphers is not None:
            info["cipher_suites_count"] = len(ciphers)
            info["cipher_suites_head"] = [int(x) for x in list(ciphers)[:10]]

    if TLS_Ext_ServerName is not None and TLS_Ext_ServerName in pkt:
        ext = pkt[TLS_Ext_ServerName]
        names = getattr(ext, "servernames", [])
        if names:
            one = names[0]
            sni = getattr(one, "servername", None)
            if isinstance(sni, bytes):
                sni = sni.decode(errors="ignore")
            info["sni"] = sni

    return info


def packet_summary(pkt, ts: float) -> Dict[str, Any]:
    item: Dict[str, Any] = {
        "timestamp": float(ts),
        "len": len(raw(pkt)),
        "ip_version": None,
        "proto": None,
        "src_ip": None,
        "dst_ip": None,
        "src_port": None,
        "dst_port": None,
        "ttl_or_hlim": None,
        "tcp_flags": None,
        "tcp_win": None,
        "tcp_opts_len": None,
        "payload_len": 0,
    }

    if IP in pkt:
        ip = pkt[IP]
        item["ip_version"] = 4
        item["proto"] = int(ip.proto)
        item["src_ip"] = ip.src
        item["dst_ip"] = ip.dst
        item["ttl_or_hlim"] = int(ip.ttl)
    elif IPv6 in pkt:
        ip6 = pkt[IPv6]
        item["ip_version"] = 6
        item["proto"] = int(ip6.nh)
        item["src_ip"] = ip6.src
        item["dst_ip"] = ip6.dst
        item["ttl_or_hlim"] = int(ip6.hlim)

    if TCP in pkt:
        t = pkt[TCP]
        item["src_port"] = int(t.sport)
        item["dst_port"] = int(t.dport)
        item["tcp_flags"] = str(t.flags)
        item["tcp_win"] = int(t.window)
        item["tcp_opts_len"] = len(getattr(t, "options", []))
        payload = bytes(t.payload) if t.payload is not None else b""
        item["payload_len"] = len(payload)
    elif UDP in pkt:
        u = pkt[UDP]
        item["src_port"] = int(u.sport)
        item["dst_port"] = int(u.dport)
        payload = bytes(u.payload) if u.payload is not None else b""
        item["payload_len"] = len(payload)

    return item


def read_pcap(path: Path) -> Dict[str, Any]:
    packets: List[bytes] = []
    summaries: List[Dict[str, Any]] = []
    tls_hits: List[Dict[str, Any]] = []

    with PcapReader(str(path)) as rd:
        for pkt in rd:
            ts = float(getattr(pkt, "time", 0.0))
            b = raw(pkt)
            packets.append(b)
            s = packet_summary(pkt, ts)
            summaries.append(s)
            t = extract_tls_info(pkt)
            if t["has_client_hello"] or t["sni"] is not None:
                tls_hits.append(t)

    flow_key = None
    for s in summaries:
        if s["src_ip"] and s["src_port"] is not None:
            flow_key = {
                "ip_version": s["ip_version"],
                "proto": s["proto"],
                "src_ip": s["src_ip"],
                "dst_ip": s["dst_ip"],
                "src_port": s["src_port"],
                "dst_port": s["dst_port"],
            }
            break

    lens = [x["len"] for x in summaries]
    payload_lens = [x["payload_len"] for x in summaries]
    times = [x["timestamp"] for x in summaries]
    iats = [times[i] - times[i - 1] for i in range(1, len(times))] if len(times) > 1 else []

    stream = b"".join(packets)
    if len(stream) > MAX_STREAM_BYTES:
        stream = stream[:MAX_STREAM_BYTES]

    return {
        "path": str(path.resolve()),
        "sha256": file_sha256(path),
        "packet_count": len(packets),
        "flow_key_first_seen": flow_key,
        "lens": lens,
        "payload_lens": payload_lens,
        "ttl_or_hlim_head": [x["ttl_or_hlim"] for x in summaries[:20]],
        "tcp_flags_head": [x["tcp_flags"] for x in summaries[:20]],
        "tcp_win_head": [x["tcp_win"] for x in summaries[:20]],
        "tls_hits_head": tls_hits[:5],
        "avg_pkt_len": mean(lens) if lens else 0.0,
        "avg_payload_len": mean(payload_lens) if payload_lens else 0.0,
        "avg_iat": mean(iats) if iats else 0.0,
        "raw_packets": packets,
        "raw_stream": stream,
    }


def common_prefix_len(a: bytes, b: bytes) -> int:
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


def packet_level_similarity(pkts_a: List[bytes], pkts_b: List[bytes]) -> Dict[str, Any]:
    n = min(len(pkts_a), len(pkts_b))
    exact_same_count = 0
    prefix_lens = []
    max_prefix = 0

    for i in range(n):
        pa, pb = pkts_a[i], pkts_b[i]
        if pa == pb:
            exact_same_count += 1
        pref = common_prefix_len(pa, pb)
        prefix_lens.append(pref)
        if pref > max_prefix:
            max_prefix = pref

    return {
        "aligned_packet_count": n,
        "exact_same_packet_count": exact_same_count,
        "exact_same_packet_ratio": (exact_same_count / n) if n else 0.0,
        "avg_common_prefix_bytes_per_packet": mean(prefix_lens) if prefix_lens else 0.0,
        "max_common_prefix_bytes_per_packet": max_prefix,
    }


def first_n_byte_similarity(a: bytes, b: bytes, n: int) -> Dict[str, Any]:
    m = min(len(a), len(b), n)
    if m == 0:
        return {
            "n_requested": n,
            "n_compared": 0,
            "equal_bytes": 0,
            "equal_ratio": 0.0,
            "common_prefix_len": 0,
        }
    aa = a[:m]
    bb = b[:m]
    equal_bytes = sum(1 for i in range(m) if aa[i] == bb[i])
    return {
        "n_requested": n,
        "n_compared": m,
        "equal_bytes": equal_bytes,
        "equal_ratio": equal_bytes / m,
        "common_prefix_len": common_prefix_len(aa, bb),
    }


def stream_longest_common_block(a: bytes, b: bytes) -> Dict[str, Any]:
    sm = difflib.SequenceMatcher(None, a, b, autojunk=False)
    m = sm.find_longest_match(0, len(a), 0, len(b))
    block = a[m.a:m.a + m.size]
    return {
        "longest_common_contiguous_bytes": int(m.size),
        "offset_in_a": int(m.a),
        "offset_in_b": int(m.b),
        "block_sha256": hashlib.sha256(block).hexdigest() if m.size > 0 else None,
    }


def compare_fields(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    def head_equal_ratio(x: List[Any], y: List[Any], n: int = 50) -> float:
        xx, yy = x[:n], y[:n]
        m = min(len(xx), len(yy))
        if m == 0:
            return 0.0
        eq = sum(1 for i in range(m) if xx[i] == yy[i])
        return eq / m

    return {
        "same_5tuple_first_seen": a["flow_key_first_seen"] == b["flow_key_first_seen"],
        "same_packet_count": a["packet_count"] == b["packet_count"],
        "pkt_len_head_equal_ratio_50": head_equal_ratio(a["lens"], b["lens"], 50),
        "payload_len_head_equal_ratio_50": head_equal_ratio(a["payload_lens"], b["payload_lens"], 50),
        "ttl_head_equal_ratio_20": head_equal_ratio(a["ttl_or_hlim_head"], b["ttl_or_hlim_head"], 20),
        "tcp_flags_head_equal_ratio_20": head_equal_ratio(a["tcp_flags_head"], b["tcp_flags_head"], 20),
        "tcp_win_head_equal_ratio_20": head_equal_ratio(a["tcp_win_head"], b["tcp_win_head"], 20),
        "tls_first_sni_a": a["tls_hits_head"][0]["sni"] if a["tls_hits_head"] else None,
        "tls_first_sni_b": b["tls_hits_head"][0]["sni"] if b["tls_hits_head"] else None,
        "tls_first_version_a": a["tls_hits_head"][0]["tls_version"] if a["tls_hits_head"] else None,
        "tls_first_version_b": b["tls_hits_head"][0]["tls_version"] if b["tls_hits_head"] else None,
        "tls_first_cipher_head_a": a["tls_hits_head"][0]["cipher_suites_head"] if a["tls_hits_head"] else [],
        "tls_first_cipher_head_b": b["tls_hits_head"][0]["cipher_suites_head"] if b["tls_hits_head"] else [],
    }


def main() -> None:
    if not PCAP_A.exists():
        raise FileNotFoundError(f"PCAP_A not found: {PCAP_A}")
    if not PCAP_B.exists():
        raise FileNotFoundError(f"PCAP_B not found: {PCAP_B}")

    a = read_pcap(PCAP_A)
    b = read_pcap(PCAP_B)

    packet_sim = packet_level_similarity(a["raw_packets"], b["raw_packets"])
    stream_sim = stream_longest_common_block(a["raw_stream"], b["raw_stream"])
    field_cmp = compare_fields(a, b)
    yatc_a = yatc_representation_from_pcap(PCAP_A)
    yatc_b = yatc_representation_from_pcap(PCAP_B)
    yatc_packet_sim = packet_level_similarity(yatc_a["packet_repr_bytes"], yatc_b["packet_repr_bytes"])
    yatc_first_n = first_n_byte_similarity(
        yatc_a["stream_repr_bytes"], yatc_b["stream_repr_bytes"], YATC_COMPARE_FIRST_N_BYTES
    )

    report = {
        "pcap_a": {
            "path": a["path"],
            "sha256": a["sha256"],
            "packet_count": a["packet_count"],
            "flow_key_first_seen": a["flow_key_first_seen"],
            "avg_pkt_len": a["avg_pkt_len"],
            "avg_payload_len": a["avg_payload_len"],
            "avg_iat": a["avg_iat"],
        },
        "pcap_b": {
            "path": b["path"],
            "sha256": b["sha256"],
            "packet_count": b["packet_count"],
            "flow_key_first_seen": b["flow_key_first_seen"],
            "avg_pkt_len": b["avg_pkt_len"],
            "avg_payload_len": b["avg_payload_len"],
            "avg_iat": b["avg_iat"],
        },
        "packet_level_similarity": packet_sim,
        "stream_level_similarity": stream_sim,
        "field_comparison": field_cmp,
        "yatc_representation_similarity": {
            "definition": (
                "Each packet -> header(80B)+payload(240B), first 5 packets, concat to 1600B stream."
            ),
            "num_packets": YATC_NUM_PACKETS,
            "header_bytes_per_packet": YATC_HEADER_HEX_LEN // 2,
            "payload_bytes_per_packet": YATC_PAYLOAD_HEX_LEN // 2,
            "stream_bytes": len(yatc_a["stream_repr_bytes"]),
            "a_stream_sha256": yatc_a["stream_repr_sha256"],
            "b_stream_sha256": yatc_b["stream_repr_sha256"],
            "packet_level": yatc_packet_sim,
            "first_n_byte_similarity": yatc_first_n,
        },
        "note": (
            "same_5tuple_first_seen=true does not guarantee same flow instance. "
            "Use TLS fields, packet length/time patterns and byte-level similarity together."
        ),
    }

    out = REPORT_PATH.resolve()
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"Done. Report saved to: {out}")
    print(f"Longest common contiguous bytes: {report['stream_level_similarity']['longest_common_contiguous_bytes']}")
    print(f"Aligned exact same packet ratio: {report['packet_level_similarity']['exact_same_packet_ratio']:.4f}")
    print(
        "YaTC first-N byte equal ratio: "
        f"{report['yatc_representation_similarity']['first_n_byte_similarity']['equal_ratio']:.4f}"
    )


if __name__ == "__main__":
    main()
