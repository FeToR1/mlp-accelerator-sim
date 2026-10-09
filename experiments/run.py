import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
NETWORKS = ("784,128,10", "784,256,128,10", "784,512,256,128,10")


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description="MLP communication and memory profiles")
    parser.add_argument("--exe", type=Path, default=ROOT / "build" / (
        "mlp_accelerator_sim.exe" if sys.platform == "win32" else "mlp_accelerator_sim"))
    parser.add_argument("--out", type=Path, default=ROOT / "experiments" / "results")
    parser.add_argument("--batch", type=int, default=8)
    args = parser.parse_args()
    if args.batch <= 0:
        parser.error("batch must be positive")
    executable = args.exe.resolve()
    if not executable.is_file():
        parser.error(f"Build simulator first or set --exe: {executable}")
    destination = args.out.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    cases = [(NETWORKS[0], pe) for pe in (1, 2, 4, 8)]
    cases += [(network, 4) for network in NETWORKS[1:]]
    summary, memory_rows, pe_rows, transaction_rows = [], [], [], []
    for network, pe_count in cases:
        profile_path = destination / f"{network.replace(',', '-')}_pe{pe_count}.json"
        result = subprocess.run([
            str(executable), "--mlp", network, str(pe_count), str(args.batch),
            "--profile", str(profile_path),
        ], capture_output=True, text=True)
        if result.returncode:
            raise SystemExit(result.stdout + result.stderr)
        profile = json.loads(profile_path.read_text(encoding="utf-8"))
        identity = {"network": network.replace(",", "->"),
                    "pe_count": pe_count, "batch": args.batch}
        memory = profile["memory"]
        macs = sum(pe["macs"] for pe in profile["pe"])
        capacity = sum(pe["capacity_macs"] for pe in profile["pe"])
        summary.append({
            **identity,
            "model_time": profile["model_time"],
            "transactions": profile["transactions"],
            "transferred_bytes": profile["bytes"],
            "ram_bytes": memory["ram"],
            "onchip_bytes": memory["total"] - memory["ram"],
            "total_memory_bytes": memory["total"],
            "macs": macs,
            "utilization": macs / capacity if capacity else 0.0,
        })
        memory_rows.append({**identity, **memory})
        pe_rows.extend({**identity, **pe} for pe in profile["pe"])
        transaction_rows.extend({**identity, "type": kind, **stats}
                                for kind, stats in profile["by_type"].items())
        print(f"{identity['network']} PE={pe_count}: "
              f"time={profile['model_time']:g}, tx={profile['transactions']}, "
              f"memory={memory['total']} B, MAC={macs}")
    for name, rows in (("summary", summary), ("memory", memory_rows),
                       ("pe", pe_rows), ("transactions", transaction_rows)):
        write_csv(destination / f"{name}.csv", rows)
    print(f"CSV: {destination}")


if __name__ == "__main__":
    main()
