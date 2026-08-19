"""Hardware detection via Lemonade Server's own /api/v1/system-info endpoint —
it already knows the CPU/GPU/NPU family and driver state; no need to re-derive
that from WMI queries.
"""
from __future__ import annotations

import json
import re
import urllib.error
import urllib.request
from dataclasses import dataclass


@dataclass
class SystemInfo:
    cpu: str
    memory_gb: int
    gpu: str
    npu: str
    os: str

    def as_dict(self) -> dict:
        return {
            "cpu": self.cpu,
            "memory": f"{self.memory_gb}GB",
            "gpu": self.gpu,
            "npu": self.npu,
            "os": self.os,
        }


class LemonadeServerUnreachable(RuntimeError):
    pass


def get_system_info(host: str, port: int) -> SystemInfo:
    url = f"http://{host}:{port}/api/v1/system-info"
    try:
        with urllib.request.urlopen(url, timeout=10) as resp:
            data = json.load(resp)
    except (urllib.error.URLError, TimeoutError) as e:
        raise LemonadeServerUnreachable(
            f"Couldn't reach Lemonade Server at {url} ({e}). "
            f"Start it (LemonadeServer.exe) and try again."
        ) from e

    devices = data.get("devices", {})
    cpu = devices.get("cpu", {}).get("name") or data.get("Processor", "Unknown CPU")

    gpus = [g["name"] for g in devices.get("amd_gpu", []) if g.get("available")]
    gpus += [g["name"] for g in devices.get("nvidia_gpu", []) if g.get("available")]
    gpu = ", ".join(gpus) if gpus else "None detected"

    npu_info = devices.get("amd_npu", {})
    if npu_info.get("available"):
        npu = f"AMD {npu_info.get('family', 'NPU')} (ready)"
    else:
        npu = "Not detected"

    mem_match = re.search(r"([\d.]+)", data.get("Physical Memory", ""))
    memory_gb = round(float(mem_match.group(1))) if mem_match else 0

    return SystemInfo(cpu=cpu, memory_gb=memory_gb, gpu=gpu, npu=npu, os=data.get("OS Version", "Unknown"))


if __name__ == "__main__":
    from bench.config import load_config

    cfg = load_config()
    info = get_system_info(cfg.lemonade_host, cfg.lemonade_port)
    print(json.dumps(info.as_dict(), indent=2))
