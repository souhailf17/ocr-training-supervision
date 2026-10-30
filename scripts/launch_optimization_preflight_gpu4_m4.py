"""Create an isolated four-H200 microbatch-4/BF16 preflight-only Job.

This launcher never starts full training: Q35OCR_PREFLIGHT_ONLY is mandatory.
It also refuses to proceed while another project pod requests GPUs.
"""
import copy
import json
import os
import re
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROJECT = "q35ocr-20260921-p560"
NAMESPACE = "mtrna-ns1"
PARENT_JOB = os.environ["Q35OCR_PARENT_JOB"]
PARENT_STEP = int(os.environ["Q35OCR_PARENT_STEP"])
PARENT_MARKER = os.environ["Q35OCR_PARENT_MARKER"]
S3_ENDPOINT = os.environ["Q35OCR_S3_ENDPOINT"]
MICROBATCH = 4
RUN_DIR = f"training-fast-gpu4-m4-from{PARENT_STEP}-bf16-preflight-v1"
NAME = f"{PROJECT}-opt-preflight-gpu4-m4-from{PARENT_STEP}-v1"

assert S3_ENDPOINT.startswith(("http://", "https://"))
assert PARENT_STEP >= 0
assert PARENT_MARKER.startswith(f"/work/{PROJECT}/") and ".." not in PARENT_MARKER.split("/")
assert re.fullmatch(r"[a-z0-9][a-z0-9-]{0,52}", PARENT_JOB)


def oc(args, **kwargs):
    return subprocess.run(
        ["oc", "-n", NAMESPACE] + args,
        text=True,
        check=True,
        **kwargs,
    )


def set_env(container, name, value):
    """Update one variable without dropping Secret-backed environment entries."""
    for item in container.setdefault("env", []):
        if item.get("name") == name:
            item.pop("valueFrom", None)
            item["value"] = value
            return
    container["env"].append({"name": name, "value": value})


existing = oc(["get", "job", NAME, "--ignore-not-found", "-o", "json"], capture_output=True)
if existing.stdout.strip():
    print("Existing preflight; inspect it instead of creating another:", NAME)
    raise SystemExit(0)

pods = json.loads(oc(["get", "pods", "-l", "project=" + PROJECT, "-o", "json"], capture_output=True).stdout)
for item in pods["items"]:
    if item.get("status", {}).get("phase") not in ("Running", "Pending"):
        continue
    requested = sum(
        int(container.get("resources", {}).get("requests", {}).get("nvidia.com/gpu", 0))
        for container in item["spec"].get("containers", [])
    )
    if requested:
        raise RuntimeError(f"Project GPU pod is active; do not overlap the preflight: {item['metadata']['name']}")

parent = json.loads(oc(["get", "job", PARENT_JOB, "-o", "json"], capture_output=True).stdout)
if parent.get("status", {}).get("active"):
    raise RuntimeError("Parent Job is still active; preserve it and wait for a safe checkpoint/hold")
pod = copy.deepcopy(parent["spec"]["template"]["spec"])
if pod.get("nodeName"):
    raise RuntimeError("Parent pod template pins nodeName; scheduler placement must remain automatic")
if pod.get("priorityClassName"):
    raise RuntimeError("Parent pod template has an explicit PriorityClass; refuse to inherit it")
pod["priority"] = 0
pod["terminationGracePeriodSeconds"] = 600

files = [
    "train_optimized_resume_gpu4_m3_from7878_v1.py",
    "train_optimized.py",
    "ordered_prefetch.py",
    "pilot_train.py",
    "large_batch.py",
    "distributed_ocr.py",
    "train_full.py",
    "arsenal_ocr.py",
    "lora.py",
    "training_config.py",
    "stage_training.py",
    "s3_common.py",
]
data = {name: (ROOT / "scripts" / name).read_text(encoding="utf-8") for name in files}
data["model-tree.json"] = (ROOT / "evidence" / "model-tree.json").read_text(encoding="utf-8")
config_map = {
    "apiVersion": "v1",
    "kind": "ConfigMap",
    "metadata": {"name": NAME + "-code", "labels": {"project": PROJECT}},
    "data": data,
}

code_volume = next(volume for volume in pod["volumes"] if volume["name"] == "code")
code_volume["configMap"]["name"] = config_map["metadata"]["name"]
container = next(
    container
    for container in pod["containers"]
    if int(container.get("resources", {}).get("requests", {}).get("nvidia.com/gpu", 0)) > 0
)
container["name"] = "train4m4preflight"
env_values = {
    "TRAIN_RANKS": "4",
    "MICROBATCH": str(MICROBATCH),
    "TRAIN_MODE": "train",
    "CUDA_VISIBLE_DEVICES": "0,1,2,3",
    "AWS_ENDPOINT_URL": S3_ENDPOINT,
    "Q35OCR_LORA_COMPUTE_DTYPE": "bf16",
    "Q35OCR_RANK_SYNC_INTERVAL": "16",
    "Q35OCR_PIN_EACH_SAMPLE": "0",
    "Q35OCR_PREFLIGHT_ONLY": "1",
    "Q35OCR_PARENT_MARKER": PARENT_MARKER,
    "Q35OCR_PARENT_STEP": str(PARENT_STEP),
    "Q35OCR_RUN_DIR_NAME": RUN_DIR,
    "PYTHONPATH": (
        f"/work/{PROJECT}/optimization/triton371/site:"
        f"/work/{PROJECT}/optimization/kernels-v1/site:/project"
    ),
}
for key, value in env_values.items():
    set_env(container, key, value)
container.setdefault("resources", {}).setdefault("requests", {})["nvidia.com/gpu"] = "4"
container.setdefault("resources", {}).setdefault("limits", {})["nvidia.com/gpu"] = "4"
container["args"] = [
    "set -euo pipefail; /workspace/.venv/bin/python -u "
    "/project/train_optimized_resume_gpu4_m3_from7878_v1.py 2>&1 | tee -a /work/"
    + PROJECT
    + "/logs/"
    + NAME
    + ".log"
]

job = {
    "apiVersion": "batch/v1",
    "kind": "Job",
    "metadata": {"name": NAME, "labels": {"project": PROJECT, "purpose": "optimization-preflight"}},
    "spec": {
        "backoffLimit": 0,
        "template": {
            "metadata": {"labels": {"project": PROJECT, "purpose": "optimization-preflight"}},
            "spec": pod,
        },
    },
}

assert env_values["Q35OCR_PREFLIGHT_ONLY"] == "1"
assert env_values["MICROBATCH"] == "4" and env_values["TRAIN_RANKS"] == "4"
assert "nodeName" not in pod
assert container["resources"]["requests"]["nvidia.com/gpu"] == "4"
assert container["resources"]["limits"]["nvidia.com/gpu"] == "4"
for resource in (config_map, job):
    oc(["create", "-f", "-"], input=json.dumps(resource))
print(NAME)
