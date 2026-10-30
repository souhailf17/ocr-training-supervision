"""Launch one four-H200 microbatch-3 continuation from checkpoint 7878."""
import json
import os
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROJECT = "q35ocr-20260921-p560"
NAME = PROJECT + "-train-fast-gpu4-m3-from7878-v1"
PARENT_JOB = PROJECT + "-train-fast-gpu4-m1-from6343-v1"
S3_ENDPOINT = os.environ["Q35OCR_S3_ENDPOINT"]
assert S3_ENDPOINT.startswith(("http://", "https://"))


def oc(args, **kwargs):
    return subprocess.run(["oc", "-n", "mtrna-ns1"] + args, text=True, check=True, **kwargs)


existing = oc(["get", "job", NAME, "--ignore-not-found", "-o", "json"], capture_output=True)
if existing.stdout.strip():
    print("Existing job; inspect, do not recreate:", NAME)
    raise SystemExit(0)

pods = json.loads(oc(["get", "pods", "-l", "project=" + PROJECT, "-o", "json"], capture_output=True).stdout)
for item in pods["items"]:
    if item.get("status", {}).get("phase") not in ("Running", "Pending"):
        continue
    requested = sum(
        int(container.get("resources", {}).get("requests", {}).get("nvidia.com/gpu", 0))
        for container in item["spec"].get("containers", [])
    )
    assert not requested, ("Another project GPU job is active", item["metadata"]["name"])

parent = json.loads(oc(["get", "job", PARENT_JOB, "-o", "json"], capture_output=True).stdout)
assert parent["status"].get("succeeded") == 1 and not parent["status"].get("active")
pod = parent["spec"]["template"]["spec"]
for key in ("nodeName", "nodeSelector", "affinity", "priorityClassName"):
    pod.pop(key, None)
pod["priority"] = 0
pod["terminationGracePeriodSeconds"] = 600

files = [
    "train_optimized_resume_gpu4_m3_from7878_v1.py", "train_optimized.py", "ordered_prefetch.py",
    "pilot_train.py", "large_batch.py", "distributed_ocr.py", "train_full.py",
    "arsenal_ocr.py", "lora.py", "training_config.py", "stage_training.py", "s3_common.py",
]
data = {name: (ROOT / "scripts" / name).read_text(encoding="utf-8") for name in files}
data["model-tree.json"] = (ROOT / "evidence" / "model-tree.json").read_text(encoding="utf-8")
cm = {
    "apiVersion": "v1", "kind": "ConfigMap",
    "metadata": {"name": NAME + "-code", "labels": {"project": PROJECT}},
    "data": data,
}
code_volume = next(volume for volume in pod["volumes"] if volume["name"] == "code")
code_volume["configMap"]["name"] = cm["metadata"]["name"]
container = pod["containers"][0]
container["name"] = "train4m3"
environment = {item["name"]: item["value"] for item in container["env"] if "value" in item}
environment.update(
    TRAIN_RANKS="4", MICROBATCH="3", TRAIN_MODE="train",
    CUDA_VISIBLE_DEVICES="0,1,2,3", AWS_ENDPOINT_URL=S3_ENDPOINT,
    PYTHONPATH=(
        "/work/" + PROJECT + "/optimization/triton371/site:"
        "/work/" + PROJECT + "/optimization/kernels-v1/site:/project"
    ),
)
container["env"] = [{"name": key, "value": value} for key, value in environment.items()]
container["resources"]["requests"].update({"nvidia.com/gpu": "4", "memory": "96Gi", "cpu": "12"})
container["resources"]["limits"].update({"nvidia.com/gpu": "4", "memory": "160Gi", "cpu": "32"})
container["args"] = [
    "set -euo pipefail; /workspace/.venv/bin/python -u "
    "/project/train_optimized_resume_gpu4_m3_from7878_v1.py 2>&1 | tee -a /work/"
    + PROJECT + "/logs/" + NAME + ".log"
]
job = {
    "apiVersion": "batch/v1", "kind": "Job",
    "metadata": {"name": NAME, "labels": {"project": PROJECT}},
    "spec": {
        "backoffLimit": 0,
        "template": {"metadata": {"labels": {"project": PROJECT}}, "spec": pod},
    },
}
assert all(key not in pod for key in ("nodeName", "nodeSelector", "affinity", "priorityClassName"))
assert pod["priority"] == 0
assert container["resources"]["requests"]["nvidia.com/gpu"] == "4"
assert environment["TRAIN_RANKS"] == "4" and environment["MICROBATCH"] == "3"
for resource in (cm, job):
    oc(["create", "-f", "-"], input=json.dumps(resource))
print(NAME)
