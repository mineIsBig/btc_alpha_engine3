"""Lium GPU rental client for model training.

Lium (lium.io) is a decentralized GPU marketplace on Bittensor SN51.
It provides GPU pods (A100, H100, H200) accessible via CLI/SDK.

We use Lium for:
1. GPU-accelerated model training (LightGBM, XGBoost, PyTorch sequence models)
2. Evolutionary search across model populations
3. Large-scale walk-forward backtesting

Architecture:
- We use the lium CLI/API to provision pods
- Upload training scripts and data
- Execute training remotely
- Download trained model artifacts
"""

from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from src.common.config import get_settings
from src.common.logging import get_logger

logger = get_logger(__name__)


class LiumPod:
    """Represents a Lium GPU pod."""

    def __init__(self, name: str, gpu_type: str = "A100", status: str = "unknown"):
        self.name = name
        self.gpu_type = gpu_type
        self.status = status
        self.ip: str | None = None
        self.ssh_port: int | None = None


class LiumClient:
    """Client for managing Lium GPU pods for training jobs.

    Wraps the `lium` CLI tool. Install with: pip install lium.io
    Authenticate with: lium init
    """

    def __init__(self, api_key: str | None = None):
        settings = get_settings()
        self.api_key = api_key or getattr(settings, "lium_api_key", "")
        self._pods: dict[str, LiumPod] = {}

    def _run_cli(
        self, args: list[str], check: bool = True
    ) -> subprocess.CompletedProcess:
        """Run a lium CLI command."""
        cmd = ["lium"] + args
        logger.debug("lium_cli", cmd=" ".join(cmd))
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            if check and result.returncode != 0:
                logger.error("lium_cli_error", stderr=result.stderr, cmd=" ".join(cmd))
            return result
        except FileNotFoundError:
            logger.error(
                "lium_cli_not_found",
                msg="Install with: pip install lium.io && lium init",
            )
            raise RuntimeError("lium CLI not installed")

    def list_available_gpus(self, gpu_type: str | None = None) -> list[dict[str, Any]]:
        """List available GPU executors."""
        args = ["ls"]
        if gpu_type:
            args.append(gpu_type)
        args.extend(["--json"])
        result = self._run_cli(args, check=False)
        if result.returncode == 0 and result.stdout.strip():
            try:
                return json.loads(result.stdout)
            except json.JSONDecodeError:
                pass
        # Fallback: parse text output
        return [{"raw": result.stdout}]

    def create_pod(
        self,
        name: str,
        gpu_type: str = "A100",
        gpu_count: int = 1,
        ttl: str | None = "6h",
    ) -> LiumPod:
        """Create a new GPU pod.

        Args:
            name: pod name
            gpu_type: GPU type (A100, H100, H200, etc.)
            gpu_count: number of GPUs
            ttl: auto-termination time (e.g. "6h", "12h")
        """
        args = [
            "up",
            "--gpu",
            gpu_type,
            "--count",
            str(gpu_count),
            "--name",
            name,
            "--yes",
        ]
        if ttl:
            args.extend(["--ttl", ttl])

        result = self._run_cli(args)
        pod = LiumPod(name=name, gpu_type=gpu_type, status="creating")
        self._pods[name] = pod

        if result.returncode == 0:
            pod.status = "running"
            logger.info("lium_pod_created", name=name, gpu=gpu_type)
        else:
            pod.status = "failed"
            logger.error("lium_pod_create_failed", name=name, error=result.stderr)

        return pod

    def upload_file(
        self, pod_name: str, local_path: str, remote_path: str = "/root/"
    ) -> bool:
        """Upload a file to a pod."""
        result = self._run_cli(["scp", pod_name, local_path, remote_path])
        return result.returncode == 0

    def execute(self, pod_name: str, command: str) -> str:
        """Execute a command on a pod."""
        result = self._run_cli(["exec", pod_name, command])
        return result.stdout

    def download_file(self, pod_name: str, remote_path: str, local_path: str) -> bool:
        """Download a file from a pod (via scp)."""
        result = self._run_cli(["scp", pod_name, f"{remote_path}", local_path])
        return result.returncode == 0

    def destroy_pod(self, pod_name: str) -> bool:
        """Destroy a pod."""
        result = self._run_cli(["rm", pod_name])
        self._pods.pop(pod_name, None)
        return result.returncode == 0

    def list_pods(self) -> list[dict[str, Any]]:
        """List active pods."""
        result = self._run_cli(["ps"], check=False)
        return [{"raw": result.stdout}]

    def submit_training_job(
        self,
        pod_name: str,
        script_path: str,
        data_path: str | None = None,
        requirements: list[str] | None = None,
    ) -> dict[str, Any]:
        """Submit a complete training job to a Lium pod.

        1. Upload script and data
        2. Install dependencies
        3. Execute training
        4. Return status
        """
        # Upload training script
        self.upload_file(pod_name, script_path, "/root/train.py")

        # Upload data if provided
        if data_path:
            self.upload_file(pod_name, data_path, "/root/data/")

        # Install dependencies
        if requirements:
            req_str = " ".join(requirements)
            self.execute(pod_name, f"pip install {req_str}")

        # Run training
        output = self.execute(pod_name, "cd /root && python train.py")

        logger.info("lium_training_submitted", pod=pod_name, script=script_path)
        return {"status": "submitted", "output": output}

    def submit_distributed_training(
        self,
        script_content: str,
        gpu_type: str = "A100",
        n_gpus: int = 1,
        ttl: str = "4h",
        pod_name: str = "alpha-train",
    ) -> dict[str, Any]:
        """High-level: create pod, upload script, run training, return results.

        For use by the autonomous agent to dispatch training jobs.
        """
        # Write script to temp file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(script_content)
            script_path = f.name

        try:
            pod = self.create_pod(
                name=pod_name, gpu_type=gpu_type, gpu_count=n_gpus, ttl=ttl
            )
            if pod.status != "running":
                return {"status": "failed", "error": "Pod creation failed"}

            # Install base deps
            self.execute(
                pod_name,
                "pip install pandas numpy scikit-learn lightgbm xgboost joblib",
            )

            # Upload and run
            self.upload_file(pod_name, script_path, "/root/train.py")
            output = self.execute(pod_name, "cd /root && python train.py 2>&1")

            return {"status": "completed", "output": output, "pod": pod_name}

        except Exception as e:
            logger.error("distributed_training_failed", error=str(e))
            return {"status": "failed", "error": str(e)}
        finally:
            # Cleanup temp file
            Path(script_path).unlink(missing_ok=True)
