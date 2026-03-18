"""Compute dispatcher: routes tasks to Targon (CPU/inference), Lium (GPU training), or Chutes (agent inference)."""
from __future__ import annotations

from typing import Any

from src.common.config import get_settings
from src.common.logging import get_logger
from src.compute.targon_client import TargonClient
from src.compute.lium_client import LiumClient
from src.compute.chutes_client import ChutesClient

logger = get_logger(__name__)


class ComputeDispatcher:
    """Routes compute tasks to the appropriate decentralized provider.

    Routing logic:
    - Agent reasoning / code gen / reflection → Chutes (serverless, fast, high-throughput)
    - CPU-bound tasks (feature engineering, sklearn) → Targon CPU
    - GPU-bound training (LightGBM GPU, PyTorch) → Lium GPU pods
    - Fallback: local execution
    """

    def __init__(self):
        self._targon: TargonClient | None = None
        self._lium: LiumClient | None = None
        self._chutes: ChutesClient | None = None
        self._initialized = False

    def _lazy_init(self) -> None:
        if self._initialized:
            return
        try:
            self._targon = TargonClient()
        except Exception as e:
            logger.warning("targon_init_failed", error=str(e))
        try:
            self._lium = LiumClient()
        except Exception as e:
            logger.warning("lium_init_failed", error=str(e))
        try:
            self._chutes = ChutesClient()
        except Exception as e:
            logger.warning("chutes_init_failed", error=str(e))
        self._initialized = True

    @property
    def targon(self) -> TargonClient:
        self._lazy_init()
        if self._targon is None:
            raise RuntimeError("Targon client not available")
        return self._targon

    @property
    def lium(self) -> LiumClient:
        self._lazy_init()
        if self._lium is None:
            raise RuntimeError("Lium client not available")
        return self._lium

    @property
    def chutes(self) -> ChutesClient:
        self._lazy_init()
        if self._chutes is None:
            raise RuntimeError("Chutes client not available")
        return self._chutes

    def agent_inference(
        self,
        prompt: str,
        system: str = "You are an expert quantitative researcher.",
        **kwargs: Any,
    ) -> str:
        """Route agent reasoning to Chutes (primary) or Targon (fallback)."""
        self._lazy_init()

        # Try Chutes first (serverless, optimized for inference)
        if self._chutes:
            try:
                return self._chutes.generate(prompt, system=system, **kwargs)
            except Exception as e:
                logger.warning("chutes_inference_failed", error=str(e))

        # Fallback to Targon
        if self._targon:
            try:
                return self._targon.generate(prompt, system=system, **kwargs)
            except Exception as e:
                logger.warning("targon_inference_failed", error=str(e))

        raise RuntimeError("No inference provider available (Chutes or Targon)")

    def agent_inference_json(self, prompt: str, system: str = "", **kwargs: Any) -> dict:
        """Route structured JSON inference."""
        self._lazy_init()
        if self._chutes:
            try:
                return self._chutes.generate_json(prompt, system=system, **kwargs)
            except Exception:
                pass
        if self._targon:
            return self._targon.generate_json(prompt, system=system, **kwargs)
        raise RuntimeError("No inference provider available")

    def submit_gpu_training(
        self,
        script_content: str,
        gpu_type: str = "A100",
        n_gpus: int = 1,
        ttl: str = "4h",
    ) -> dict[str, Any]:
        """Submit GPU training job to Lium."""
        self._lazy_init()
        if self._lium:
            return self._lium.submit_distributed_training(
                script_content=script_content,
                gpu_type=gpu_type,
                n_gpus=n_gpus,
                ttl=ttl,
            )
        raise RuntimeError("Lium client not available")

    def health(self) -> dict[str, bool]:
        """Check health of all compute providers."""
        self._lazy_init()
        return {
            "targon": self._targon.health_check() if self._targon else False,
            "chutes": self._chutes.health_check() if self._chutes else False,
            "lium": self._lium is not None,
        }
