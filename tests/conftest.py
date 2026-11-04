"""测试公共夹具：离线配置 + 隔离数据目录，避免污染真实 data/。"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from graphrag.bootstrap import Container  # noqa: E402
from graphrag.config import Settings  # noqa: E402


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    base = Settings.from_env()
    return base.model_copy(
        update={
            "data_dir": tmp_path / "data",
            "embedding_backend": "hashing",
            "llm_api_key": "",
            "adaptive_retrieval": True,
        }
    )


@pytest.fixture
def container(settings: Settings) -> Container:
    return Container(settings)


@pytest.fixture
def samples_dir() -> Path:
    return ROOT / "samples"
