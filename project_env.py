import os
from pathlib import Path
from typing import Optional


REPO_ROOT = Path(__file__).resolve().parent


def load_dotenv(path: Optional[Path] = None, *, prefix: str = "") -> None:
    """加载项目配置，优先保留调用者设置的环境变量。"""
    env_path = path or (REPO_ROOT / ".env")
    if not env_path.exists():
        return

    for raw_line in env_path.read_text(encoding="utf-8-sig").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if key and key.startswith(prefix):
            os.environ.setdefault(key, value.strip().strip('"').strip("'"))
