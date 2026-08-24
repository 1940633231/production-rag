"""环境变量加载工具：手动解析 .env 文件，零依赖替代 python-dotenv。

用法:
    from app.core.env import load_env
    load_env()  # 加载 .env 到 os.environ
"""
import os
from pathlib import Path


def load_env(env_path: str = ".env"):
    """加载 .env 文件到 os.environ。

    - 不覆盖已有环境变量（让真实 shell 环境变量优先）
    - 支持 # 注释、引号包裹的值
    - 跳过 ${VAR} 引用（如 PYTHONPATH=${workspaceFolder}，由 IDE 处理）
    """
    path = Path(env_path)
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip()
        # 跳过 ${VAR} 引用（IDE 变量替换，Python 不处理）
        if "${" in value:
            continue
        # 去掉引号
        if len(value) >= 2 and value[0] in ("'", '"') and value[-1] == value[0]:
            value = value[1:-1]
        # 不覆盖已有环境变量
        if key not in os.environ:
            os.environ[key] = value
