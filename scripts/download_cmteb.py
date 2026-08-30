"""下载 C-MTEB 中文检索评测数据集（T2Retrieval + qrels）。

数据来源:
  https://huggingface.co/datasets/C-MTEB/T2Retrieval
  https://huggingface.co/datasets/C-MTEB/T2Retrieval-qrels

默认使用国内镜像:
  https://hf-mirror.com

下载到:
  data/evaluation/cmteb/

用法:
  .venv\\Scripts\\python.exe scripts\\download_cmteb.py

也可以指定镜像:
  $env:HF_ENDPOINT="https://hf-mirror.com"
  .venv\\Scripts\\python.exe scripts\\download_cmteb.py
"""

import argparse
import os
import sys
import time
from pathlib import Path
from urllib.parse import quote

import requests


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

OUTPUT_ROOT = PROJECT_ROOT / "data" / "evaluation" / "cmteb"

# (repo, repo 内文件路径)
FILES = [
    (
        "C-MTEB/T2Retrieval",
        "data/corpus-00000-of-00001-8afe7b7a7eca49e3.parquet",
    ),
    (
        "C-MTEB/T2Retrieval",
        "data/queries-00000-of-00001-930bf3b805a80dd9.parquet",
    ),
    (
        "C-MTEB/T2Retrieval-qrels",
        "data/dev-00000-of-00001-92ed0416056ff7e1.parquet",
    ),
]

CHUNK_SIZE = 1024 * 1024  # 1 MB
TIMEOUT = (15, 60)


def get_endpoint() -> str:
    """获取 Hugging Face endpoint。默认使用 hf-mirror.com。"""
    endpoint = os.environ.get("HF_ENDPOINT", "https://hf-mirror.com")
    return endpoint.rstrip("/")


def build_url(endpoint: str, repo: str, filename: str) -> str:
    """构造 Hugging Face resolve URL。"""
    return (
        f"{endpoint}/datasets/"
        f"{quote(repo, safe='/')}/resolve/main/"
        f"{quote(filename, safe='/')}"
    )


def format_size(size: int) -> str:
    """格式化文件大小。"""
    units = ["B", "KB", "MB", "GB", "TB"]

    value = float(size)
    for unit in units:
        if value < 1024 or unit == units[-1]:
            return f"{value:.2f} {unit}"
        value /= 1024

    return f"{value:.2f} TB"


def format_speed(bytes_per_sec: float) -> str:
    return f"{format_size(int(bytes_per_sec))}/s"


def download_file(url: str, destination: Path) -> None:
    """下载单个文件，支持断点续传。"""

    destination.parent.mkdir(parents=True, exist_ok=True)

    # 下载完成后才会正式生成 destination。
    # 中途中断则保留 .part。
    part_file = destination.with_suffix(destination.suffix + ".part")

    if destination.exists():
        size = destination.stat().st_size

        if size > 0:
            print(f"已存在，跳过: {destination}")
            print(f"文件大小: {format_size(size)}")
            return

    existing_size = part_file.stat().st_size if part_file.exists() else 0

    headers = {}

    if existing_size > 0:
        headers["Range"] = f"bytes={existing_size}-"
        print(
            f"发现未完成下载: {part_file.name} "
            f"({format_size(existing_size)})"
        )
        print("尝试断点续传...")

    try:
        response = requests.get(
            url,
            headers=headers,
            stream=True,
            timeout=TIMEOUT,
            allow_redirects=True,
        )
    except requests.RequestException as exc:
        raise RuntimeError(f"连接失败: {exc}") from exc

    # 如果服务器支持 Range，应该返回 206。
    # 如果返回 200，说明服务器没有按 Range 续传。
    if existing_size > 0 and response.status_code == 200:
        response.close()

        print("服务器不支持断点续传，将重新下载。")

        try:
            response = requests.get(
                url,
                stream=True,
                timeout=TIMEOUT,
                allow_redirects=True,
            )
        except requests.RequestException as exc:
            raise RuntimeError(f"连接失败: {exc}") from exc

        existing_size = 0

    if response.status_code not in (200, 206):
        response.close()
        raise RuntimeError(
            f"下载失败: HTTP {response.status_code}\n"
            f"URL: {url}"
        )

    content_length = response.headers.get("Content-Length")

    if content_length:
        content_length = int(content_length)

        if response.status_code == 206:
            total_size = existing_size + content_length
        else:
            total_size = content_length
    else:
        total_size = None

    mode = "ab" if existing_size > 0 and response.status_code == 206 else "wb"

    downloaded = existing_size
    start_time = time.time()
    last_print_time = start_time

    print(f"开始下载: {destination.name}")

    try:
        with response:
            with open(part_file, mode) as file:
                for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                    if not chunk:
                        continue

                    file.write(chunk)
                    downloaded += len(chunk)

                    now = time.time()

                    # 不要每个 chunk 都刷新，避免终端刷屏。
                    if now - last_print_time >= 1:
                        elapsed = max(now - start_time, 0.001)
                        speed = (downloaded - existing_size) / elapsed

                        if total_size:
                            percent = downloaded / total_size * 100

                            remaining = (
                                (total_size - downloaded) / speed
                                if speed > 0
                                else 0
                            )

                            print(
                                f"\r进度: {format_size(downloaded)} / "
                                f"{format_size(total_size)} "
                                f"({percent:.1f}%) | "
                                f"速度: {format_speed(speed)} | "
                                f"剩余: {remaining / 60:.1f} min",
                                end="",
                                flush=True,
                            )
                        else:
                            print(
                                f"\r已下载: {format_size(downloaded)} | "
                                f"速度: {format_speed(speed)}",
                                end="",
                                flush=True,
                            )

                        last_print_time = now

    except (OSError, requests.RequestException) as exc:
        print()
        print(f"\n下载中断，已保留临时文件:")
        print(f"  {part_file}")
        raise RuntimeError(f"下载失败: {exc}") from exc

    print()

    # 下载完成后再检查大小。
    if total_size is not None and downloaded != total_size:
        raise RuntimeError(
            f"文件大小异常: "
            f"实际 {downloaded} bytes，"
            f"预期 {total_size} bytes"
        )

    # 下载完成后再重命名。
    part_file.replace(destination)

    print(f"完成: {destination}")
    print(f"大小: {format_size(destination.stat().st_size)}")


def download(output_root: Path) -> None:
    endpoint = get_endpoint()

    print(f"HF Endpoint: {endpoint}")
    print(f"输出目录: {output_root}")
    print()

    for repo, filename in FILES:
        print("=" * 80)
        print(f"数据集: {repo}")
        print(f"文件:   {filename}")

        url = build_url(endpoint, repo, filename)

        # 保持 repo 目录结构：
        # data/evaluation/cmteb/C-MTEB/T2Retrieval/...
        destination = output_root / repo / filename

        print(f"URL:    {url}")
        print(f"保存:   {destination}")
        print()

        download_file(url, destination)

    print()
    print("=" * 80)
    print("全部下载完成。")
    print(f"输出目录: {output_root}")


def main():
    parser = argparse.ArgumentParser(
        description="下载 C-MTEB 中文检索评测数据"
    )

    parser.add_argument(
        "--output",
        default=str(OUTPUT_ROOT),
        help=f"下载根目录（默认 {OUTPUT_ROOT}）",
    )

    args = parser.parse_args()

    try:
        download(Path(args.output))
    except KeyboardInterrupt:
        print("\n用户中断下载。")
        sys.exit(130)
    except Exception as exc:
        print(f"\n错误: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
