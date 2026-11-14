#!/usr/bin/env python3
"""生成演示样例：正常、错配、超限、高压缩率、损坏。

用法:
    python scripts/make_samples.py [输出目录]

输出目录下:
    normal/R1.fastq.gz  normal/R2.fastq.gz       # 合法双端
    mismatch/R1.fastq.gz mismatch/R2.fastq.gz    # R2 第 2 条读段名与 R1 不一致
    oversized/R1.fastq.gz oversized/R2.fastq.gz  # 展开后远超 64 MiB 上限
    high_ratio/R1.fastq.gz high_ratio/R2.fastq.gz
        # 高压缩率：约 150 KiB 的压缩文件、单个约 32 KiB 请求块即可展开出约 7 MiB，
        # 用于演示「边解压边核算、超限即拒」而不会先把整块结果解进内存
    corrupt/R1.fastq.gz                          # 损坏的 gzip（截断）
"""

from __future__ import annotations

import gzip
import os
import sys
from pathlib import Path


def _record(name: bytes, seq: bytes, qual: bytes) -> bytes:
    return b"@" + name + b"\n" + seq + b"\n+\n" + qual + b"\n"


def _pair(index: int, length: int = 80) -> tuple[bytes, bytes]:
    # 确定性生成，避免引入随机依赖
    seq1 = bytes(65 + (i * 7 + index) % 26 for i in range(length))
    seq2 = bytes(65 + (i * 11 + index) % 26 for i in range(length))
    qual = bytes(33 + (i * 3 + index) % 60 for i in range(length))
    base = f"sample-seq-{index:06d}".encode()
    return _record(base + b"/1", seq1, qual), _record(base + b"/2", seq2, qual)


def _write_pair(directory: Path, count: int, *, mismatch_at: int | None = None) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    with gzip.open(directory / "R1.fastq.gz", "wb") as f1, gzip.open(
        directory / "R2.fastq.gz", "wb"
    ) as f2:
        for i in range(count):
            r1, r2 = _pair(i)
            if mismatch_at is not None and i == mismatch_at:
                # 读段名被调包，去掉 /2 后仍与 R1 对不上
                r2 = r2.replace(b"sample-seq-", b"sample-WRONG-", 1)
            f1.write(r1)
            f2.write(r2)


def _high_ratio_pair(count: int, length: int = 200) -> tuple[bytes, bytes]:
    """极高压缩率的一对文件：重复序列 + 恒定质量值。

    40_000 条、每条 200 bp 时展开约 17 MiB，而 gzip 压缩后只有约 150 KiB，
    即单个 32 KiB 请求块内就可能压着数 MiB 待展开数据。
    """
    seq = b"ACGT" * (length // 4)
    qual = b"I" * length
    r1 = bytearray()
    r2 = bytearray()
    for i in range(count):
        base = f"high-ratio-{i:07d}".encode()
        r1 += _record(base + b"/1", seq, qual)
        r2 += _record(base + b"/2", seq, qual)
    return bytes(r1), bytes(r2)


def main(out: Path) -> None:
    # 1) 正常样例：1000 对，约 330 KB 展开数据
    _write_pair(out / "normal", 1000)

    # 2) 错配样例：第 2 条（index=1）读段名不一致
    _write_pair(out / "mismatch", 10, mismatch_at=1)

    # 3) 超限样例：每条记录 4*82≈328 字节，40 万对 ≈ 128 MiB，超过 64 MiB
    _write_pair(out / "oversized", 400_000)

    # 4) 高压缩率样例：32 KiB 级请求块可展开出约 32 MiB
    hr_dir = out / "high_ratio"
    hr_dir.mkdir(parents=True, exist_ok=True)
    hr1, hr2 = _high_ratio_pair(40_000)
    for name, data in (("R1.fastq.gz", hr1), ("R2.fastq.gz", hr2)):
        raw = gzip.compress(data, 9)
        (hr_dir / name).write_bytes(raw)
        assert len(raw) < 256 * 1024

    # 4) 损坏 gzip：合法流被截断
    corrupt_dir = out / "corrupt"
    corrupt_dir.mkdir(parents=True, exist_ok=True)
    tmp_gz = out / "_tmp_corrupt.gz"
    with gzip.open(tmp_gz, "wb") as f:
        for i in range(5):
            r1, _ = _pair(i)
            f.write(r1)
    blob = tmp_gz.read_bytes()
    tmp_gz.unlink()
    (corrupt_dir / "R1.fastq.gz").write_bytes(blob[: len(blob) // 2])

    for path in sorted(out.rglob("*.gz")):
        print(f"{path.relative_to(out)}  {os.path.getsize(path)} bytes")


if __name__ == "__main__":
    target = Path(sys.argv[1] if len(sys.argv) > 1 else "samples")
    main(target)
