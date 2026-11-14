"""高压缩率单块上传的回归测试。

历史缺陷：一次 ``decomp.decompress(chunk)`` 会把整个请求块一次性展开，先占满内存、
再在整块生成之后才报超限，期间还独占总事件循环。请求块为数十 KiB 时，现实 FASTQ
数据即可展开出数 MiB，刻意构造的数据可达数十 MiB。

这里覆盖：
* 单块高压缩率 gzip 在解压途中即触发 413，且内存中只出现小块展开数据；
* 本批被清理、配额回收，其他批仍可查询 / 上传 / 取消；
* 单成员 gzip、FASTQ 结构与双端配对规则不变。
"""

from __future__ import annotations

import asyncio
import gzip
import io
import os
import tracemalloc

import pytest

from app.validation import (
    DECOMPRESS_PIECE_BYTES,
    _GzipExpander,
    stream_gunzip,
)
from app.storage import SpaceManager
from tests.helpers import create_batch, good_pair, gz_bytes, upload_bytes


def _compressible_records(n: int, length: int = 200):
    """序列高度重复 => 极高压缩率，少量压缩字节即可展开成很大体积。"""
    r1, r2 = [], []
    for i in range(n):
        seq = b"ACGT" * (length // 4)
        qual = b"I" * length
        base = f"bomb-{i:07d}".encode()
        r1.append((base + b"/1", seq, qual))
        r2.append((base + b"/2", seq, qual))
    return r1, r2


class _OneShotRequest:
    """整个请求体作为单个 chunk 一次性交付，模拟「一次收到很大的压缩块」。"""

    def __init__(self, blob: bytes) -> None:
        self._blob = blob

    def stream(self):
        async def _gen():
            yield self._blob

        return _gen()


def test_single_huge_chunk_reserves_piecewise_and_rejects_early(tmp_path):
    """一个请求块可展开到数十 MiB：必须逐块核算并在途中立即 413。"""
    # 约 16 MiB 展开体积，压缩后百余 KiB，一次交付
    records = _compressible_records(n=40_000, length=200)[0]
    blob = gz_bytes(records)
    expanded_bytes = sum(len(s) + len(q) + len(name) + 7 for name, s, q in records)
    assert expanded_bytes > 8 * 1024 * 1024
    assert len(blob) < 256 * 1024  # 单块就能装下，压缩率极高
    assert expanded_bytes // len(blob) > 100

    limit = 512 * 1024
    space = SpaceManager(limit)
    out = tmp_path / "R1.fastq"

    reached: list[int] = []

    async def scenario():
        original_reserve = space.reserve

        async def tracking_reserve(batch_id: str, additional: int) -> None:
            # 记录每次预留的块大小，确认按小块而非整块核算
            reached.append(additional)
            await original_reserve(batch_id, additional)

        space.reserve = tracking_reserve  # type: ignore[method-assign]

        with pytest.raises(Exception) as excinfo:
            await stream_gunzip(
                request=_OneShotRequest(blob),
                gz_path=tmp_path / "R1.fastq.gz.uploading",
                out_path=out,
                space=space,
                batch_id="bomb",
                cancel_event=asyncio.Event(),
                idle_timeout=30.0,
            )
        return excinfo.value

    tracemalloc.start()
    try:
        exc = asyncio.run(scenario())
        _current, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()

    # 途中触发配额错误（而不是等整包解完）
    from app.storage import QuotaExceededError

    assert isinstance(exc, QuotaExceededError), exc
    # 已落盘部分必然小于上限（不会先写出整块超额结果）
    assert out.stat().st_size <= limit
    # 逐块预留：第一个是压缩输入块本身；之后展开侧每次预留不超过一块
    assert reached[0] == len(blob)
    expansion_reserves = reached[1:]
    assert expansion_reserves and max(expansion_reserves) <= DECOMPRESS_PIECE_BYTES
    # 分配总量不会先跑到整包展开体积（最多是输入块 + 上限 + 一块）
    assert sum(reached) <= len(blob) + limit + DECOMPRESS_PIECE_BYTES
    # 内存中只多出一块展开数据
    assert peak < 4 * 1024 * 1024, peak


def test_expander_never_materializes_more_than_one_piece():
    """即使一个输入块能展开出数十 MiB，生成器每次也只产出 <= 一块。"""
    payload = b"A" * (32 * 1024 * 1024)  # 32 MiB
    blob = gzip.compress(payload, 9)
    assert len(blob) < 64 * 1024  # 单个请求块大小级别

    expander = _GzipExpander()
    pieces = list(expander.feed(blob)) + list(expander.finish())
    assert b"".join(pieces) == payload
    assert max(len(p) for p in pieces) <= DECOMPRESS_PIECE_BYTES


def test_quota_exceeded_by_single_high_ratio_chunk_cleans_only_that_batch(client, settings):
    """端到端：单块高压缩率上传 413；本批清空，其他批查询 / 上传 / 取消不受影响。"""
    from app.main import service

    records = _compressible_records(n=4_000, length=200)[0]
    blob = gz_bytes(records)
    assert len(blob) < 16 * 1024  # 一次请求块即可容纳

    original_limit = service.space._limit
    service.space._limit = 1024 * 1024
    try:
        bomb = create_batch(client, "ratio-bomb")
        resp = upload_bytes(client, bomb, "R1", blob)
        assert resp.status_code == 413, resp.text
        assert "quota" in resp.json()["detail"]

        # 本批：failed、文件清空、配额回收
        detail = client.get(f"/batches/{bomb}").json()
        assert detail["status"] == "failed"
        assert list((settings.batch_dir / bomb).iterdir()) == []
        assert service.space._allocated.get(bomb) in (None, 0)
        assert client.get(f"/batches/{bomb}/files/R1/download").status_code == 409

        # 其他批仍可正常查询、上传、取消
        other = create_batch(client, "ratio-other")
        assert client.get(f"/batches/{other}").status_code == 200
        r1, r2 = good_pair()
        assert upload_bytes(client, other, "R1", gz_bytes(r1)).status_code == 200
        done = upload_bytes(client, other, "R2", gz_bytes(r2))
        assert done.status_code == 200 and done.json()["status"] == "ready"

        cancel_me = create_batch(client, "ratio-cancel")
        assert client.post(f"/batches/{cancel_me}/cancel").json()["status"] == "cancelled"
    finally:
        service.space._limit = original_limit


def test_trailing_data_inside_one_large_chunk_still_rejected():
    """结束标记落在超大输入块内部时，剩余切片必须被识别为多成员并拒绝。"""
    from app.validation import StreamError

    payload = os.urandom(4 * 1024 * 1024)
    first = gzip.compress(payload, 9)
    second = gzip.compress(b"trailing-member")
    # 输入块远大于 DECOMPRESS_INPUT_BYTES，结束标记后仍有同块数据
    blob = first + second
    assert len(blob) > 2 * 64 * 1024

    expander = _GzipExpander()
    with pytest.raises(StreamError, match="multi-member"):
        for _piece in expander.feed(blob):
            pass


def test_expansion_yields_event_loop_so_other_work_progresses(tmp_path):
    """单个超高压缩块展开期间，事件循环仍被让出：并发任务持续得到调度。"""
    records = _compressible_records(n=40_000, length=200)[0]
    blob = gz_bytes(records)
    limit = 64 * 1024 * 1024  # 足够大，不触发配额，完整展开
    space = SpaceManager(limit)
    ticker_count = [0]
    done = [False]

    async def ticker():
        while not done[0]:
            ticker_count[0] += 1
            await asyncio.sleep(0)

    async def scenario():
        task = asyncio.create_task(ticker())
        total = await stream_gunzip(
            request=_OneShotRequest(blob),
            gz_path=tmp_path / "R1.fastq.gz.uploading",
            out_path=tmp_path / "R1.fastq",
            space=space,
            batch_id="tick",
            cancel_event=asyncio.Event(),
            idle_timeout=30.0,
        )
        done[0] = True
        await task
        return total

    total = asyncio.run(scenario())
    # 展开确实发生：输出远大于单块
    assert total > 100 * DECOMPRESS_PIECE_BYTES
    # 展开期间 ticker 被调度了很多次 => 没有独占事件循环
    assert ticker_count[0] > 100, ticker_count[0]


def test_single_member_and_pairing_rules_still_enforced(client, settings):
    """回归：修复接收链后，单成员 gzip 与双端配对规则保持不变。"""
    r1, r2 = good_pair()

    # 多成员 gzip 仍然拒绝
    buf1, buf2 = io.BytesIO(), io.BytesIO()
    with gzip.GzipFile(fileobj=buf1, mode="wb") as fh:
        for rec in r1[:2]:
            fh.write(b"@" + rec[0] + b"\n" + rec[1] + b"\n+\n" + rec[2] + b"\n")
    with gzip.GzipFile(fileobj=buf2, mode="wb") as fh:
        rec = r1[2]
        fh.write(b"@" + rec[0] + b"\n" + rec[1] + b"\n+\n" + rec[2] + b"\n")
    multi = create_batch(client, "still-multi")
    resp = upload_bytes(client, multi, "R1", buf1.getvalue() + buf2.getvalue())
    assert resp.status_code == 422
    assert "multi-member" in resp.json()["detail"]

    # FASTQ 结构错误仍然拒绝
    broken = create_batch(client, "still-struct")
    name, seq, qual = r1[0]
    bad = [(name, seq, qual[:-1])] + r1[1:]
    resp = upload_bytes(client, broken, "R1", gz_bytes(bad))
    assert resp.status_code == 422
    assert "length" in resp.json()["detail"]

    # 双端配对不一致仍然整批拒收
    pair = create_batch(client, "still-pair")
    assert upload_bytes(client, pair, "R1", gz_bytes(r1)).status_code == 200
    wrong = list(r2)
    wrong[1] = (b"read-WRONG/2", wrong[1][1], wrong[1][2])
    resp = upload_bytes(client, pair, "R2", gz_bytes(wrong))
    assert resp.status_code == 422
    assert "mismatch" in resp.json()["detail"]
    assert client.get(f"/batches/{pair}").json()["status"] == "failed"
    assert list((settings.batch_dir / pair).iterdir()) == []
