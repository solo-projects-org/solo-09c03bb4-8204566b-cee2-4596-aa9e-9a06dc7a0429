"""流式 gzip 解压与四行制 FASTQ 校验。

展开过程使用带 ``max_length`` 的有界 ``decompress``：每次只在内存中产生一小块
（默认 64 KiB）展开数据，随即向 SpaceManager 预留配额并落盘。因此：

* 即使单个请求块是几十 KiB 的高压缩率 gzip（可展开到几十 MiB），也不会先在内存
  里生成整块结果，内存中最多只多出一块的展开数据；
* 超限在解压途中即被发现（最多多做一块），而不是等整包解完才报；
* 块与块之间 ``await`` 落盘（并让出事件循环），其他批次的查询 / 上传 / 取消
  不会被这次展开拖住。
"""

from __future__ import annotations

import zlib
from pathlib import Path
from typing import BinaryIO, Iterator

from .storage import QuotaExceededError, SpaceManager

# 单次 decompress 最多产生的展开字节数，也是配额核算 / 事件循环让出的粒度。
DECOMPRESS_PIECE_BYTES = 64 * 1024
# 喂给解压器的单块压缩字节上限。zlib 每次返回都会复制 unconsumed_tail，
# 若直接把一个超大请求块整块喂进去会退化成 O(n^2) 复制；这里先自行切片。
DECOMPRESS_INPUT_BYTES = 64 * 1024


class StreamError(Exception):
    """上传流处理过程中的可预期错误（坏 gzip / 客户端中断等）。"""


class _GzipExpander:
    """增量、有界内存的单成员 gzip 解压器。

    ``feed()`` 是同步生成器：每次迭代最多产出 ``piece_bytes`` 字节展开数据，
    调用方可以逐块核算配额、落盘并在块间让出事件循环。解压本身的状态很小，
    不会随展开体积增长。
    """

    def __init__(
        self,
        piece_bytes: int = DECOMPRESS_PIECE_BYTES,
        input_bytes: int = DECOMPRESS_INPUT_BYTES,
    ) -> None:
        if piece_bytes <= 0 or input_bytes <= 0:
            raise ValueError("piece_bytes and input_bytes must be positive")
        self._piece_bytes = piece_bytes
        self._input_bytes = input_bytes
        self._decomp = zlib.decompressobj(16 + zlib.MAX_WBITS)
        self._fed = False

    def feed(self, data: bytes) -> Iterator[bytes]:
        """喂入一块压缩字节，按 ``piece_bytes`` 为上限产出展开数据。

        坏流抛 ``zlib.error`` -> ``StreamError``；gzip 结束标记之后仍有数据
        （多成员归档 / 尾随数据）同样拒绝。
        """
        if self._decomp.eof:
            # 上一个块已经到达结束标记，这里出现的任何字节都是尾随数据
            raise StreamError("multi-member gzip archives are not accepted")
        self._fed = True
        chunk_size = self._input_bytes
        total = len(data)
        for start in range(0, total, chunk_size):
            piece = data[start : start + chunk_size]
            is_last_slice = start + chunk_size >= total
            while True:
                try:
                    out = self._decomp.decompress(piece, self._piece_bytes)
                except zlib.error as exc:
                    raise StreamError(f"corrupt gzip stream: {exc}") from exc
                piece = self._decomp.unconsumed_tail

                if out:
                    yield out

                if self._decomp.eof:
                    # 本块内到达结束标记；本片剩余字节或后续切片都算尾随数据
                    if not is_last_slice or piece or self._decomp.unused_data:
                        raise StreamError("multi-member gzip archives are not accepted")
                    return
                if not piece and len(out) < self._piece_bytes:
                    # 该输入片已全部消费，且没有更多缓冲输出
                    break
                # 否则 unconsumed_tail 里还有被 max_length 截住的输入，继续

    def finish(self) -> Iterator[bytes]:
        """压缩流结束后调用：确认正常结束标记，并以有界方式兜底残余输出。"""
        if not self._fed:
            raise StreamError("empty request body")
        if not self._decomp.eof:
            raise StreamError("truncated gzip stream: missing end-of-stream marker")
        # eof 后输入已全部消费，正常路径下没有缓冲输出；这里用带 max_length 的
        # decompress(b"") 兜底，避免任何可能的大块 flush() 一次性进内存。
        while True:
            tail = self._decomp.decompress(b"", self._piece_bytes)
            if not tail:
                return
            yield tail


async def stream_gunzip(
    *,
    request,
    gz_path: Path,
    out_path: Path,
    space: SpaceManager,
    batch_id: str,
    cancel_event,
    idle_timeout: float = 15.0,
) -> int:
    """读取请求体，一边把原始 gzip 字节写入临时文件，一边有界地流式展开。

    落盘分两类，都在写入前按真实字节数向 SpaceManager 预留：
    压缩临时文件的每个请求块，以及展开结果的每个 ``DECOMPRESS_PIECE_BYTES`` 块。
    返回展开后的总字节数。取消 / 坏包 / 超限 / 静默超时均抛异常，由调用方清理本批。
    """
    import asyncio

    import aiofiles

    expander = _GzipExpander()
    out_written = 0
    got_any = False
    cancelled: list[bool] = [False]

    gz_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    body_stream = request.stream()

    async def _next_chunk():
        # 每个块最多等 idle_timeout 秒，客户端长时间静默即中断本批
        chunk = await asyncio.wait_for(body_stream.__anext__(), timeout=idle_timeout)
        # 取消信号在文件上下文之外处理，保证 aiofiles 句柄能正常关闭
        if cancel_event.is_set():
            cancelled[0] = True
        return chunk

    async def _drain(gen, out_f) -> None:
        """逐块核算配额并落盘展开数据；取消时关闭生成器并停止。"""
        nonlocal out_written
        try:
            for plain in gen:
                if cancel_event.is_set():
                    cancelled[0] = True
                    return
                # 展开数据边产生边核算：先预留这一块再落盘，绝不先攒整块结果
                await space.reserve(batch_id, len(plain))
                try:
                    await out_f.write(plain)
                except BaseException:
                    await space.release(batch_id, len(plain))
                    raise
                out_written += len(plain)
                # 每块之后让出事件循环，其他批次的请求得以推进
                await asyncio.sleep(0)
        finally:
            gen.close()

    try:
        async with aiofiles.open(gz_path, "wb") as gz_f, aiofiles.open(out_path, "wb") as out_f:
            while True:
                try:
                    chunk = await _next_chunk()
                except StopAsyncIteration:
                    break
                except asyncio.TimeoutError as exc:
                    raise StreamError(
                        f"no data received within {idle_timeout}s, upload aborted"
                    ) from exc

                if cancelled[0]:
                    break
                if not chunk:
                    continue
                got_any = True

                # 1) 压缩字节先预留再落盘
                await space.reserve(batch_id, len(chunk))
                try:
                    await gz_f.write(chunk)
                except BaseException:
                    await space.release(batch_id, len(chunk))
                    raise

                # 2) 有界流式解压：一个请求块可能展开成很多块，逐块核算并落盘
                await _drain(expander.feed(chunk), out_f)
                if cancelled[0]:
                    break

            # 3) 必须到达 gzip 流正常结束点；残留输出同样按块核算
            if not cancelled[0]:
                await _drain(expander.finish(), out_f)
    except QuotaExceededError:
        raise
    except StreamError:
        raise
    except Exception as exc:  # 客户端断连等
        raise StreamError(f"upload stream interrupted: {exc}") from exc

    # 文件句柄已正常关闭，再抛出取消信号
    if cancelled[0]:
        raise StreamError("batch cancelled while receiving")

    if not got_any:
        raise StreamError("empty request body")

    return out_written


# --------------------------------------------------------------------------- #
# FASTQ 校验
# --------------------------------------------------------------------------- #


def _read_lines(fh: BinaryIO) -> Iterator[bytes]:
    """逐行生成去掉行尾换行的字节行；文件未以换行结尾视为损坏。"""
    while True:
        raw = fh.readline()
        if not raw:
            return
        if not raw.endswith(b"\n"):
            raise ValueError("file does not end with newline")
        yield raw[:-2] if raw.endswith(b"\r\n") else raw[:-1]


def _next_line(lines: Iterator[bytes], message: str) -> bytes:
    try:
        return next(lines)
    except StopIteration:
        raise ValueError(message) from None


def _iter_records(lines: Iterator[bytes], side: str) -> Iterator[tuple[bytes, bytes, bytes]]:
    """把四行制行流转成 (header, seq, qual) 记录流，并做单端结构校验。"""
    index = 0
    while True:
        try:
            header = next(lines)
        except StopIteration:
            return
        seq = _next_line(lines, f"{side} truncated: missing sequence at record {index + 1}")
        plus = _next_line(lines, f"{side} truncated: missing '+' line at record {index + 1}")
        qual = _next_line(lines, f"{side} truncated: missing quality at record {index + 1}")

        if not header.startswith(b"@"):
            raise ValueError(f"{side} record {index + 1}: header must start with '@'")
        if not seq:
            raise ValueError(f"{side} record {index + 1}: empty sequence")
        if plus != b"+":
            raise ValueError(f"{side} record {index + 1}: third line must be '+'")
        if len(seq) != len(qual):
            raise ValueError(
                f"{side} record {index + 1}: sequence length {len(seq)} "
                f"!= quality length {len(qual)}"
            )
        yield header, seq, qual
        index += 1


def _base_name(header_line: bytes, suffix: bytes) -> bytes:
    """取读段名（首个空白前的 token）并去掉末尾的 /1 或 /2。"""
    token = header_line[1:].split(None, 1)[0]
    if token.endswith(suffix):
        token = token[: -len(suffix)]
    return token


def validate_fastq_structure(path: Path) -> int:
    """校验单个文件为严格四行制 FASTQ，返回读段条数。"""
    count = 0
    with open(path, "rb") as fh:
        for _header, _seq, _qual in _iter_records(_read_lines(fh), path.name):
            count += 1
    if count == 0:
        raise ValueError("FASTQ contains no records")
    return count


def validate_fastq_pair(r1_path: Path, r2_path: Path) -> int:
    """两端交叉校验：条数相同、逐对读段名去 /1 /2 后一致、序列与质量等长。

    返回读段对数量。
    """
    pairs = 0
    with open(r1_path, "rb") as f1, open(r2_path, "rb") as f2:
        rec1 = _iter_records(_read_lines(f1), "R1")
        rec2 = _iter_records(_read_lines(f2), "R2")
        while True:
            r1: tuple[bytes, bytes, bytes] | None = next(rec1, None)
            r2: tuple[bytes, bytes, bytes] | None = next(rec2, None)
            if r1 is None and r2 is None:
                break
            if r1 is None or r2 is None:
                raise ValueError("R1 and R2 have different record counts")
            h1, _s1, _q1 = r1
            h2, _s2, _q2 = r2
            if _base_name(h1, b"/1") != _base_name(h2, b"/2"):
                raise ValueError(
                    f"record {pairs + 1}: read name mismatch after stripping /1 /2: "
                    f"{h1!r} vs {h2!r}"
                )
            pairs += 1

    if pairs == 0:
        raise ValueError("FASTQ pair contains no records")
    return pairs
