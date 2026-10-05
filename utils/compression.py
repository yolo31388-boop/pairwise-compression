"""
游戏内压缩算法与数据压缩系统 - 核心模块

功能：
- LZ77 压缩：滑动窗口 + 最长匹配
- Huffman 压缩：频率统计 + 编码树（前缀编码）
- 解压：校验和验证 + 进度回调
- 字典压缩：从样本训练常见模式，支持增量更新
- 压缩级别：参数化（窗口/匹配长度）+ fast/default/max 预设
"""
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple
import heapq
import math
import time
import zlib


@dataclass
class Config:
    pass


# LZ77 token: ("lit", byte) 或 ("match", distance, length)
Token = Tuple


class Compression:
    """游戏资源压缩系统。"""

    #: 压缩级别预设：在速度与压缩率之间权衡
    LEVEL_PRESETS: Dict[str, Dict] = {
        "fast": {"level": 1, "window_size": 1024, "min_match": 4, "max_match": 32, "use_huffman": False},
        "default": {"level": 5, "window_size": 4096, "min_match": 3, "max_match": 128, "use_huffman": True},
        "max": {"level": 9, "window_size": 32768, "min_match": 3, "max_match": 258, "use_huffman": True},
    }

    MAGIC = b"GCMP"  # 压缩包魔数，用于解压验证

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self._state = {}
        self._history = []
        self._dictionary: Dict[bytes, int] = {}
        self._dict_counter = 0

    def update(self, dt: float):
        pass

    def reset(self):
        self._state = {}
        self._history = []

    # ------------------------------------------------------------------
    # 压缩级别
    # ------------------------------------------------------------------
    def compression_level(self, level="default") -> Dict:
        """返回压缩级别对应的参数。

        level 可以是预设名（"fast"/"default"/"max"）或 1-9 的整数，
        整数会映射到最近的预设并线性插值窗口与匹配长度。
        """
        if isinstance(level, str):
            if level not in self.LEVEL_PRESETS:
                raise ValueError(f"未知的压缩级别预设: {level!r}")
            return dict(self.LEVEL_PRESETS[level])
        if isinstance(level, int) and 1 <= level <= 9:
            if level <= 3:
                base, hi = self.LEVEL_PRESETS["fast"], self.LEVEL_PRESETS["default"]
                t = (level - 1) / 4.0
            else:
                base, hi = self.LEVEL_PRESETS["default"], self.LEVEL_PRESETS["max"]
                t = (level - 5) / 4.0
            t = max(0.0, min(1.0, t))
            return {
                "level": level,
                "window_size": int(base["window_size"] + (hi["window_size"] - base["window_size"]) * t),
                "min_match": base["min_match"],
                "max_match": int(base["max_match"] + (hi["max_match"] - base["max_match"]) * t),
                "use_huffman": level >= 4,
            }
        raise ValueError(f"无效的压缩级别: {level!r}")

    # ------------------------------------------------------------------
    # LZ77：滑动窗口 + 最长匹配
    # ------------------------------------------------------------------
    def lz77_compress(self, data: bytes, window_size: int = 4096,
                      min_match: int = 3, max_match: int = 258) -> List[Token]:
        """LZ77 压缩，返回 token 列表。

        使用滑动窗口在历史数据中查找最长匹配；
        找不到长度 >= min_match 的匹配时输出字面量。
        """
        tokens: List[Token] = []
        n = len(data)
        pos = 0
        while pos < n:
            best_len = 0
            best_dist = 0
            start = max(0, pos - window_size)
            # 在滑动窗口内搜索最长匹配
            for cand in range(start, pos):
                length = 0
                while (length < max_match and pos + length < n
                       and data[cand + length] == data[pos + length]):
                    length += 1
                if length > best_len:
                    best_len = length
                    best_dist = pos - cand
                    if best_len >= max_match:
                        break
            if best_len >= min_match:
                tokens.append(("match", best_dist, best_len))
                pos += best_len
            else:
                tokens.append(("lit", data[pos]))
                pos += 1
        return tokens

    def lz77_decompress(self, tokens: List[Token]) -> bytes:
        """LZ77 解压。"""
        out = bytearray()
        for token in tokens:
            if token[0] == "lit":
                out.append(token[1])
            elif token[0] == "match":
                _, dist, length = token
                if dist <= 0 or dist > len(out):
                    raise ValueError(f"非法的匹配距离: {dist}")
                for _ in range(length):
                    out.append(out[-dist])
            else:
                raise ValueError(f"未知的 token 类型: {token[0]!r}")
        return bytes(out)

    # ------------------------------------------------------------------
    # Huffman：频率统计 + 编码树
    # ------------------------------------------------------------------
    def huffman_frequencies(self, data: bytes) -> Dict[int, int]:
        """统计每个字节出现的频率。"""
        freq: Dict[int, int] = {}
        for b in data:
            freq[b] = freq.get(b, 0) + 1
        return freq

    def huffman_build_tree(self, freq: Dict[int, int]):
        """根据频率构建 Huffman 编码树。

        返回嵌套元组：叶子为字节值，内部节点为 (left, right)。
        """
        if not freq:
            return None
        heap = []
        # (权重, 次序, 节点)；次序保证权重相同时可比较
        for i, (sym, w) in enumerate(sorted(freq.items())):
            heap.append((w, i, sym))
        heapq.heapify(heap)
        counter = len(heap)
        while len(heap) > 1:
            w1, _, n1 = heapq.heappop(heap)
            w2, _, n2 = heapq.heappop(heap)
            heapq.heappush(heap, (w1 + w2, counter, (n1, n2)))
            counter += 1
        return heap[0][2]

    def huffman_codes(self, tree) -> Dict[int, str]:
        """从编码树生成前缀编码表 {字节: 比特串}。"""
        codes: Dict[int, str] = {}
        if tree is None:
            return codes
        if isinstance(tree, int):  # 只有一个符号
            codes[tree] = "0"
            return codes

        def walk(node, prefix):
            if isinstance(node, int):
                codes[node] = prefix
            else:
                walk(node[0], prefix + "0")
                walk(node[1], prefix + "1")

        walk(tree, "")
        return codes

    def huffman_compress(self, data: bytes) -> Dict:
        """Huffman 压缩。

        返回 {"freq", "codes", "bits", "data", "padding"}，
        其中 data 为按位打包后的字节串。
        """
        freq = self.huffman_frequencies(data)
        tree = self.huffman_build_tree(freq)
        codes = self.huffman_codes(tree)
        bits = "".join(codes[b] for b in data)
        padding = (8 - len(bits) % 8) % 8
        bits += "0" * padding
        packed = bytearray()
        for i in range(0, len(bits), 8):
            packed.append(int(bits[i:i + 8], 2))
        return {
            "freq": freq,
            "codes": codes,
            "bits_length": len(bits) - padding,
            "data": bytes(packed),
            "padding": padding,
        }

    def huffman_decompress(self, payload: Dict) -> bytes:
        """Huffman 解压，用编码表逐位解码。"""
        codes = payload["codes"]
        padding = payload["padding"]
        bits_length = payload["bits_length"]
        raw = payload["data"]
        if not codes:
            return b""
        reverse = {code: sym for sym, code in codes.items()}
        bits = "".join(f"{byte:08b}" for byte in raw)
        if padding:
            bits = bits[:-padding]
        bits = bits[:bits_length]
        out = bytearray()
        current = ""
        for bit in bits:
            current += bit
            if current in reverse:
                out.append(reverse[current])
                current = ""
        if current:
            raise ValueError("Huffman 数据损坏：存在无法解码的比特")
        return bytes(out)

    # ------------------------------------------------------------------
    # 字典：训练 + 更新
    # ------------------------------------------------------------------
    def train_dictionary(self, samples: List[bytes], min_pattern: int = 4,
                         max_pattern: int = 16, min_count: int = 2) -> Dict[bytes, int]:
        """从样本数据中学习常见模式，构建字典。

        统计所有长度在 [min_pattern, max_pattern] 的子串出现次数，
        出现 >= min_count 次的模式进入字典。
        """
        counts: Dict[bytes, int] = {}
        for sample in samples:
            n = len(sample)
            for size in range(min_pattern, max_pattern + 1):
                for i in range(0, n - size + 1):
                    pat = sample[i:i + size]
                    counts[pat] = counts.get(pat, 0) + 1
        # 长模式优先，避免短模式抢占长模式的压缩机会
        for pat in sorted(counts, key=lambda p: (-len(p), p)):
            if counts[pat] >= min_count and pat not in self._dictionary:
                self._dictionary[pat] = self._dict_counter
                self._dict_counter += 1
        return dict(self._dictionary)

    def update_dictionary(self, data: bytes, min_pattern: int = 4,
                          max_pattern: int = 16, min_count: int = 2) -> Dict[bytes, int]:
        """用新数据增量更新字典，把新出现的常见模式加入字典。"""
        before = len(self._dictionary)
        self.train_dictionary([data], min_pattern, max_pattern, min_count)
        added = len(self._dictionary) - before
        self._history.append(("dict_update", added))
        return dict(self._dictionary)

    def dictionary_compress(self, data: bytes) -> List[Tuple]:
        """用字典压缩：命中的模式输出 ("ref", id)，否则输出 ("lit", byte)。"""
        if not self._dictionary:
            raise ValueError("字典为空，请先训练字典")
        # 按长度降序，贪心匹配最长模式
        patterns = sorted(self._dictionary.items(), key=lambda kv: -len(kv[0]))
        tokens: List[Tuple] = []
        pos = 0
        n = len(data)
        while pos < n:
            matched = False
            for pat, pid in patterns:
                if data.startswith(pat, pos):
                    tokens.append(("ref", pid))
                    pos += len(pat)
                    matched = True
                    break
            if not matched:
                tokens.append(("lit", data[pos]))
                pos += 1
        return tokens

    def dictionary_decompress(self, tokens: List[Tuple]) -> bytes:
        """字典解压。"""
        id_to_pat = {pid: pat for pat, pid in self._dictionary.items()}
        out = bytearray()
        for token in tokens:
            if token[0] == "lit":
                out.append(token[1])
            elif token[0] == "ref":
                if token[1] not in id_to_pat:
                    raise ValueError(f"未知的字典引用: {token[1]}")
                out.extend(id_to_pat[token[1]])
            else:
                raise ValueError(f"未知的 token 类型: {token[0]!r}")
        return bytes(out)

    # ------------------------------------------------------------------
    # 整体压缩 / 解压（带验证与进度）
    # ------------------------------------------------------------------
    def compress(self, data: bytes, level="default") -> Dict:
        """完整压缩流程：LZ77 -> 序列化 -> （可选）Huffman，附校验和。"""
        params = self.compression_level(level)
        tokens = self.lz77_compress(
            data,
            window_size=params["window_size"],
            min_match=params["min_match"],
            max_match=params["max_match"],
        )
        # 序列化 token 流：lit -> 0x00 <byte>；match -> 0x01 <dist:2B> <len:2B>
        stream = bytearray()
        for token in tokens:
            if token[0] == "lit":
                stream.append(0)
                stream.append(token[1])
            else:
                _, dist, length = token
                stream.append(1)
                stream.extend(dist.to_bytes(2, "big"))
                stream.extend(length.to_bytes(2, "big"))
        body = bytes(stream)
        method = "lz77"
        if params["use_huffman"]:
            huff = self.huffman_compress(body)
            if len(huff["data"]) < len(body):
                body = huff["data"]
                method = "lz77+huffman"
                huffman_meta = {
                    "codes": huff["codes"],
                    "padding": huff["padding"],
                    "bits_length": huff["bits_length"],
                }
            else:
                huffman_meta = None
        else:
            huffman_meta = None
        return {
            "magic": self.MAGIC,
            "method": method,
            "params": params,
            "body": body,
            "huffman": huffman_meta,
            "size": len(data),
            "checksum": zlib.crc32(data),
        }

    def decompress(self, payload: Dict,
                   progress_callback: Optional[Callable[[float], None]] = None) -> bytes:
        """解压：先验证魔数与校验和，再分阶段解压并汇报进度（0.0 ~ 1.0）。"""
        if not isinstance(payload, dict) or payload.get("magic") != self.MAGIC:
            raise ValueError("数据验证失败：不是有效的压缩包")

        def report(p: float):
            if progress_callback:
                progress_callback(p)

        report(0.0)
        body = payload["body"]
        # 阶段一：Huffman 解码
        if payload["method"] == "lz77+huffman":
            meta = payload["huffman"]
            body = self.huffman_decompress({
                "codes": meta["codes"],
                "padding": meta["padding"],
                "bits_length": meta["bits_length"],
                "data": body,
            })
        report(0.5)
        # 阶段二：反序列化 token 流
        tokens: List[Token] = []
        i = 0
        n = len(body)
        while i < n:
            tag = body[i]
            if tag == 0:
                if i + 1 >= n:
                    raise ValueError("数据验证失败：字面量被截断")
                tokens.append(("lit", body[i + 1]))
                i += 2
            elif tag == 1:
                if i + 5 > n:
                    raise ValueError("数据验证失败：匹配 token 被截断")
                dist = int.from_bytes(body[i + 1:i + 3], "big")
                length = int.from_bytes(body[i + 3:i + 5], "big")
                tokens.append(("match", dist, length))
                i += 5
            else:
                raise ValueError(f"数据验证失败：未知标记 {tag}")
        report(0.7)
        # 阶段三：LZ77 重建
        result = self.lz77_decompress(tokens)
        report(0.9)
        # 阶段四：完整性验证
        if len(result) != payload["size"]:
            raise ValueError("数据验证失败：解压长度与原始大小不一致")
        if zlib.crc32(result) != payload["checksum"]:
            raise ValueError("数据验证失败：校验和不匹配，数据已损坏")
        report(1.0)
        return result
