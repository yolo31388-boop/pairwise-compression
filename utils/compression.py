"""
游戏内压缩算法与数据压缩系统 - 核心模块

功能：
- LZ77 压缩：滑动窗口 + 最长匹配
- Huffman 压缩：频率统计 + 编码树 + 前缀编码
- 解压：魔数/长度/校验和验证 + 进度回调
- 字典：从样本训练常见模式 + 增量更新
- 压缩级别：窗口/匹配参数 + fast/default/max 预设
"""
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple, Union
import heapq
import struct
import zlib


@dataclass
class Config:
    pass


# LZ77 令牌：(offset, length, literal)，literal 为 NO_LITERAL 表示流末尾无后续字面量
Token = Tuple[int, int, int]


class Compression:
    MAGIC = b"GCMP"
    VERSION = 1
    FLAG_DICTIONARY = 0x01
    NO_LITERAL = 0xFFFF
    MAX_DICTIONARY_SIZE = 32768
    _HEADER = struct.Struct(">4sBBBBIII")
    _TOKEN = struct.Struct(">IHH")

    # 压缩级别预设：在速度与压缩率之间权衡
    LEVEL_PRESETS: Dict[str, Dict] = {
        "fast": {"level": 1, "window_size": 4096, "min_match": 3,
                 "max_match": 64, "max_chain": 8, "use_dictionary": False},
        "default": {"level": 6, "window_size": 32768, "min_match": 3,
                    "max_match": 258, "max_chain": 64, "use_dictionary": True},
        "max": {"level": 9, "window_size": 65536, "min_match": 3,
                "max_match": 512, "max_chain": 256, "use_dictionary": True},
    }

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self._state = {}
        self._history = []
        self.dictionary: bytes = b""
        self.progress: float = 0.0
        self.compression_level(self.config.get("level", "default"))

    def update(self, dt: float):
        pass

    def reset(self):
        self._state = {}
        self._history = []
        self.progress = 0.0

    # ------------------------------------------------------------------
    # 压缩级别
    # ------------------------------------------------------------------
    def compression_level(self, level: Union[str, int, Dict] = "default") -> Dict:
        """设置压缩级别。支持预设名、1-9 整数或自定义参数字典，返回生效参数。"""
        if isinstance(level, str):
            if level not in self.LEVEL_PRESETS:
                raise ValueError(f"未知的压缩级别预设: {level}")
            params = dict(self.LEVEL_PRESETS[level])
        elif isinstance(level, bool):
            raise TypeError("压缩级别不支持布尔值")
        elif isinstance(level, int):
            if not 1 <= level <= 9:
                raise ValueError("压缩级别必须在 1-9 之间")
            params = {
                "level": level,
                "window_size": min(1 << (10 + level), 65536),
                "min_match": 3,
                "max_match": 32 * (level + 1),
                "max_chain": max(4, 1 << (level // 2 + 2)),
                "use_dictionary": level >= 4,
            }
        elif isinstance(level, dict):
            params = {**self.get_level_params(), **level}
        else:
            raise TypeError(f"不支持的压缩级别类型: {type(level).__name__}")
        self.level = params.get("level", 0)
        self.window_size = int(params["window_size"])
        self.min_match = int(params["min_match"])
        self.max_match = int(params["max_match"])
        self.max_chain = int(params["max_chain"])
        self.use_dictionary = bool(params.get("use_dictionary", True))
        return self.get_level_params()

    def get_level_params(self) -> Dict:
        return {
            "level": self.level,
            "window_size": self.window_size,
            "min_match": self.min_match,
            "max_match": self.max_match,
            "max_chain": self.max_chain,
            "use_dictionary": self.use_dictionary,
        }

    # ------------------------------------------------------------------
    # LZ77：滑动窗口 + 最长匹配
    # ------------------------------------------------------------------
    def lz77_compress(self, data: bytes, window_size: Optional[int] = None,
                      min_match: Optional[int] = None, max_match: Optional[int] = None,
                      max_chain: Optional[int] = None,
                      dictionary: Optional[bytes] = None) -> List[Token]:
        data = bytes(data)
        window_size = window_size or self.window_size
        min_match = min_match or self.min_match
        max_match = max_match or self.max_match
        max_chain = max_chain or self.max_chain
        if dictionary is None:
            dictionary = self.dictionary if self.use_dictionary else b""
        dictionary = bytes(dictionary)

        buf = dictionary + data
        start = len(dictionary)
        total = len(buf)
        tokens: List[Token] = []
        # 3 字节哈希 -> 出现位置列表，用于在滑动窗口内快速找候选匹配
        table: Dict[bytes, List[int]] = {}
        for pos in range(start):
            if pos + 3 <= total:
                table.setdefault(buf[pos:pos + 3], []).append(pos)

        pos = start
        while pos < total:
            best_offset = 0
            best_length = 0
            if pos + min_match <= total:
                candidates = table.get(buf[pos:pos + 3], [])
                checked = 0
                # 从最近的位置往回找，保留最长匹配
                for cand in reversed(candidates):
                    if pos - cand > window_size:
                        break
                    max_len = min(max_match, total - pos)
                    length = 0
                    while length < max_len and buf[cand + length] == buf[pos + length]:
                        length += 1
                    if length > best_length:
                        best_length = length
                        best_offset = pos - cand
                        if best_length >= max_len:
                            break
                    checked += 1
                    if checked >= max_chain:
                        break
            if best_length >= min_match:
                nxt = pos + best_length
                literal = buf[nxt] if nxt < total else self.NO_LITERAL
                tokens.append((best_offset, best_length, literal))
                end = nxt + (1 if literal != self.NO_LITERAL else 0)
                for p in range(pos, min(end, total - 2)):
                    table.setdefault(buf[p:p + 3], []).append(p)
                pos = end
            else:
                tokens.append((0, 0, buf[pos]))
                if pos + 3 <= total:
                    table.setdefault(buf[pos:pos + 3], []).append(pos)
                pos += 1
        return tokens

    def lz77_decompress(self, tokens: List[Token], dictionary: Optional[bytes] = None) -> bytes:
        if dictionary is None:
            dictionary = self.dictionary if self.use_dictionary else b""
        out = bytearray(dictionary)
        for offset, length, literal in tokens:
            if offset > 0 and length > 0:
                src = len(out) - offset
                if src < 0:
                    raise ValueError("数据损坏：LZ77 偏移越界")
                for k in range(length):
                    out.append(out[src + k])
            if literal != self.NO_LITERAL:
                out.append(literal)
        return bytes(out[len(dictionary):])

    # ------------------------------------------------------------------
    # Huffman：频率统计 + 编码树 + 前缀编码
    # ------------------------------------------------------------------
    def build_frequency(self, data: bytes) -> Dict[int, int]:
        freqs: Dict[int, int] = {}
        for byte in bytes(data):
            freqs[byte] = freqs.get(byte, 0) + 1
        return freqs

    def _build_tree(self, freqs: Dict[int, int]):
        heap = []
        counter = 0
        for sym in sorted(freqs):
            heap.append((freqs[sym], counter, ("leaf", sym)))
            counter += 1
        heapq.heapify(heap)
        if not heap:
            return None
        while len(heap) > 1:
            f1, _, n1 = heapq.heappop(heap)
            f2, _, n2 = heapq.heappop(heap)
            heapq.heappush(heap, (f1 + f2, counter, ("node", n1, n2)))
            counter += 1
        return heap[0][2]

    def _generate_codes(self, tree) -> Dict[int, str]:
        if tree is None:
            return {}
        if tree[0] == "leaf":
            return {tree[1]: "0"}
        codes: Dict[int, str] = {}
        stack = [(tree, "")]
        while stack:
            node, prefix = stack.pop()
            if node[0] == "leaf":
                codes[node[1]] = prefix
            else:
                stack.append((node[1], prefix + "0"))
                stack.append((node[2], prefix + "1"))
        return codes

    def huffman_compress(self, data: bytes) -> Dict:
        data = bytes(data)
        freqs = self.build_frequency(data)
        codes = self._generate_codes(self._build_tree(freqs))
        bits = "".join(codes[b] for b in data)
        padding = (8 - len(bits) % 8) % 8
        if padding:
            bits += "0" * padding
        payload = bytearray()
        for i in range(0, len(bits), 8):
            payload.append(int(bits[i:i + 8], 2))
        return {"freqs": freqs, "padding": padding,
                "length": len(data), "data": bytes(payload)}

    def huffman_decompress(self, payload: Dict) -> bytes:
        freqs = payload["freqs"]
        length = payload["length"]
        tree = self._build_tree(freqs)
        if tree is None:
            if length != 0:
                raise ValueError("数据损坏：Huffman 长度不一致")
            return b""
        if tree[0] == "leaf":
            return bytes([tree[1]]) * length
        bits = "".join(format(b, "08b") for b in payload["data"])
        if payload["padding"]:
            bits = bits[:-payload["padding"]]
        out = bytearray()
        node = tree
        for bit in bits:
            node = node[1] if bit == "0" else node[2]
            if node[0] == "leaf":
                out.append(node[1])
                node = tree
                if len(out) == length:
                    break
        if len(out) != length:
            raise ValueError("数据损坏：Huffman 解码长度不一致")
        return bytes(out)

    # ------------------------------------------------------------------
    # 字典：训练 + 更新
    # ------------------------------------------------------------------
    def train_dictionary(self, samples: List[bytes], pattern_length: int = 8,
                         max_size: Optional[int] = None) -> int:
        """从样本中学习常见模式，构建字典，返回字典大小。"""
        max_size = max_size or self.MAX_DICTIONARY_SIZE
        counts: Dict[bytes, int] = {}
        for sample in samples:
            sample = bytes(sample)
            for i in range(0, max(0, len(sample) - pattern_length + 1)):
                pat = sample[i:i + pattern_length]
                counts[pat] = counts.get(pat, 0) + 1
        common = sorted((p for p, c in counts.items() if c >= 2),
                        key=lambda p: (-counts[p], p))
        dictionary = bytearray()
        for pat in common:
            if len(dictionary) + pattern_length > max_size:
                break
            if pat in dictionary:
                continue
            dictionary += pat
        self.dictionary = bytes(dictionary)
        return len(self.dictionary)

    def update_dictionary(self, data: bytes, pattern_length: int = 8,
                          top_k: int = 16) -> int:
        """把新数据中的高频模式加入字典，返回新增字节数。"""
        data = bytes(data)
        counts: Dict[bytes, int] = {}
        for i in range(0, max(0, len(data) - pattern_length + 1)):
            pat = data[i:i + pattern_length]
            counts[pat] = counts.get(pat, 0) + 1
        common = sorted((p for p, c in counts.items() if c >= 2),
                        key=lambda p: (-counts[p], p))[:top_k]
        dictionary = bytearray(self.dictionary)
        added = 0
        for pat in common:
            if len(dictionary) + pattern_length > self.MAX_DICTIONARY_SIZE:
                break
            if pat in dictionary:
                continue
            dictionary += pat
            added += pattern_length
        self.dictionary = bytes(dictionary)
        return added

    # ------------------------------------------------------------------
    # 完整压缩 / 解压管线
    # ------------------------------------------------------------------
    def compress(self, data: bytes, level: Union[str, int, Dict, None] = None,
                 progress_callback: Optional[Callable[[float], None]] = None) -> bytes:
        if level is not None:
            self.compression_level(level)
        data = bytes(data)
        self._report(progress_callback, 0.0)
        tokens = self.lz77_compress(data)
        token_bytes = self._serialize_tokens(tokens)
        self._report(progress_callback, 0.4)
        payload = self.huffman_compress(token_bytes)
        body = self._serialize_payload(payload)
        self._report(progress_callback, 0.8)
        use_dict = self.use_dictionary and bool(self.dictionary)
        flags = self.FLAG_DICTIONARY if use_dict else 0
        header = self._HEADER.pack(self.MAGIC, self.VERSION, self.level, flags, 0,
                                   len(data), len(self.dictionary) if use_dict else 0,
                                   zlib.crc32(data))
        self._report(progress_callback, 1.0)
        return header + body

    def decompress(self, blob: bytes,
                   progress_callback: Optional[Callable[[float], None]] = None) -> bytes:
        self._report(progress_callback, 0.0)
        blob = bytes(blob)
        if len(blob) < self._HEADER.size:
            raise ValueError("数据损坏：头部不完整")
        magic, version, _level, flags, _reserved, orig_size, dict_size, crc = \
            self._HEADER.unpack(blob[:self._HEADER.size])
        if magic != self.MAGIC:
            raise ValueError("数据损坏：魔数不匹配")
        if version != self.VERSION:
            raise ValueError(f"不支持的格式版本: {version}")
        if flags & self.FLAG_DICTIONARY and len(self.dictionary) != dict_size:
            raise ValueError("数据损坏：字典不匹配")
        payload = self._deserialize_payload(blob[self._HEADER.size:])
        self._report(progress_callback, 0.3)
        token_bytes = self.huffman_decompress(payload)
        self._report(progress_callback, 0.6)
        tokens = self._deserialize_tokens(token_bytes)
        data = self.lz77_decompress(tokens)
        self._report(progress_callback, 0.9)
        if len(data) != orig_size:
            raise ValueError("数据损坏：解压长度不匹配")
        if zlib.crc32(data) != crc:
            raise ValueError("数据损坏：校验和不匹配")
        self._report(progress_callback, 1.0)
        return data

    def get_progress(self) -> float:
        return self.progress

    def _report(self, callback, value: float):
        self.progress = value
        if callback is not None:
            callback(value)

    # ------------------------------------------------------------------
    # 序列化
    # ------------------------------------------------------------------
    def _serialize_tokens(self, tokens: List[Token]) -> bytes:
        out = bytearray(struct.pack(">I", len(tokens)))
        for offset, length, literal in tokens:
            out += self._TOKEN.pack(offset, length, literal)
        return bytes(out)

    def _deserialize_tokens(self, blob: bytes) -> List[Token]:
        if len(blob) < 4:
            raise ValueError("数据损坏：令牌流不完整")
        count = struct.unpack(">I", blob[:4])[0]
        expected = 4 + count * self._TOKEN.size
        if len(blob) != expected:
            raise ValueError("数据损坏：令牌流长度不一致")
        tokens = []
        pos = 4
        for _ in range(count):
            tokens.append(self._TOKEN.unpack(blob[pos:pos + self._TOKEN.size]))
            pos += self._TOKEN.size
        return tokens

    def _serialize_payload(self, payload: Dict) -> bytes:
        freqs = payload["freqs"]
        out = bytearray(struct.pack(">H", len(freqs)))
        for sym in sorted(freqs):
            out += struct.pack(">BI", sym, freqs[sym])
        out += struct.pack(">BII", payload["padding"], payload["length"],
                           len(payload["data"]))
        out += payload["data"]
        return bytes(out)

    def _deserialize_payload(self, blob: bytes) -> Dict:
        try:
            if len(blob) < 2:
                raise ValueError
            num_symbols = struct.unpack(">H", blob[:2])[0]
            pos = 2
            freqs: Dict[int, int] = {}
            for _ in range(num_symbols):
                sym, freq = struct.unpack(">BI", blob[pos:pos + 5])
                pos += 5
                freqs[sym] = freq
            padding, length, data_len = struct.unpack(">BII", blob[pos:pos + 9])
            pos += 9
            data = blob[pos:pos + data_len]
            if len(data) != data_len or pos + data_len != len(blob):
                raise ValueError
        except (struct.error, IndexError, ValueError):
            raise ValueError("数据损坏：负载不完整")
        return {"freqs": freqs, "padding": padding, "length": length, "data": data}
