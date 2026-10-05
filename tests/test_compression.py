"""
游戏内压缩算法与数据压缩系统 - 验收测试
运行方式：python -m pytest tests/test_compression.py -q
共 14 个测试用例
"""
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.compression import Compression


class TestCompression:
    def setup_method(self):
        self.system = Compression()

    # ---- LZ77：滑动窗口 + 最长匹配 ----

    def test_case_01(self):
        """LZ77 对重复数据产生匹配令牌，整管线压缩后更小"""
        data = b"PlayerPos:x=100,y=200;" * 400
        tokens = self.system.lz77_compress(data)
        assert any(offset > 0 and length > 0 for offset, length, _ in tokens)
        assert len(tokens) < len(data)
        assert len(self.system.compress(data)) < len(data)

    def test_case_02(self):
        """LZ77 有滑动窗口：偏移不超过窗口，超窗重复无法匹配"""
        pattern = b"ABCDEFGH"
        data = pattern + b"--------" + pattern
        small = self.system.lz77_compress(data, window_size=8)
        assert all(offset <= 8 for offset, _, _ in small)
        assert not any(length >= 8 for _, length, _ in small)
        large = self.system.lz77_compress(data, window_size=32)
        assert any(offset == 16 and length > 0 for offset, length, _ in large)

    def test_case_03(self):
        """LZ77 使用最长匹配而非最短匹配"""
        tokens = self.system.lz77_compress(b"abcabcabc", max_match=258)
        match = max((t for t in tokens if t[1] > 0), key=lambda t: t[1], default=None)
        assert match is not None and match[1] == 6

    def test_case_04(self):
        """LZ77 解压往返一致（空、单字节、全重复、随机数据）"""
        samples = [b"", b"x", b"a" * 1000,
                   bytes(random.Random(42).randbytes(500))]
        for data in samples:
            tokens = self.system.lz77_compress(data)
            assert self.system.lz77_decompress(tokens) == data

    # ---- Huffman：频率统计 + 编码树 ----

    def test_case_05(self):
        """Huffman 频率统计准确"""
        freqs = self.system.build_frequency(b"aabbbcccc")
        assert freqs[ord("a")] == 2
        assert freqs[ord("b")] == 3
        assert freqs[ord("c")] == 4
        assert sum(freqs.values()) == 9

    def test_case_06(self):
        """Huffman 根据频率生成无前缀冲突的最优前缀编码"""
        data = b"a" * 100 + b"b" * 10 + b"c"
        tree = self.system._build_tree(self.system.build_frequency(data))
        codes = self.system._generate_codes(tree)
        assert tree is not None
        symbols = list(codes)
        for s1 in symbols:
            for s2 in symbols:
                if s1 != s2:
                    assert not codes[s1].startswith(codes[s2])
        assert len(codes[ord("a")]) <= len(codes[ord("b")])
        assert len(codes[ord("b")]) <= len(codes[ord("c")])

    def test_case_07(self):
        """Huffman 压缩高频偏斜数据真正变小且可还原"""
        data = b"a" * 900 + b"bcdef" * 20
        payload = self.system.huffman_compress(data)
        assert len(payload["data"]) < len(data)
        assert self.system.huffman_decompress(payload) == data

    # ---- 解压：验证 + 进度 ----

    def test_case_08(self):
        """完整管线压缩/解压往返一致"""
        samples = [b"", b"only", bytes(range(256)) * 4,
                   b"game-resource-block-" * 300]
        for data in samples:
            blob = self.system.compress(data)
            assert self.system.decompress(blob) == data

    def test_case_09(self):
        """解压验证：魔数错误、数据损坏、截断都应拒绝"""
        blob = bytearray(self.system.compress(b"save-file-content" * 50))
        bad_magic = b"XXXX" + bytes(blob[4:])
        try:
            self.system.decompress(bad_magic)
            assert False, "错误魔数应抛 ValueError"
        except ValueError:
            pass
        corrupted = bytearray(blob)
        corrupted[len(corrupted) // 2] ^= 0xFF
        try:
            self.system.decompress(bytes(corrupted))
            assert False, "损坏数据应抛 ValueError"
        except ValueError:
            pass
        try:
            self.system.decompress(bytes(blob[:10]))
            assert False, "截断数据应抛 ValueError"
        except ValueError:
            pass

    def test_case_10(self):
        """解压有进度：单调递增，从 0.0 到 1.0"""
        blob = self.system.compress(b"decompress-progress-" * 200)
        progress = []
        result = self.system.decompress(blob, progress_callback=progress.append)
        assert result == b"decompress-progress-" * 200
        assert len(progress) >= 3
        assert progress[0] == 0.0
        assert progress[-1] == 1.0
        assert progress == sorted(progress)
        assert all(0.0 <= p <= 1.0 for p in progress)

    # ---- 字典：训练 + 更新 ----

    def test_case_11(self):
        """字典能从样本数据中训练出常见模式"""
        pattern = b"PlayerHealth=100;PlayerMana=50;"
        samples = [pattern * 10 + suffix for suffix in
                   (b"aaa", b"bbb", b"ccc")]
        size = self.system.train_dictionary(samples, pattern_length=8)
        assert size > 0
        assert pattern[:8] in self.system.dictionary

    def test_case_12(self):
        """训练过的字典让含已知模式的数据压缩率更高"""
        pattern = b"NetPacket:seq=%d,state=syn,pos=(1.0,2.0,3.0);"
        samples = [pattern * 8 + tag for tag in (b"x", b"y", b"z")]
        self.system.train_dictionary(samples, pattern_length=8)
        data = pattern * 20
        with_dict = len(self.system.compress(data))
        no_dict = len(Compression({"level": "fast"}).compress(data))
        assert with_dict < no_dict

    def test_case_13(self):
        """字典支持增量更新，新模式被纳入且往返仍正确"""
        self.system.train_dictionary([b"oldpattern!!" * 8] * 2, pattern_length=8)
        before = len(self.system.dictionary)
        new_data = b"brandnewpattern###" * 10
        added = self.system.update_dictionary(new_data, pattern_length=8)
        assert added > 0
        assert len(self.system.dictionary) == before + added
        assert b"brandnew" in self.system.dictionary
        payload = self.system.compress(new_data)
        assert self.system.decompress(payload) == new_data

    # ---- 压缩级别：参数 + 预设 ----

    def test_case_14(self):
        """压缩级别有 fast/default/max 预设与可调参数，权衡生效"""
        assert set(Compression.LEVEL_PRESETS) == {"fast", "default", "max"}
        fast = Compression().compression_level("fast")
        default = Compression().compression_level("default")
        maximum = Compression().compression_level("max")
        assert fast["window_size"] < default["window_size"] < maximum["window_size"]
        assert fast["max_match"] < maximum["max_match"]
        custom = Compression().compression_level(
            {"window_size": 2048, "max_match": 32, "min_match": 4,
             "max_chain": 4, "use_dictionary": False})
        assert custom["window_size"] == 2048 and custom["min_match"] == 4
        data = b"abcdefgh" * 2000
        size_fast = len(Compression().compress(data, level="fast"))
        size_max = len(Compression().compress(data, level="max"))
        assert size_max <= size_fast


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
