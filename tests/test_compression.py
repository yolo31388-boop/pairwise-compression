"""
游戏内压缩算法与数据压缩系统 - 验收测试
运行方式：python -m pytest tests/test_compression.py -q
共 14 个测试用例
"""
import pytest
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.compression import Compression


class TestCompression:
    def setup_method(self):
        self.system = Compression()

    # ---- LZ77：滑动窗口 ----
    def test_case_01(self):
        # 重复数据应被压缩为匹配 token，而不是逐字节字面量
        data = b"abcabcabcabcabcabc"
        tokens = self.system.lz77_compress(data)
        matches = [t for t in tokens if t[0] == "match"]
        assert len(matches) > 0, "重复数据未产生任何匹配，LZ77 未生效"
        assert len(tokens) < len(data), "压缩后 token 数不少于原始字节数"

    # ---- LZ77：滑动窗口边界 ----
    def test_case_02(self):
        # 重复间隔在窗口内能找到匹配，超出窗口则不能
        unit = b"XYZ"
        gap = bytes(range(1, 101))
        data = unit + gap + unit
        tokens = self.system.lz77_compress(data, window_size=200, min_match=3)
        assert any(t[0] == "match" for t in tokens), "窗口内的重复未被找到"
        tokens_small = self.system.lz77_compress(data, window_size=10, min_match=3)
        assert all(t[0] == "lit" for t in tokens_small), "超出窗口的重复不应被匹配"

    # ---- LZ77：最长匹配 ----
    def test_case_03(self):
        # 贪心最长匹配：aaaa... 应产生一个长匹配而非多个短匹配
        data = b"a" * 20
        tokens = self.system.lz77_compress(data, min_match=3, max_match=258)
        matches = [t for t in tokens if t[0] == "match"]
        assert len(matches) == 1, f"应只有一个最长匹配，实际 {len(matches)} 个"
        assert matches[0][2] == 19, f"最长匹配长度应为 19，实际 {matches[0][2]}"

    # ---- LZ77：往返 ----
    def test_case_04(self):
        data = bytes(range(256)) * 3 + b"hello hello hello"
        tokens = self.system.lz77_compress(data)
        assert self.system.lz77_decompress(tokens) == data

    # ---- Huffman：频率统计 ----
    def test_case_05(self):
        data = b"aaabbc"
        freq = self.system.huffman_frequencies(data)
        assert freq == {ord('a'): 3, ord('b'): 2, ord('c'): 1}

    # ---- Huffman：前缀编码 + 频率越高编码越短 ----
    def test_case_06(self):
        data = b"aaaaaaaaaaaaaabbbbc"
        payload = self.system.huffman_compress(data)
        codes = payload["codes"]
        # 前缀性：任何编码都不是另一个编码的前缀
        all_codes = list(codes.values())
        for c1 in all_codes:
            for c2 in all_codes:
                if c1 != c2:
                    assert not c2.startswith(c1), f"{c2} 以 {c1} 为前缀，违反前缀编码"
        # 频率最高的符号编码长度不超过频率最低的
        assert len(codes[ord('a')]) <= len(codes[ord('c')])

    # ---- Huffman：往返 ----
    def test_case_07(self):
        data = b"the quick brown fox jumps over the lazy dog" * 5
        payload = self.system.huffman_compress(data)
        assert self.system.huffman_decompress(payload) == data

    # ---- Huffman：实际压缩效果 ----
    def test_case_08(self):
        # 高度偏斜的数据，Huffman 编码后应明显变小
        data = b"a" * 900 + b"b" * 90 + b"c" * 10
        payload = self.system.huffman_compress(data)
        assert len(payload["data"]) < len(data) // 2, "Huffman 未产生压缩效果"

    # ---- 解压：完整往返 ----
    def test_case_09(self):
        data = b"game-resource-block-" * 200
        payload = self.system.compress(data, level="default")
        assert len(payload["body"]) < len(data), "压缩包不比原始数据小"
        assert self.system.decompress(payload) == data

    # ---- 解压：损坏数据验证 ----
    def test_case_10(self):
        data = b"important game data" * 50
        payload = self.system.compress(data)
        # 篡改内容
        corrupted = dict(payload)
        corrupted["body"] = payload["body"][:-1] + bytes([payload["body"][-1] ^ 0xFF])
        with pytest.raises(ValueError):
            self.system.decompress(corrupted)
        # 篡改校验和
        bad_checksum = dict(payload)
        bad_checksum["checksum"] = payload["checksum"] ^ 0xFFFF
        with pytest.raises(ValueError):
            self.system.decompress(bad_checksum)
        # 非法魔数
        with pytest.raises(ValueError):
            self.system.decompress({"magic": b"XXXX"})

    # ---- 解压：进度回调 ----
    def test_case_11(self):
        data = b"progress-tracking-data" * 100
        payload = self.system.compress(data)
        progress = []
        result = self.system.decompress(payload, progress_callback=progress.append)
        assert result == data
        assert len(progress) >= 2, "解压过程未汇报进度"
        assert progress[0] == 0.0 and progress[-1] == 1.0
        assert progress == sorted(progress), "进度应单调递增"

    # ---- 字典：训练 ----
    def test_case_12(self):
        samples = [b"player_position_update_" * 3, b"player_position_update_" * 2]
        dictionary = self.system.train_dictionary(samples)
        assert len(dictionary) > 0, "训练后字典为空"
        assert any(b"player_position" in pat for pat in dictionary), \
            "常见模式未进入字典"

    # ---- 字典：更新 + 压缩往返 ----
    def test_case_13(self):
        self.system.train_dictionary([b"alpha_beta_gamma_" * 4])
        # 新模式通过 update 加入字典
        self.system.update_dictionary(b"delta_epsilon_zeta_" * 4)
        assert any(b"delta_epsilon" in pat for pat in self.system._dictionary), \
            "新数据模式未加入字典"
        # 字典压缩能压缩重复数据并可还原
        data = b"alpha_beta_gamma_" * 10
        tokens = self.system.dictionary_compress(data)
        assert len(tokens) < len(data) // 4, "字典未压缩重复数据"
        assert self.system.dictionary_decompress(tokens) == data

    # ---- 压缩级别：参数与预设 ----
    def test_case_14(self):
        # 三个预设齐全
        for name in ("fast", "default", "max"):
            params = self.system.compression_level(name)
            assert "window_size" in params and "max_match" in params
        fast = self.system.compression_level("fast")
        maximum = self.system.compression_level("max")
        # 级别参数体现速度/压缩率权衡
        assert fast["window_size"] < maximum["window_size"]
        assert fast["max_match"] < maximum["max_match"]
        # 整数级别可用且单调
        lvl1 = self.system.compression_level(1)
        lvl9 = self.system.compression_level(9)
        assert lvl1["window_size"] <= lvl9["window_size"]
        # max 级别压缩率不低于 fast 级别
        data = (b"texture_chunk_" + bytes(range(64))) * 100
        fast_size = len(self.system.compress(data, "fast")["body"])
        max_size = len(self.system.compress(data, "max")["body"])
        assert max_size <= fast_size, "最大压缩级别反而更大"
        # 非法级别报错
        with pytest.raises(ValueError):
            self.system.compression_level("ultra")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
