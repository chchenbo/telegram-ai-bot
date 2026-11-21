import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from ai_bot.textutil import (  # noqa: E402
    sanitize_for_telegram,
    split_text,
    take_prefix,
    trim_history,
    utf16_len,
)


class Utf16LenTest(unittest.TestCase):
    def test_ascii(self):
        self.assertEqual(utf16_len("abc"), 3)

    def test_cjk(self):
        self.assertEqual(utf16_len("你好"), 2)

    def test_emoji_counts_double(self):
        self.assertEqual(utf16_len("😀"), 2)
        self.assertEqual(utf16_len("a😀b"), 4)


class SplitTextTest(unittest.TestCase):
    def test_short_text_is_untouched(self):
        self.assertEqual(split_text("你好，世界"), ["你好，世界"])

    def test_blank_input_yields_nothing(self):
        self.assertEqual(split_text("  \n\t "), [])

    def test_every_chunk_fits_the_utf16_budget(self):
        text = ("第一段内容，讲一点事情。" * 80 + "\n\n") * 4
        chunks = split_text(text, 100)
        self.assertGreater(len(chunks), 1)
        for chunk in chunks:
            self.assertLessEqual(utf16_len(chunk), 100)

    def test_reassembly_loses_nothing_but_whitespace(self):
        text = "\n".join(f"第 {i} 行：一些中文内容，带标点。" for i in range(400))
        chunks = split_text(text, 120)
        joined = "".join(chunks).replace("\n", "").replace(" ", "")
        self.assertEqual(joined, text.replace("\n", "").replace(" ", ""))

    def test_emoji_never_split_in_the_middle(self):
        text = "😀" * 3000
        chunks = split_text(text, 100)
        for chunk in chunks:
            self.assertLessEqual(utf16_len(chunk), 100)
        self.assertEqual("".join(chunks), text)

    def test_single_huge_token_still_progresses(self):
        chunks = split_text("x" * 10000, 50)
        self.assertEqual(sum(len(c) for c in chunks), 10000)
        self.assertTrue(all(len(c) <= 50 for c in chunks))


class TakePrefixTest(unittest.TestCase):
    def test_lossless(self):
        text = "abcdef\nghijklmn" * 40
        head, rest = take_prefix(text, 40)
        self.assertEqual(head + rest, text)
        self.assertLessEqual(utf16_len(head), 40)
        self.assertTrue(head)

    def test_prefers_natural_break(self):
        text = "第一句话结束了。第二句话也结束了。"
        head, _ = take_prefix(text, 12)
        self.assertTrue(head.endswith("。"))

    def test_short_text_returns_all(self):
        self.assertEqual(take_prefix("abc", 10), ("abc", ""))


class TrimHistoryTest(unittest.TestCase):
    def _history(self, n):
        return [
            {"role": "user" if i % 2 == 0 else "assistant", "content": f"m{i}"}
            for i in range(n)
        ]

    def test_keeps_message_limit(self):
        trimmed = trim_history(self._history(100), max_messages=10, max_chars=10**9)
        self.assertEqual(len(trimmed), 10)
        self.assertEqual(trimmed[-1]["content"], "m99")

    def test_never_starts_with_assistant(self):
        trimmed = trim_history(self._history(10), max_messages=5, max_chars=10**9)
        self.assertEqual(trimmed[0]["role"], "user")

    def test_respects_char_budget(self):
        history = [{"role": "user", "content": "x" * 100} for _ in range(20)]
        trimmed = trim_history(history, max_messages=100, max_chars=250)
        self.assertLessEqual(sum(len(m["content"]) for m in trimmed), 250)
        self.assertEqual(len(trimmed), 2)

    def test_empty_history(self):
        self.assertEqual(trim_history([]), [])


class SanitizeTest(unittest.TestCase):
    def test_strips_control_chars_but_keeps_newlines(self):
        self.assertEqual(sanitize_for_telegram("a\x00b\x07c\n\t d"), "abc\n\t d")

    def test_trims_outer_whitespace(self):
        self.assertEqual(sanitize_for_telegram("\n  hi  \n"), "hi")


if __name__ == "__main__":
    unittest.main()
