import pathlib
import sys
import tempfile
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

from ai_bot.store import ChatState, Store  # noqa: E402


class StoreTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.path = pathlib.Path(self.tmp.name) / "nested" / "state.json"

    def tearDown(self):
        self.tmp.cleanup()

    def test_missing_file_is_empty(self):
        store = Store(self.path)
        store.load()
        self.assertEqual(store.chats, {})

    def test_roundtrip(self):
        store = Store(self.path)
        state = store.get(42)
        state.history = [{"role": "user", "content": "hello"}]
        state.model = "deepseek-v4-pro"
        state.thinking = True
        state.total_turns = 3
        store.save()

        reloaded = Store(self.path)
        reloaded.load()
        again = reloaded.get(42)
        self.assertEqual(again.history, [{"role": "user", "content": "hello"}])
        self.assertEqual(again.model, "deepseek-v4-pro")
        self.assertIs(again.thinking, True)
        self.assertEqual(again.total_turns, 3)

    def test_corrupt_file_is_backed_up_and_ignored(self):
        self.path.parent.mkdir(parents=True)
        self.path.write_text("{not json", encoding="utf-8")
        store = Store(self.path)
        store.load()
        self.assertEqual(store.chats, {})
        self.assertTrue(self.path.with_suffix(".json.broken").exists())

    def test_unknown_roles_are_dropped(self):
        state = ChatState.from_dict(
            {
                "history": [
                    {"role": "user", "content": "a"},
                    {"role": "system", "content": "b"},
                    {"role": "assistant", "content": "c"},
                ]
            }
        )
        self.assertEqual([m["role"] for m in state.history], ["user", "assistant"])

    def test_save_is_atomic_enough_for_reload(self):
        store = Store(self.path)
        store.get(1).history = [{"role": "user", "content": "x"}]
        store.save()
        store.get(1).history.append({"role": "assistant", "content": "y"})
        store.save()
        reloaded = Store(self.path)
        reloaded.load()
        self.assertEqual(len(reloaded.get(1).history), 2)
        self.assertFalse(self.path.with_suffix(".json.tmp").exists())


if __name__ == "__main__":
    unittest.main()
