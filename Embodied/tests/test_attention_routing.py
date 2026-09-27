import importlib.util
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from transformers import AutoConfig

MODULE_PATH = Path(__file__).resolve().parents[1] / "locateanything_worker.py"
spec = importlib.util.spec_from_file_location("locateanything_worker", MODULE_PATH)
worker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(worker)


class AttentionRoutingTest(unittest.TestCase):
    def setUp(self):
        self.config = SimpleNamespace(
            _attn_implementation="magi",
            text_config=SimpleNamespace(_attn_implementation="magi"),
            vision_config=SimpleNamespace(_attn_implementation=None),
        )
        self.model = mock.Mock(config=self.config)
        self.model.to.return_value = self.model
        self.model.eval.return_value = self.model
        self.tokenizer = mock.patch.object(worker.AutoTokenizer, "from_pretrained").start()
        self.processor = mock.patch.object(worker.AutoProcessor, "from_pretrained").start()
        self.config_loader = mock.patch.object(AutoConfig, "from_pretrained", return_value=self.config).start()
        self.model_loader = mock.patch.object(worker.AutoModel, "from_pretrained", return_value=self.model).start()
        self.addCleanup(mock.patch.stopall)

    def test_explicit_sdpa_reaches_text_without_changing_vision_default(self):
        worker.LocateAnythingWorker("model", device="cpu", attn="sdpa")
        self.config_loader.assert_called_once()
        self.assertEqual(self.config._attn_implementation, "sdpa")
        self.assertEqual(self.config.text_config._attn_implementation, "sdpa")
        self.assertIsNone(self.config.vision_config._attn_implementation)
        self.assertIs(self.model_loader.call_args.kwargs["config"], self.config)

    def test_default_keeps_saved_configuration(self):
        worker.LocateAnythingWorker("model", device="cpu")
        self.config_loader.assert_not_called()
        self.assertNotIn("config", self.model_loader.call_args.kwargs)

    def test_unknown_backend_fails_before_loading(self):
        with self.assertRaisesRegex(ValueError, "standard"):
            worker.LocateAnythingWorker("model", device="cpu", attn="typo")
        self.model_loader.assert_not_called()


    def test_explicit_vision_backend_is_independent(self):
        worker.LocateAnythingWorker("model", device="cpu", attn="sdpa", vision_attn="sdpa")
        self.assertEqual(self.config.text_config._attn_implementation, "sdpa")
        self.assertEqual(self.config.vision_config._attn_implementation, "sdpa")
        self.assertIs(self.model_loader.call_args.kwargs["config"], self.config)

    def test_batch_default_preserves_la_flash(self):
        with mock.patch.object(worker.LocateAnythingWorker, "_init_batch_runtime") as init:
            worker.LocateAnythingWorker("model", use_batch_runtime=True)
        self.assertEqual(init.call_args.kwargs["attn"], "la_flash")
        self.model_loader.assert_not_called()

    def test_effective_backend_mismatch_fails(self):
        def load_model(*args, **kwargs):
            self.config.text_config._attn_implementation = "magi"
            return self.model
        self.model_loader.side_effect = load_model
        with self.assertRaisesRegex(RuntimeError, "not activated"):
            worker.LocateAnythingWorker("model", device="cpu", attn="sdpa")


if __name__ == "__main__":
    unittest.main()
