import os
import unittest


if os.environ.get('RUN_TORCH_TESTS') != '1':

    class CallbackTests(unittest.TestCase):
        @unittest.skip("Set RUN_TORCH_TESTS=1 to enable callback tests.")
        def test_placeholder(self):
            pass


else:
    from src.training.callbacks.base import CallbackList, TrainerCallback, BatchOutput
    from src.training.callbacks.diagnostics import DiagnosticsCallback


    class DummyTrainer:
        def __init__(self) -> None:
            self.logs = []


    class RecordingCallback(TrainerCallback):
        def __init__(self) -> None:
            self.events = []

        def on_train_begin(self, trainer):
            self.events.append('begin')


    class CallbackTests(unittest.TestCase):
        def test_callback_list_invocation(self):
            recorder = RecordingCallback()
            callbacks = CallbackList([recorder])
            trainer = DummyTrainer()
            callbacks.on_train_begin(trainer)
            self.assertEqual(recorder.events, ['begin'])

        def test_diagnostics_batch_accumulation(self):
            diag = DiagnosticsCallback(model_dir='/tmp')
            trainer = DummyTrainer()
            diag.on_batch_end(trainer, 0, 0, BatchOutput(loss=1.0, loss_components={'MSE': 1.0}))
            self.assertIn('MSE', diag.train_history)


if __name__ == '__main__':
    unittest.main()
