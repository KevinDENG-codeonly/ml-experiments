import pytest
import torch
from unittest.mock import MagicMock, patch
from core.train import set_seed, evaluate_model

@patch("core.train.random.seed")
@patch("core.train.np.random.seed")
@patch("core.train.torch.manual_seed")
@patch("core.train.torch.cuda.is_available", return_value=False)
def test_set_seed_basic(mock_cuda, mock_torch, mock_np, mock_random):
    set_seed(42)
    mock_random.assert_called_with(42)
    mock_np.assert_called_with(42)
    mock_torch.assert_called_with(42)

@patch("core.train.SmoothingFunction")
def test_evaluate_model_basic(mock_smooth):
    # Mock encoder, decoder, dataloader, vocab
    encoder = MagicMock()
    decoder = MagicMock()
    decoder.vocab_size = 10
    decoder.sample.return_value = [1, 2, 3]
    encoder.eval = MagicMock()
    decoder.eval = MagicMock()
    # Mock dataloader
    batch = (
        torch.randn(2, 3, 224, 224),  # images
        torch.randint(0, 10, (2, 5)), # captions
        torch.tensor([5, 5]),         # lengths
        torch.tensor([1, 2])          # image_ids
    )
    dataloader = [batch]
    # Mock vocab
    class DummyVocab:
        def __call__(self, x): return 1
        def get_word(self, idx): return f"word{idx}"
    vocab = DummyVocab()
    # Mock criterion
    criterion = torch.nn.CrossEntropyLoss()
    # Patch BLEU smoothing
    mock_smooth.return_value.method1 = lambda: None
    avg_loss, avg_bleu = evaluate_model(
        encoder, decoder, dataloader, criterion, vocab, torch.device("cpu"), 5, False
    )
    assert isinstance(avg_loss, float)
    assert isinstance(avg_bleu, float)

@patch("core.train.SmoothingFunction")
def test_evaluate_model_empty_dataloader(mock_smooth):
    encoder = MagicMock()
    decoder = MagicMock()
    decoder.vocab_size = 10
    decoder.sample.return_value = [1, 2, 3]
    encoder.eval = MagicMock()
    decoder.eval = MagicMock()
    dataloader = []
    class DummyVocab:
        def __call__(self, x): return 1
        def get_word(self, idx): return f"word{idx}"
    vocab = DummyVocab()
    criterion = torch.nn.CrossEntropyLoss()
    mock_smooth.return_value.method1 = lambda: None
    avg_loss, avg_bleu = evaluate_model(
        encoder, decoder, dataloader, criterion, vocab, torch.device("cpu"), 5, False
    )
    assert avg_loss == 0
    assert avg_bleu == 0 