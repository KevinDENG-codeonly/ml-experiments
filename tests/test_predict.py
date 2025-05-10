import pytest
from unittest.mock import patch, MagicMock
from pathlib import Path
from core.predict import load_model_and_vocab, preprocess_image, generate_caption

@patch("core.predict.torch.load")
@patch("core.predict.CocoCaptionsDataset.load_vocabulary")
@patch("core.predict.EncoderCNN")
@patch("core.predict.DecoderRNN")
def test_load_model_and_vocab_success(mock_decoder, mock_encoder, mock_load_vocab, mock_torch_load, tmp_path):
    ckpt_path = tmp_path / "model.pth"
    ckpt_path.write_bytes(b"fake")
    mock_torch_load.return_value = {
        'config': {
            'model': {'embed_size': 8, 'hidden_size': 8, 'encoder_cnn_model': 'resnet50', 'decoder_num_layers': 1},
            'dataset': {'vocab_path': str(tmp_path / 'vocab.json')}
        },
        'vocab_size': 4,
        'encoder_state_dict': {},
        'decoder_state_dict': {}
    }
    mock_load_vocab.return_value = MagicMock(__len__=lambda self: 4, __call__=lambda self, x: 1)
    mock_encoder.return_value = MagicMock()
    mock_decoder.return_value = MagicMock()
    encoder, decoder, vocab, cfg, start_idx, end_idx = load_model_and_vocab(ckpt_path, MagicMock())
    assert encoder is not None
    assert decoder is not None
    assert hasattr(vocab, "__call__")
    assert isinstance(cfg, dict)

@patch("core.predict.torch.load")
def test_load_model_and_vocab_missing_ckpt(mock_torch_load, tmp_path):
    with pytest.raises(FileNotFoundError):
        load_model_and_vocab(tmp_path / "missing.pth", MagicMock())

@patch("core.predict.Image.open")
def test_preprocess_image_success(mock_open, tmp_path):
    img_path = tmp_path / "img.jpg"
    img_path.write_bytes(b"fake")
    mock_img = MagicMock()
    mock_open.return_value.convert.return_value = mock_img
    tensor = preprocess_image(img_path, 224, MagicMock())
    assert hasattr(tensor, "shape") or hasattr(tensor, "size")

def test_preprocess_image_missing(tmp_path):
    with pytest.raises(FileNotFoundError):
        preprocess_image(tmp_path / "missing.jpg", 224, MagicMock())

@patch("core.predict.EncoderCNN")
@patch("core.predict.DecoderRNN")
def test_generate_caption_basic(mock_decoder, mock_encoder):
    encoder = MagicMock()
    decoder = MagicMock()
    encoder.return_value = MagicMock()
    decoder.sample.return_value = [1, 2, 3]
    class DummyVocab:
        def __call__(self, x): return 1
        def get_word(self, idx): return f"word{idx}"
    vocab = DummyVocab()
    image_tensor = MagicMock()
    caption = generate_caption(encoder, decoder, image_tensor, vocab, 1, 2, 5, MagicMock())
    assert isinstance(caption, str) 