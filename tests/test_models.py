import pytest
import torch

from src.models import EncoderCNN, DecoderRNN
from core.utils import START_TOKEN, END_TOKEN # Import needed for sampling test

# --- Fixtures ---
@pytest.fixture
def dummy_encoder_input() -> torch.Tensor:
    """Provides a dummy batch of images."""
    return torch.randn(4, 3, 224, 224) # Batch size 4, C=3, H=224, W=224

@pytest.fixture
def dummy_decoder_input(dummy_encoder: EncoderCNN) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Provides dummy features, captions, and lengths for the decoder."""
    batch_size = 4
    embed_size = dummy_encoder.linear.out_features # Match encoder output size
    vocab_size = 1000
    max_len = 15
    
    # Dummy features (simulate output from encoder)
    features = torch.randn(batch_size, embed_size)
    
    # Dummy captions (sorted by length descending for packing)
    captions = torch.randint(4, vocab_size, (batch_size, max_len), dtype=torch.long)
    # Add start token (index 1) and end token (index 2), pad with 0
    lengths = torch.tensor([12, 10, 8, 5], dtype=torch.long) # Example sorted lengths
    captions_padded = torch.zeros((batch_size, max_len), dtype=torch.long)
    for i in range(batch_size):
        seq_len = lengths[i].item()
        captions_padded[i, 0] = 1 # Start token
        captions_padded[i, 1:seq_len-1] = captions[i, :seq_len-2]
        captions_padded[i, seq_len-1] = 2 # End token
        # Rest are zeros (padding)

    return features, captions_padded, lengths

@pytest.fixture
def dummy_vocab() -> dict:
    """Provides a minimal mock vocabulary for sampling tests."""
    # In a real scenario, we might load the actual vocab or use a more complex mock
    # For sampling, we mainly need get_word and indices for special tokens
    class MockVocab:
        def __init__(self):
            self.word2idx = {PAD_TOKEN: 0, START_TOKEN: 1, END_TOKEN: 2, UNKNOWN_TOKEN: 3, "hello": 4, "world": 5}
            self.idx2word = {v: k for k, v in self.word2idx.items()}
        def __call__(self, word):
            return self.word2idx.get(word, 3)
        def get_word(self, idx):
            return self.idx2word.get(idx)
    return MockVocab()

# --- EncoderCNN Tests ---
@pytest.mark.parametrize("model_name", ["resnet50", "resnet101"])
@pytest.mark.parametrize("embed_size", [128, 512])
def test_encoder_cnn_init_and_forward(dummy_encoder_input, model_name, embed_size):
    """Tests EncoderCNN initialization and forward pass shape."""
    encoder = EncoderCNN(embed_size=embed_size, cnn_model_name=model_name, pretrained=True)
    assert encoder.linear.out_features == embed_size
    
    features = encoder(dummy_encoder_input)
    assert features.shape == (dummy_encoder_input.size(0), embed_size)
    # Check requires_grad (only last linear/bn layer should have requires_grad=True by default)
    # This depends on whether we uncomment the freezing loop in __init__
    # assert not next(encoder.cnn.parameters()).requires_grad
    assert encoder.linear.weight.requires_grad
    assert encoder.bn.weight.requires_grad

def test_encoder_cnn_unsupported_model():
    """Tests that an unsupported CNN model name raises ValueError."""
    with pytest.raises(ValueError):
        EncoderCNN(embed_size=256, cnn_model_name="unsupported_cnn")

# --- DecoderRNN Tests ---
@pytest.mark.parametrize("num_layers", [1, 2])
@pytest.mark.parametrize("hidden_size", [256, 512])
def test_decoder_rnn_init_and_forward(dummy_decoder_input, num_layers, hidden_size):
    """Tests DecoderRNN initialization and forward pass shape with packing."""
    features, captions, lengths = dummy_decoder_input
    embed_size = features.size(1)
    vocab_size = 1000 # Should match dummy data generation if needed
    batch_size = features.size(0)
    max_len = captions.size(1)

    decoder = DecoderRNN(
        embed_size=embed_size, 
        hidden_size=hidden_size, 
        vocab_size=vocab_size, 
        num_layers=num_layers
    )
    
    output = decoder(features, captions, lengths)
    
    # Output shape should be (batch_size, seq_len - 1, vocab_size)
    assert output.shape == (batch_size, max_len - 1, vocab_size)

def test_decoder_rnn_init_hidden_state(dummy_decoder_input):
    """Tests the shape of the initial hidden state."""
    features, _, _ = dummy_decoder_input
    embed_size = features.size(1)
    vocab_size = 1000
    hidden_size = 512
    num_layers = 2
    batch_size = features.size(0)
    
    decoder = DecoderRNN(embed_size, hidden_size, vocab_size, num_layers)
    h0, c0 = decoder.init_hidden_state(features)
    
    assert h0.shape == (num_layers, batch_size, hidden_size)
    assert c0.shape == (num_layers, batch_size, hidden_size)

def test_decoder_rnn_sample(dummy_vocab):
    """Tests the DecoderRNN sample method for basic execution and output format."""
    embed_size = 128
    hidden_size = 256
    vocab_size = len(dummy_vocab.word2idx)
    num_layers = 1
    decoder = DecoderRNN(embed_size, hidden_size, vocab_size, num_layers)
    decoder.eval() # Set to eval mode for sampling
    
    # Create single dummy feature
    image_feature = torch.randn(1, embed_size)
    
    start_idx = dummy_vocab(START_TOKEN)
    end_idx = dummy_vocab(END_TOKEN)
    max_len_sample = 10

    with torch.no_grad():
        sampled_ids = decoder.sample(image_feature, max_len_sample, start_idx, end_idx)
    
    assert isinstance(sampled_ids, list)
    assert all(isinstance(idx, int) for idx in sampled_ids)
    # Check if it stopped early (< max_len) and ended with END token OR reached max_len
    if sampled_ids:
        assert len(sampled_ids) <= max_len_sample
        if len(sampled_ids) < max_len_sample:
            assert sampled_ids[-1] == end_idx