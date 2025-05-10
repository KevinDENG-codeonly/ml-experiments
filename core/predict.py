import torch
from torchvision import transforms
from PIL import Image
import argparse
from pathlib import Path

# Project-specific imports
from src.models import EncoderCNN, DecoderRNN
from core.dataset import Vocabulary, CocoCaptionsDataset # For Vocabulary class and token constants
from core.utils import START_TOKEN, END_TOKEN, PAD_TOKEN, UNKNOWN_TOKEN # Ensure these are accessible

def load_model_and_vocab(checkpoint_path: Path, device: torch.device) -> tuple:
    """
    Loads the model checkpoint, config, and vocabulary.

    Args:
        checkpoint_path: Path to the .pth model checkpoint file.
        device: Device to load models onto.

    Returns:
        A tuple containing: (encoder, decoder, vocab, cfg, start_token_idx, end_token_idx)
    """
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    cfg = checkpoint['config']
    vocab_size_ckpt = checkpoint['vocab_size']
    # Other saved params like embed_size, hidden_size are in cfg['model']

    # Load vocabulary
    # The vocab_path should be in cfg['dataset']['vocab_path']
    # CocoCaptionsDataset.load_vocabulary expects a class method call
    # We need to ensure that the Vocabulary class itself can be loaded if its structure is complex
    # or that vocab_path directly points to a file that CocoCaptionsDataset.load_vocabulary can handle.
    # The current CocoCaptionsDataset.load_vocabulary loads a JSON with word2idx etc.
    vocab_path_from_cfg = Path(cfg['dataset']['vocab_path'])
    if not vocab_path_from_cfg.exists():
        raise FileNotFoundError(f"Vocabulary file specified in checkpoint config not found: {vocab_path_from_cfg}")
    
    # Use the static method from CocoCaptionsDataset to load the vocab
    # This assumes the vocab was saved by CocoCaptionsDataset.save_vocabulary
    vocab = CocoCaptionsDataset.load_vocabulary(vocab_path_from_cfg)
    
    if len(vocab) != vocab_size_ckpt:
        print(f"Warning: Loaded vocabulary size ({len(vocab)}) does not match checkpoint vocab_size ({vocab_size_ckpt}).")
        # This could be an issue if the vocab file was changed after checkpointing.

    # Initialize models
    encoder = EncoderCNN(
        embed_size=cfg['model']['embed_size'],
        cnn_model_name=cfg['model']['encoder_cnn_model'],
        pretrained=False # For inference, pretrained weights are already part of state_dict if saved
    ).to(device)

    decoder = DecoderRNN(
        embed_size=cfg['model']['embed_size'],
        hidden_size=cfg['model']['hidden_size'],
        vocab_size=len(vocab), # Use loaded vocab's size
        num_layers=cfg['model']['decoder_num_layers'],
        dropout_prob=0 # No dropout during inference
    ).to(device)

    # Load state dictionaries
    # Check if encoder_state_dict exists (might not if trained with cached features and encoder was fixed)
    if 'encoder_state_dict' in checkpoint:
        encoder.load_state_dict(checkpoint['encoder_state_dict'])
    else:
        print("Warning: encoder_state_dict not found in checkpoint. This is expected if model was trained with fixed cached features.")
        # If the encoder was truly fixed and not part of training, its state (e.g. from torchvision pretrained) is still needed.
        # The current EncoderCNN init handles pretrained=True by default if not found in checkpoint state. 
        # If pretrained=False was used for init above, ensure this is the desired behavior.
        # For now, we assume if not in checkpoint, the initial pretrained (if any) or random weights are used,
        # which means it MUST have been trained or loaded if it was part of the original model setup.
        # This implies that for inference, the encoder used for feature extraction must be consistent.
        # The most robust way is to always save encoder_state_dict or ensure it's correctly re-initialized.
        # Let's assume for now if not present, the model was trained with cached features, so encoder state is less critical
        # for the decoder part, but for generating features for a NEW image, a trained encoder is needed.
        # If we always expect to run the encoder on a new image, its state_dict should always be there.
        # The current train.py saves encoder_state_dict unless cache_features is true for epoch checkpoints.
        # best_model.pth always saves it. So this warning might be rare for best_model.

    decoder.load_state_dict(checkpoint['decoder_state_dict'])

    encoder.eval()
    decoder.eval()

    start_token_idx = vocab(START_TOKEN)
    end_token_idx = vocab(END_TOKEN)

    return encoder, decoder, vocab, cfg, start_token_idx, end_token_idx

def preprocess_image(image_path: Path, image_size: int, device: torch.device) -> torch.Tensor:
    """
    Loads and preprocesses an image for inference.

    Args:
        image_path: Path to the input image.
        image_size: Target size for image resizing (e.g., 224).
        device: Device to move the tensor to.

    Returns:
        Preprocessed image tensor.
    """
    if not image_path.exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")
    
    # Standard ImageNet normalization
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        normalize,
    ])
    
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0) # Add batch dimension
    return image_tensor.to(device)

def generate_caption(
    encoder: EncoderCNN, 
    decoder: DecoderRNN, 
    image_tensor: torch.Tensor, 
    vocab: Vocabulary, 
    start_token_idx: int, 
    end_token_idx: int, 
    max_len: int,
    device: torch.device # Added device to ensure tensors are on correct device for decoder.sample
) -> str:
    """
    Generates a caption for a preprocessed image tensor.
    """
    encoder.eval()
    decoder.eval()

    with torch.no_grad():
        # Ensure image_tensor is on the correct device before passing to encoder
        features = encoder(image_tensor.to(device))
        # Ensure features are on the correct device for decoder.sample (it should handle internal device placement)
        sampled_ids = decoder.sample(
            features.to(device), 
            max_len=max_len, 
            start_token_idx=start_token_idx, 
            end_token_idx=end_token_idx
        )

    # Convert word IDs to words
    caption_words = []
    special_token_indices = {vocab(START_TOKEN), vocab(END_TOKEN), vocab(PAD_TOKEN), vocab(UNKNOWN_TOKEN)}
    for word_id in sampled_ids:
        if word_id == end_token_idx:
            break
        if word_id not in special_token_indices:
            word = vocab.get_word(word_id)
            if word:
                caption_words.append(word)
    
    return " ".join(caption_words)

def main():
    parser = argparse.ArgumentParser(description="Generate a caption for an image using a trained model.")
    parser.add_argument("--image_path", type=str, required=True, help="Path to the input image.")
    parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to the trained model checkpoint (.pth file).")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"], help="Device to use for inference (cpu or cuda).")
    args = parser.parse_args()

    img_path = Path(args.image_path)
    ckpt_path = Path(args.checkpoint_path)

    selected_device = torch.device(args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    print(f"Using device: {selected_device}")

    try:
        encoder, decoder, vocab, cfg_loaded, start_idx, end_idx = load_model_and_vocab(ckpt_path, selected_device)
        print("Model and vocabulary loaded successfully.")

        image_tensor = preprocess_image(img_path, cfg_loaded['dataset']['image_size'], selected_device)
        print(f"Image {img_path} preprocessed.")
        
        max_caption_length_pred = cfg_loaded['dataset'].get('max_caption_length', 30) # From config used for training dataset

        caption = generate_caption(
            encoder, decoder, image_tensor, vocab, 
            start_idx, end_idx, max_caption_length_pred, selected_device
        )
        
        print(f"\nImage: {args.image_path}")
        print(f"Generated Caption: {caption}")

    except FileNotFoundError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        import traceback
        traceback.print_exc()