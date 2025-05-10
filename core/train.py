import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms

import random
import numpy as np
import os
from pathlib import Path
import time
import json # For potentially saving training stats

# NLTK for BLEU score
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

# Project-specific imports
from src.config_loader import load_config
from core.dataset import CocoCaptionsDataset, download_and_extract_coco, RAW_DATA_DIR, PROCESSED_DATA_DIR, VOCAB_FILE, PAD_TOKEN, START_TOKEN, END_TOKEN, UNKNOWN_TOKEN, collate_fn_with_padding # Ensure all needed tokens are importable
from core.utils import clean_and_tokenize_text # Explicitly import for use in evaluate_model
from src.models import EncoderCNN, DecoderRNN

def set_seed(seed_value: int):
    """Set seed for reproducibility."""
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)
    # Potentially add torch.backends.cudnn.deterministic = True and torch.backends.cudnn.benchmark = False
    # but they can impact performance.

def evaluate_model(
    encoder: EncoderCNN,
    decoder: DecoderRNN,
    dataloader: DataLoader,
    criterion: nn.Module,
    vocab: "Vocabulary", # Pass the vocabulary object for decoding
    device: torch.device,
    max_caption_len_eval: int, # From config, for decoder.sample
    use_cached_features: bool # New flag
) -> Tuple[float, float]: # Returns (avg_loss, avg_bleu4)
    """Evaluates the model on a given dataset, calculating loss and BLEU-4 score."""
    encoder.eval()  # Set encoder to evaluation mode
    decoder.eval()  # Set decoder to evaluation mode
    total_loss = 0.0
    total_bleu4 = 0.0
    num_samples = 0 # Count total samples for accurate BLEU averaging

    # For BLEU calculation
    # Get actual token indices from vocab, not just string constants
    start_token_idx = vocab(START_TOKEN)
    end_token_idx = vocab(END_TOKEN)
    pad_token_idx = vocab(PAD_TOKEN)
    unk_token_idx = vocab(UNKNOWN_TOKEN)
    special_token_indices = {start_token_idx, end_token_idx, pad_token_idx, unk_token_idx}

    smoothing_function = SmoothingFunction().method1 # Or other methods like method4

    with torch.no_grad(): # Disable gradient calculation
        for image_or_feature_batch, captions_batch, lengths_batch, image_ids_batch in dataloader:
            image_or_feature_batch = image_or_feature_batch.to(device)
            captions_batch = captions_batch.to(device)
            lengths_batch = lengths_batch.to(device) # Ensure lengths are on device if needed later, though pack wants CPU
            # image_ids_batch remain on CPU usually

            if use_cached_features:
                # image_or_feature_batch is already the feature tensor
                features = image_or_feature_batch
            else:
                # image_or_feature_batch is an image tensor, pass through encoder
                features = encoder(image_or_feature_batch)
            
            # Pass lengths to decoder
            outputs = decoder(features, captions_batch, lengths_batch)
            targets = captions_batch[:, 1:]
            loss = criterion(outputs.reshape(-1, decoder.vocab_size), targets.reshape(-1))
            total_loss += loss.item() * image_or_feature_batch.size(0) # Weighted by batch size
            
            # BLEU score calculation
            for i in range(image_or_feature_batch.size(0)):
                img_feature_for_sample = features[i].unsqueeze(0) # (1, embed_size)
                img_id = image_ids_batch[i].item() # Get single image ID

                # Generate hypothesis caption
                sampled_ids = decoder.sample(
                    img_feature_for_sample, 
                    max_len=max_caption_len_eval,
                    start_token_idx=start_token_idx,
                    end_token_idx=end_token_idx
                )
                
                hypothesis_tokens = []
                for word_idx in sampled_ids:
                    if word_idx == end_token_idx:
                        break # Stop at <end> token
                    if word_idx not in special_token_indices:
                         word = vocab.get_word(word_idx)
                         if word: # Should always be true if not special and in vocab
                            hypothesis_tokens.append(word)
                
                # Get reference captions for this image_id
                # dataloader.dataset is the CocoCaptionsDataset instance for validation
                reference_anns = dataloader.dataset.coco.imgToAnns[img_id]
                references_tokens_list = []
                for ann in reference_anns:
                    ref_caption_text = ann['caption']
                    # Use the same cleaning and tokenization as dataset preprocessing for consistency
                    # from core.utils import clean_and_tokenize_text (might need to import if not already)
                    # For now, assume clean_and_tokenize_text is available or use simpler split
                    # We need the vocab to map back, so it should be clean
                    ref_tokens_raw = clean_and_tokenize_text(ref_caption_text) # from core.utils
                    ref_tokens = []
                    for token in ref_tokens_raw:
                        # if vocab(token) not in special_token_indices: # Not strictly needed for refs
                        ref_tokens.append(token)
                    if ref_tokens: # only add if not empty
                        references_tokens_list.append(ref_tokens)
                
                if hypothesis_tokens and references_tokens_list:
                    bleu4 = sentence_bleu(references_tokens_list, hypothesis_tokens, smoothing_function=smoothing_function, weights=(0.25, 0.25, 0.25, 0.25))
                    total_bleu4 += bleu4
            
            num_samples += image_or_feature_batch.size(0)

    avg_loss = total_loss / num_samples if num_samples > 0 else 0
    avg_bleu4 = total_bleu4 / num_samples if num_samples > 0 else 0
    
    return avg_loss, avg_bleu4

def train():
    """Main training function for the image captioning model."""
    # 1. Load Configuration
    # ---------------------
    cfg = load_config() # Loads from configs/config.yaml by default
    print("Configuration loaded successfully.")

    # 2. Setup Device and Seed
    # ------------------------
    if cfg['general']['device'] == 'cuda' and not torch.cuda.is_available():
        print("CUDA specified but not available. Switching to CPU.")
        cfg['general']['device'] = 'cpu'
    device = torch.device(cfg['general']['device'])
    print(f"Using device: {device}")

    set_seed(cfg['general']['random_seed'])
    print(f"Random seed set to: {cfg['general']['random_seed']}")

    # Feature Caching Config
    use_cached_features_config = cfg.get('feature_caching', {}).get('cache_features', False)
    feature_cache_base_dir_config = Path(cfg.get('feature_caching', {}).get('image_features_dir', "data/processed/image_features"))
    if use_cached_features_config:
        print(f"Using pre-cached features from: {feature_cache_base_dir_config}")
    else:
        print("Not using cached features. Images will be processed by CNN encoder during training.")

    # 3. Prepare Dataset and DataLoader
    # ---------------------------------
    print("Preparing dataset and dataloader...")
    
    # Ensure data is downloaded and extracted (can be skipped if already done)
    # This might take a long time if data isn't present
    print("Checking/Downloading COCO dataset...")
    annotations_base_dir, train_img_base_dir, val_img_base_dir = download_and_extract_coco(force_download=False)
    train_annotations_file = annotations_base_dir / "captions_train2014.json"
    train_image_dir = train_img_base_dir
    val_annotations_file = annotations_base_dir / "captions_val2014.json"
    val_image_dir = val_img_base_dir

    if not train_annotations_file.exists() or not train_image_dir.exists():
        print(f"Error: Missing training dataset files.")
        return
    if not val_annotations_file.exists() or not val_image_dir.exists():
        print(f"Error: Missing validation dataset files.")
        # Decide if training should proceed without validation or stop
        # For now, we'll allow it but validation will be skipped.
        # return 

    # Image transformations
    image_size = cfg['dataset']['image_size']
    # Normalize with ImageNet stats, common for pretrained models
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    shared_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        normalize,
    ])
    # Training transform can have augmentations like RandomHorizontalFlip
    train_specific_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(), 
        transforms.ToTensor(),
        normalize,
    ])

    # Create training dataset instance
    # Vocabulary will be built/loaded by CocoCaptionsDataset constructor
    train_dataset = CocoCaptionsDataset(
        annotations_file=train_annotations_file,
        img_dir=train_image_dir,
        transform=train_specific_transform if not use_cached_features_config else None, # Use transform with augmentation for training
        freq_threshold=cfg['dataset']['vocab_freq_threshold'],
        max_seq_length=cfg['dataset']['max_caption_length'],
        build_vocab_on_init=True, # Will load from vocab_file_path if exists
        vocab_file_path=Path(cfg['dataset']['vocab_path']), # Use path from config
        use_cached_features=use_cached_features_config,
        feature_cache_dir=feature_cache_base_dir_config / "train2014" if use_cached_features_config else None
    )
    vocab = train_dataset.vocab # Get the built/loaded vocabulary
    vocab_size = len(vocab)
    pad_idx = vocab(PAD_TOKEN) # Get PAD token index for loss ignore_index
    print(f"Training vocabulary built/loaded. Size: {vocab_size}. Pad index: {pad_idx}")

    # Use collate_fn in DataLoader
    train_dataloader = DataLoader(
        train_dataset, batch_size=cfg['training']['batch_size'], 
        shuffle=True, num_workers=4, pin_memory=True if device.type == 'cuda' else False,
        collate_fn=collate_fn_with_padding # Add collate_fn
    )
    print(f"Training Dataloader created. Batches: {len(train_dataloader)}")

    # Validation Dataset (using the same vocabulary)
    val_dataloader = None
    if val_annotations_file.exists() and (not use_cached_features_config and val_image_dir.exists() or use_cached_features_config):
        val_dataset = CocoCaptionsDataset(
            annotations_file=val_annotations_file,
            img_dir=val_image_dir,
            vocab=vocab, # Crucially use the training vocab
            transform=shared_transform if not use_cached_features_config else None, # No augmentation for validation
            max_seq_length=cfg['dataset']['max_caption_length'],
            build_vocab_on_init=False, # Do not rebuild vocab for val set
            use_cached_features=use_cached_features_config,
            feature_cache_dir=feature_cache_base_dir_config / "val2014" if use_cached_features_config else None
        )
        # Use collate_fn in DataLoader for validation too
        val_dataloader = DataLoader(
            val_dataset, batch_size=cfg['training']['batch_size'], 
            shuffle=False, num_workers=4, pin_memory=True if device.type == 'cuda' else False,
            collate_fn=collate_fn_with_padding # Add collate_fn
        )
        print(f"Validation Dataloader created. Batches: {len(val_dataloader)}")
    else:
        print("Validation files/image dir not found or inconsistent with cache settings, skipping validation setup.")

    # 4. Initialize Models
    # --------------------
    print("Initializing models...")
    encoder = EncoderCNN(
        embed_size=cfg['model']['embed_size'],
        cnn_model_name=cfg['model']['encoder_cnn_model'],
        pretrained=cfg['model']['encoder_pretrained']
    ).to(device)

    decoder = DecoderRNN(
        embed_size=cfg['model']['embed_size'],
        hidden_size=cfg['model']['hidden_size'],
        vocab_size=vocab_size,
        num_layers=cfg['model']['decoder_num_layers'],
        dropout_prob=cfg['model']['decoder_dropout_prob']
    ).to(device)
    print("Models initialized and moved to device.")

    # 5. Define Loss, Optimizer, and Scheduler
    # ---------------------------------------
    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx).to(device)
    print(f"Loss function: CrossEntropyLoss (ignoring index {pad_idx})")

    if use_cached_features_config:
        # Only train decoder if features are pre-cached and fixed
        params_to_optimize = decoder.parameters()
        print("Optimizer will only update DecoderRNN parameters (using cached features).")
    else:
        # Fine-tune decoder and last layers of encoder
        params_to_optimize = list(decoder.parameters()) + list(encoder.linear.parameters()) + list(encoder.bn.parameters())
        print("Optimizer will update DecoderRNN and last layers of EncoderCNN.")

    if cfg['training']['optimizer_type'].lower() == 'adam':
        optimizer = optim.Adam(params_to_optimize, lr=cfg['training']['learning_rate'])
    elif cfg['training']['optimizer_type'].lower() == 'sgd':
        optimizer = optim.SGD(params_to_optimize, lr=cfg['training']['learning_rate'], momentum=0.9) # Example SGD params
    else:
        raise ValueError(f"Unsupported optimizer type: {cfg['training']['optimizer_type']}")
    print(f"Optimizer: {cfg['training']['optimizer_type']}")

    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=cfg['training']['lr_scheduler_step_size'],
        gamma=cfg['training']['lr_scheduler_gamma']
    )
    print("Learning rate scheduler: StepLR")

    # 6. Training Loop
    # ----------------
    print("\nStarting training...")
    num_epochs = cfg['training']['num_epochs']
    log_interval = cfg['training']['log_interval']
    save_every = cfg['training']['save_every_epochs']
    checkpoint_dir = Path(cfg['training']['checkpoint_dir']) / cfg['general']['experiment_name']
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    print(f"Checkpoints will be saved in: {checkpoint_dir}")

    best_val_loss = float('inf')
    training_start_time = time.time()
    for epoch in range(1, num_epochs + 1):
        epoch_start_time = time.time()
        total_train_loss = 0.0
        
        encoder.train() # Set encoder to training mode (if it has layers like Dropout, BatchNorm)
        decoder.train() # Set decoder to training mode

        for i, (image_or_feature_batch, captions_batch, lengths_batch, _) in enumerate(train_dataloader):
            batch_start_time = time.time()
            image_or_feature_batch = image_or_feature_batch.to(device)
            captions_batch = captions_batch.to(device)
            # lengths_batch should stay on CPU for pack_padded_sequence in model
            # lengths_batch = lengths_batch.to(device) # Not needed for model
            optimizer.zero_grad()

            if use_cached_features_config:
                features = image_or_feature_batch
            else:
                features = encoder(image_or_feature_batch)
            
            # Pass lengths to decoder
            outputs = decoder(features, captions_batch, lengths_batch)
            targets = captions_batch[:, 1:]
            loss = criterion(outputs.reshape(-1, vocab_size), targets.reshape(-1))
            
            # Backward pass and optimization
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params_to_optimize, cfg['training']['gradient_clip_value'])
            optimizer.step()

            total_train_loss += loss.item() * image_or_feature_batch.size(0) # Weighted by batch size

            if (i + 1) % log_interval == 0:
                batch_time = time.time() - batch_start_time
                avg_batch_loss = loss.item() # Current batch loss
                print(f"Epoch [{epoch}/{num_epochs}], Step [{i+1}/{len(train_dataloader)}], "
                      f"Train Loss: {avg_batch_loss:.4f}, Batch Time: {batch_time:.2f}s, LR: {scheduler.get_last_lr()[0]:.6f}")
        
        # End of epoch
        avg_train_loss = total_train_loss / len(train_dataloader)
        epoch_time_taken = time.time() - epoch_start_time
        print(f"\nEpoch {epoch} Training Summary:")
        print(f"  Average Train Loss: {avg_train_loss:.4f}, Time: {epoch_time_taken:.2f}s, LR: {scheduler.get_last_lr()[0]:.6f}")

        # Validation step
        if val_dataloader:
            print(f"Epoch {epoch} Validating...")
            val_loss, val_bleu4 = evaluate_model(
                encoder, decoder, val_dataloader, criterion, 
                train_dataset.vocab, # Pass training vocab
                device,
                cfg['dataset']['max_caption_length'], # For decoder.sample
                use_cached_features=use_cached_features_config # Pass flag here
            )
            print(f"Epoch {epoch} Validation Summary: Avg Loss: {val_loss:.4f}, Avg BLEU-4: {val_bleu4:.4f}")

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_path = checkpoint_dir / "best_model.pth"
                torch.save({
                    'epoch': epoch,
                    'encoder_state_dict': encoder.state_dict(),
                    'decoder_state_dict': decoder.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'vocab_size': vocab_size,
                    'embed_size': cfg['model']['embed_size'],
                    'hidden_size': cfg['model']['hidden_size'],
                    'num_layers': cfg['model']['decoder_num_layers'],
                    'config': cfg,
                    'validation_loss': best_val_loss,
                    'bleu4_score': val_bleu4 # Save BLEU score as well
                }, best_model_path)
                print(f"New best model saved to {best_model_path} (Val Loss: {best_val_loss:.4f}, BLEU-4: {val_bleu4:.4f})")
        else:
            print("Skipping validation.")

        # Step the learning rate scheduler
        scheduler.step()

        # Save model checkpoint
        if epoch % save_every == 0:
            checkpoint_path = checkpoint_dir / f"model_epoch_{epoch}.pth"
            save_dict = {
                'epoch': epoch,
                'decoder_state_dict': decoder.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'vocab_size': vocab_size,
                'embed_size': cfg['model']['embed_size'],
                'hidden_size': cfg['model']['hidden_size'],
                'num_layers': cfg['model']['decoder_num_layers'],
                'config': cfg # Save config for reproducibility
            }
            if not use_cached_features_config: # Only save encoder state if it was trained
                save_dict['encoder_state_dict'] = encoder.state_dict()
            torch.save(save_dict, checkpoint_path)
            print(f"Checkpoint saved to {checkpoint_path}")
        print("-"*50)

    total_training_time = time.time() - training_start_time
    print(f"\nTotal training finished in {total_training_time // 3600:.0f}h {(total_training_time % 3600) // 60:.0f}m {total_training_time % 60:.0f}s")
    print("Training complete.")