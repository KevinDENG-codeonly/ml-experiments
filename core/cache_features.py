import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from pathlib import Path
import os
from tqdm import tqdm

# Project-specific imports
from src.config_loader import load_config
from src.models import EncoderCNN
from core.dataset import CocoCaptionsDataset, download_and_extract_coco # For paths and dataset structure

def precompute_and_cache_features():
    """Precomputes image features using the EncoderCNN and saves them to disk."""
    # 1. Load Configuration
    cfg = load_config()
    print("Configuration loaded for feature caching.")

    # 2. Setup Device
    if cfg['general']['device'] == 'cuda' and not torch.cuda.is_available():
        print("CUDA specified but not available. Switching to CPU for feature caching.")
        cfg['general']['device'] = 'cpu'
    device = torch.device(cfg['general']['device'])
    print(f"Using device: {device} for feature caching.")

    # 3. Initialize Encoder Model
    print("Initializing EncoderCNN model...")
    encoder = EncoderCNN(
        embed_size=cfg['model']['embed_size'],
        cnn_model_name=cfg['model']['encoder_cnn_model'],
        pretrained=cfg['model']['encoder_pretrained']
    ).to(device)
    encoder.eval() # Set to evaluation mode
    print("EncoderCNN model initialized.")

    # 4. Define Image Transformations (should match training, minus data augmentation for consistency if features are fixed)
    image_size = cfg['dataset']['image_size']
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    # For caching, we generally don't want random augmentations like RandomHorizontalFlip
    # unless the plan is to cache multiple augmented versions, which is not typical for this setup.
    cache_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        normalize,
    ])
    print("Image transform for caching defined.")

    # 5. Get dataset paths
    print("Fetching dataset paths...")
    annotations_base_dir, train_img_base_dir, val_img_base_dir = download_and_extract_coco(force_download=False)
    
    # Base directory for cached features from config
    base_features_dir = Path(cfg['feature_caching']['image_features_dir'])
    base_features_dir.mkdir(parents=True, exist_ok=True)
    print(f"Base directory for cached features: {base_features_dir}")

    # Batch size for feature caching from config
    cache_batch_size = cfg['feature_caching'].get('feature_cache_batch_size', 128) # Default if not in config

    # Define sets to process: (name, annotation_file, image_dir, output_feature_subdir)
    datasets_to_process = []
    train_annotations_file = annotations_base_dir / "captions_train2014.json"
    val_annotations_file = annotations_base_dir / "captions_val2014.json"

    if train_annotations_file.exists() and train_img_base_dir.exists():
        datasets_to_process.append((
            "train2014",
            train_annotations_file,
            train_img_base_dir,
            base_features_dir / "train2014"
        ))
    else:
        print(f"Skipping training set feature caching: files not found ({train_annotations_file}, {train_img_base_dir})")

    if val_annotations_file.exists() and val_img_base_dir.exists():
        datasets_to_process.append((
            "val2014",
            val_annotations_file,
            val_img_base_dir,
            base_features_dir / "val2014"
        ))
    else:
        print(f"Skipping validation set feature caching: files not found ({val_annotations_file}, {val_img_base_dir})")

    # 6. Process each dataset (train, val)
    for name, ann_file, img_dir, feature_out_dir in datasets_to_process:
        print(f"\nProcessing dataset: {name}")
        print(f"  Annotations: {ann_file}")
        print(f"  Image Dir: {img_dir}")
        print(f"  Output Features Dir: {feature_out_dir}")
        feature_out_dir.mkdir(parents=True, exist_ok=True)

        # We need a Vocabulary object to instantiate CocoCaptionsDataset, even if we don't use captions for caching.
        # It can be a dummy one or the one from training if available.
        # For simplicity, CocoCaptionsDataset will try to build/load one if not provided.
        # We can pass `build_vocab_on_init=False` and `vocab=None` if we modify dataset to allow this for caching.
        # OR, ensure vocab.json exists or let it build a temp one.
        # Let's use the actual vocab path to be safe, so it either loads or builds once.
        vocab_path = Path(cfg['dataset']['vocab_path'])
        
        # CocoCaptionsDataset needs freq_threshold and max_seq_length, get from config
        dataset = CocoCaptionsDataset(
            annotations_file=ann_file,
            img_dir=img_dir,
            transform=cache_transform,
            # Vocab related params, less critical for caching but needed by constructor
            vocab_file_path=vocab_path, 
            freq_threshold=cfg['dataset']['vocab_freq_threshold'],
            max_seq_length=cfg['dataset']['max_caption_length'],
            build_vocab_on_init=True # Allow it to load/build vocab if needed
        )

        # We need to iterate through unique image IDs. CocoCaptionsDataset iterates through annotations.
        # A better approach for caching might be to directly get unique image IDs from coco.getImgIds()
        # and then load each image.
        # For now, we process based on annotation IDs, but save features per image_id, skipping duplicates.
        
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=cache_batch_size,
            shuffle=False, # No need to shuffle for caching
            num_workers=4 # Adjust based on your system
        )

        print(f"Caching features for {len(dataset.coco.getImgIds())} unique images in {name} set...")
        cached_image_ids = set()

        with torch.no_grad():
            for images, _, image_ids_batch in tqdm(dataloader, desc=f"Caching {name} features"):
                images = images.to(device)
                batch_features = encoder(images) # (batch_size, embed_size)
                
                # Save features for each image in the batch
                for i in range(images.size(0)):
                    image_id = image_ids_batch[i].item()
                    if image_id in cached_image_ids:
                        continue # Already cached this image if multiple annotations pointed to it
                    
                    feature_save_path = feature_out_dir / f"{image_id}.pt"
                    # Detach from graph, move to CPU before saving
                    torch.save(batch_features[i].cpu().detach(), feature_save_path)
                    cached_image_ids.add(image_id)
        
        print(f"Finished caching features for {name}. Total unique images cached: {len(cached_image_ids)}")

    print("\nFeature caching process complete.")