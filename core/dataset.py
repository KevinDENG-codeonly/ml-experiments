import os
import shutil
import zipfile
import requests
import json # Added for loading annotations
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional, Callable, Sequence

from tqdm import tqdm
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from pycocotools.coco import COCO # For COCO dataset

from core.utils import Vocabulary, clean_and_tokenize_text # Import from utils

# --- Configuration ---
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
IMAGE_FEATURES_DIR = PROCESSED_DATA_DIR / "image_features"
VOCAB_FILE = PROCESSED_DATA_DIR / "vocab.json" # For saving/loading vocab

# MS COCO 2014 URLs
COCO_BASE_URL = "http://images.cocodataset.org/"
COCO_ANNOTATIONS_URL = COCO_BASE_URL + "annotations/annotations_trainval2014.zip"
COCO_TRAIN_IMAGES_URL = COCO_BASE_URL + "zips/train2014.zip"
COCO_VAL_IMAGES_URL = COCO_BASE_URL + "zips/val2014.zip"

# Special tokens (ensure these match what Vocabulary class in utils.py expects if it hardcodes them)
# Or, pass these to Vocabulary constructor if it's made more flexible.
# For now, assuming Vocabulary in utils.py uses these exact strings as defaults or is built with them.
START_TOKEN = "<start>"
END_TOKEN = "<end>"
UNKNOWN_TOKEN = "<unk>"
PAD_TOKEN = "<pad>"


def _download_file(url: str, destination_path: Path, description: str, force_download: bool = False) -> None:
    """
    Downloads a file from a URL with a progress bar.

    Args:
        url: URL to download from.
        destination_path: Path to save the downloaded file.
        description: Description for the TQDM progress bar.
        force_download: If True, re-downloads even if file exists.
    """
    if destination_path.exists() and not force_download:
        print(f"{description} already exists at {destination_path}. Skipping download.")
        return

    print(f"Downloading {description} from {url} to {destination_path}...")
    try:
        response = requests.get(url, stream=True, timeout=120)
        response.raise_for_status()
        total_size = int(response.headers.get('content-length', 0))
        block_size = 8192

        with open(destination_path, 'wb') as file, tqdm(
            desc=description, total=total_size, unit='iB',
            unit_scale=True, unit_divisor=1024,
        ) as bar:
            for data in response.iter_content(block_size):
                bar.update(len(data))
                file.write(data)
        if total_size != 0 and bar.n != total_size:
            print(f"ERROR: Downloaded size ({bar.n}) does not match expected size ({total_size}) for {description}.")
        else:
            print(f"{description} downloaded successfully.")
    except requests.exceptions.RequestException as e:
        print(f"Error downloading {description}: {e}")
        if destination_path.exists():
            try:
                destination_path.unlink(missing_ok=True)
            except OSError as oe:
                print(f"Error deleting partially downloaded file {destination_path}: {oe}")
        raise

def download_and_extract_coco(
    force_download: bool = False
) -> Tuple[Path, Path, Path]:
    """
    Downloads and extracts the MS COCO 2014 dataset (images and annotations).

    Returns:
        A tuple containing paths to:
            - annotations directory (e.g., data/raw/annotations)
            - train2014 images directory (e.g., data/raw/train2014)
            - val2014 images directory (e.g., data/raw/val2014)
    """
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    annotations_zip_path = RAW_DATA_DIR / "annotations_trainval2014.zip"
    annotations_dir = RAW_DATA_DIR / "annotations"
    train_images_zip_path = RAW_DATA_DIR / "train2014.zip"
    train_images_dir = RAW_DATA_DIR / "train2014"
    val_images_zip_path = RAW_DATA_DIR / "val2014.zip"
    val_images_dir = RAW_DATA_DIR / "val2014"

    # Download and extract annotations
    _download_file(COCO_ANNOTATIONS_URL, annotations_zip_path, "COCO Annotations", force_download)
    if not annotations_dir.exists() or force_download or not any(annotations_dir.iterdir()):
        print(f"Extracting {annotations_zip_path} to {RAW_DATA_DIR}...")
        with zipfile.ZipFile(annotations_zip_path, 'r') as zip_ref:
            zip_ref.extractall(RAW_DATA_DIR)
        print("Annotations extracted.")
    else:
        print("Annotations directory already exists and is not empty. Skipping extraction.")

    # Download and extract training images
    _download_file(COCO_TRAIN_IMAGES_URL, train_images_zip_path, "COCO Train Images", force_download)
    if not train_images_dir.exists() or force_download or not any(train_images_dir.iterdir()):
        print(f"Extracting {train_images_zip_path} to {RAW_DATA_DIR}...")
        with zipfile.ZipFile(train_images_zip_path, 'r') as zip_ref:
            zip_ref.extractall(RAW_DATA_DIR)
        print("Training images extracted.")
    else:
        print("Train images directory already exists and is not empty. Skipping extraction.")

    # Download and extract validation images
    _download_file(COCO_VAL_IMAGES_URL, val_images_zip_path, "COCO Val Images", force_download)
    if not val_images_dir.exists() or force_download or not any(val_images_dir.iterdir()):
        print(f"Extracting {val_images_zip_path} to {RAW_DATA_DIR}...")
        with zipfile.ZipFile(val_images_zip_path, 'r') as zip_ref:
            zip_ref.extractall(RAW_DATA_DIR)
        print("Validation images extracted.")
    else:
        print("Validation images directory already exists and is not empty. Skipping extraction.")

    # Ensure the expected extracted annotation files exist
    expected_train_ann_file = annotations_dir / "captions_train2014.json"
    expected_val_ann_file = annotations_dir / "captions_val2014.json"
    if not expected_train_ann_file.exists() or not expected_val_ann_file.exists():
        if annotations_zip_path.stem:
            nested_parent_dir = RAW_DATA_DIR / annotations_zip_path.stem
            if nested_parent_dir.is_dir():
                nested_annotations_dir = nested_parent_dir / "annotations"
                if nested_annotations_dir.exists() and (nested_annotations_dir / "captions_train2014.json").exists():
                    print(f"Moving nested annotations from {nested_annotations_dir.parent} to {annotations_dir.parent}")
                    annotations_dir.mkdir(parents=True, exist_ok=True)
                    for item in nested_annotations_dir.iterdir():
                        shutil.move(str(item), str(annotations_dir / item.name))
                    shutil.rmtree(nested_parent_dir)
                    print("Nested annotations moved.")

    if not expected_train_ann_file.exists() or not expected_val_ann_file.exists():
        print(f"Warning: Expected annotation files not found in {annotations_dir} even after checks.")
        if not list(annotations_dir.iterdir()) and annotations_zip_path.exists():
            print(f"Attempting re-extraction of {annotations_zip_path} to {annotations_dir.parent}")
            with zipfile.ZipFile(annotations_zip_path, 'r') as zip_ref:
                zip_ref.extractall(annotations_dir.parent)

    return annotations_dir, train_images_dir, val_images_dir

class CocoCaptionsDataset(Dataset[Tuple[torch.Tensor, torch.Tensor, int, int]]):
    """COCO Captions Dataset compatible with torch.utils.data.Dataset."""

    def __init__(
        self,
        annotations_file: Path,
        img_dir: Optional[Path] = None,
        vocab: Optional[Vocabulary] = None,
        transform: Optional[Callable[[Image.Image], torch.Tensor]] = None,
        freq_threshold: int = 5,
        max_seq_length: int = 25,
        build_vocab_on_init: bool = True,
        vocab_file_path: Path = VOCAB_FILE,
        use_cached_features: bool = False,
        feature_cache_dir: Optional[Path] = None
    ):
        """
        Args:
            annotations_file: Path to the COCO annotations JSON file.
            img_dir: Directory with all images (needed if not use_cached_features).
            vocab: Optional pre-built Vocabulary object.
            transform: Optional transform for images (if not use_cached_features).
            freq_threshold: Frequency threshold for building vocabulary.
            max_seq_length: Maximum sequence length for captions.
            build_vocab_on_init: If True and vocab is None, build vocab from annotations.
            vocab_file_path: Path to save/load the vocabulary.
            use_cached_features: If True, load precomputed features.
            feature_cache_dir: Directory of cached image features.
        """
        self.annotations_file = annotations_file
        self.img_dir = img_dir
        self.coco = COCO(str(annotations_file))
        self.ids = list(self.coco.anns.keys())
        self.transform = transform
        self.max_seq_length = max_seq_length
        self.vocab_file_path = vocab_file_path

        self.use_cached_features = use_cached_features
        self.feature_cache_dir = feature_cache_dir

        if self.use_cached_features:
            if self.feature_cache_dir is None:
                raise ValueError("feature_cache_dir must be provided if use_cached_features is True.")
            if not self.feature_cache_dir.exists():
                 print(f"Warning: Specified feature_cache_dir {self.feature_cache_dir} does not exist.")
        elif self.img_dir is None:
             raise ValueError("img_dir must be provided if use_cached_features is False.")

        if vocab:
            self.vocab = vocab
        elif build_vocab_on_init:
            if self.vocab_file_path.exists():
                print(f"Loading vocabulary from {self.vocab_file_path}")
                self.vocab = self.load_vocabulary(self.vocab_file_path)
            else:
                print("Building vocabulary...")
                self.vocab = self._build_vocabulary(freq_threshold)
                self.save_vocabulary(self.vocab, self.vocab_file_path)
        else:
            # If not building and not provided, create a minimal one for utils.PAD_TOKEN etc.
            # Or raise error if vocab is strictly needed later.
            # For now, assume if build_vocab_on_init is False, vocab must be provided for meaningful use.
            print("Warning: Vocabulary not provided and build_vocab_on_init is False. Ensure vocab is handled externally.")
            # Create a dummy vocab if it's None to prevent errors on self.vocab() calls later for special tokens
            # This might hide issues if the user intended to provide a full vocab.
            self.vocab = Vocabulary(freq_threshold=1) # Minimal vocab with special tokens

    def _build_vocabulary(self, freq_threshold: int) -> Vocabulary:
        """Builds vocabulary from all captions in the dataset."""
        all_captions_tokenized: List[List[str]] = []
        print("Tokenizing captions for vocabulary building...")
        for ann_id in tqdm(self.ids, desc="Tokenizing captions"):
            caption_text = self.coco.anns[ann_id]['caption']
            tokens = clean_and_tokenize_text(caption_text)
            all_captions_tokenized.append(tokens)
        
        vocab = Vocabulary(freq_threshold=freq_threshold)
        # The Vocabulary class in utils.py should ideally take special tokens as args
        # or ensure its defaults match START_TOKEN, END_TOKEN, PAD_TOKEN, UNKNOWN_TOKEN here.
        # For now, we assume they match by string value.
        vocab.build_vocabulary(all_captions_tokenized)
        return vocab

    def save_vocabulary(self, vocab: Vocabulary, file_path: Path) -> None:
        """Saves the vocabulary (word2idx and idx2word) to a JSON file."""
        file_path.parent.mkdir(parents=True, exist_ok=True)
        vocab_data = {
            "word2idx": vocab.word2idx,
            "idx2word": {int(k): v for k, v in vocab.idx2word.items()}, # JSON keys must be str
            "freq_threshold": vocab.freq_threshold,
            "special_tokens": {
                "pad": PAD_TOKEN,
                "start": START_TOKEN,
                "end": END_TOKEN,
                "unk": UNKNOWN_TOKEN
            }
        }
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(vocab_data, f, ensure_ascii=False, indent=4)
        print(f"Vocabulary saved to {file_path}")

    @classmethod
    def load_vocabulary(cls, file_path: Path) -> Vocabulary:
        """Loads a vocabulary from a JSON file."""
        if not file_path.exists():
            raise FileNotFoundError(f"Vocabulary file not found: {file_path}")
        with open(file_path, 'r', encoding='utf-8') as f:
            vocab_data = json.load(f)
        
        # Recreate Vocabulary object
        # Ensure special tokens defined in this file match those used when vocab was saved,
        # or that Vocabulary class handles this robustly.
        loaded_vocab = Vocabulary(freq_threshold=vocab_data["freq_threshold"])
        loaded_vocab.word2idx = vocab_data["word2idx"]
        # Convert JSON string keys back to int for idx2word
        loaded_vocab.idx2word = {int(k): v for k, v in vocab_data["idx2word"].items()}
        loaded_vocab.idx = len(loaded_vocab.word2idx) # Recalculate current index

        # Verify special tokens (optional, but good practice)
        # This assumes Vocabulary adds special tokens in a fixed order if re-initialized.
        # A more robust way is to ensure Vocabulary takes special tokens as args.
        # print(f"Loaded <unk> token: {loaded_vocab.word2idx[UNKNOWN_TOKEN]}")
        print(f"Vocabulary loaded from {file_path}. Size: {len(loaded_vocab)}")
        return loaded_vocab

    def __len__(self) -> int:
        """Returns the total number of image-caption pairs."""
        return len(self.ids)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, int, int]:
        """Returns one data pair (image/feature, caption, caption_length, image_id)."""
        ann_id = self.ids[index]
        caption_text = self.coco.anns[ann_id]['caption']
        img_id = self.coco.anns[ann_id]['image_id']

        tokens = clean_and_tokenize_text(caption_text)
        numericalized_caption: List[int] = [self.vocab(START_TOKEN)]
        numericalized_caption.extend([self.vocab(token) for token in tokens])
        numericalized_caption.append(self.vocab(END_TOKEN))
        
        actual_caption_length = len(numericalized_caption) # Length before padding/truncation

        if actual_caption_length < self.max_seq_length:
            numericalized_caption.extend(
                [self.vocab(PAD_TOKEN)] * (self.max_seq_length - actual_caption_length)
            )
            # actual_caption_length remains the true length of content
        else:
            numericalized_caption = numericalized_caption[:self.max_seq_length]
            if numericalized_caption[-1] != self.vocab(END_TOKEN):
                 numericalized_caption[-1] = self.vocab(END_TOKEN)
            actual_caption_length = self.max_seq_length # After truncation, this is the effective length
        
        caption_tensor = torch.tensor(numericalized_caption, dtype=torch.long)

        if self.use_cached_features:
            if not self.feature_cache_dir: raise RuntimeError("feature_cache_dir not set with use_cached_features.")
            feature_file = self.feature_cache_dir / f"{img_id}.pt"
            try:
                image_or_feature_tensor = torch.load(feature_file, map_location='cpu')
            except FileNotFoundError:
                raise FileNotFoundError(f"Cached feature {feature_file} for img_id {img_id} not found.")
        else:
            if not self.img_dir: raise RuntimeError("img_dir not set and not using cached features.")
            img_info = self.coco.loadImgs(img_id)[0]
            img_file_name = img_info['file_name']
            img_path = self.img_dir / img_file_name
            try:
                image = Image.open(img_path).convert("RGB")
            except FileNotFoundError:
                raise FileNotFoundError(f"Image {img_path} for img_id {img_id} not found.")
            if self.transform:
                image_or_feature_tensor = self.transform(image)
            else:
                image_or_feature_tensor = transforms.ToTensor()(image) # Default if no transform

        return image_or_feature_tensor, caption_tensor, actual_caption_length, img_id

# --- Collate Function ---
def collate_fn_with_padding(
    batch: List[Tuple[torch.Tensor, torch.Tensor, int, int]]
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Collate function to handle padding and sorting for sequences.

    Args:
        batch: A list of tuples, where each tuple is 
               (image_or_feature, caption_tensor, caption_length, img_id).

    Returns:
        A tuple containing:
            - batched_images_or_features: (batch_size, ...)
            - batched_captions: (batch_size, max_seq_length)
            - batched_lengths: (batch_size,) - sorted lengths
            - batched_img_ids: (batch_size,)
    """
    # Sort the batch by caption_length in descending order
    batch.sort(key=lambda x: x[2], reverse=True)

    images_or_features, captions, lengths, img_ids = zip(*batch)

    # Stack images/features and captions (captions are already padded to max_seq_length)
    batched_images_or_features = torch.stack(images_or_features, 0)
    batched_captions = torch.stack(captions, 0)
    
    # Convert lengths to a tensor
    batched_lengths = torch.tensor(lengths, dtype=torch.long)
    
    # Convert img_ids to a tensor (assuming img_ids are integers)
    batched_img_ids = torch.tensor(img_ids, dtype=torch.long)

    return batched_images_or_features, batched_captions, batched_lengths, batched_img_ids
    
    # Future steps to call here for testing:
    # 1. Load annotations
    # 2. Build vocabulary
    # 3. Create PyTorch Dataset instance
    # 4. Test data loading 