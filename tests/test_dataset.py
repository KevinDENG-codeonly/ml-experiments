import pytest
import torch
from pathlib import Path
from unittest.mock import patch, MagicMock
from core.dataset import CocoCaptionsDataset, download_and_extract_coco, collate_fn_with_padding
from core.utils import Vocabulary

@pytest.fixture
def dummy_vocab():
    vocab = Vocabulary(freq_threshold=1)
    vocab.add_word("test")
    return vocab

@pytest.fixture
def dummy_annotations(tmp_path):
    # Create a minimal COCO-style annotation file
    ann = {
        "images": [{"id": 1, "file_name": "img1.jpg"}],
        "annotations": [{"id": 1, "image_id": 1, "caption": "A test caption."}],
    }
    ann_path = tmp_path / "captions_test.json"
    with open(ann_path, "w") as f:
        import json
        json.dump(ann, f)
    return ann_path

@pytest.fixture
def dummy_img_dir(tmp_path):
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    img_path = img_dir / "img1.jpg"
    from PIL import Image
    Image.new("RGB", (224, 224)).save(img_path)
    return img_dir

@patch("core.dataset.COCO")
def test_coco_captions_dataset_init(mock_coco, dummy_annotations, dummy_img_dir, dummy_vocab):
    mock_coco.return_value.anns = {1: {"caption": "A test caption.", "image_id": 1}}
    mock_coco.return_value.imgToAnns = {1: [{"caption": "A test caption."}]}
    mock_coco.return_value.loadImgs.return_value = [{"file_name": "img1.jpg"}]
    dataset = CocoCaptionsDataset(
        annotations_file=dummy_annotations,
        img_dir=dummy_img_dir,
        vocab=dummy_vocab,
        build_vocab_on_init=False
    )
    assert len(dataset) == 1
    img, caption, length, img_id = dataset[0]
    assert isinstance(img, torch.Tensor)
    assert isinstance(caption, torch.Tensor)
    assert isinstance(length, int)
    assert isinstance(img_id, int)

@patch("core.dataset.COCO")
def test_coco_captions_dataset_missing_img(mock_coco, dummy_annotations, dummy_vocab):
    mock_coco.return_value.anns = {1: {"caption": "A test caption.", "image_id": 1}}
    mock_coco.return_value.imgToAnns = {1: [{"caption": "A test caption."}]}
    mock_coco.return_value.loadImgs.return_value = [{"file_name": "missing.jpg"}]
    dataset = CocoCaptionsDataset(
        annotations_file=dummy_annotations,
        img_dir=Path("/nonexistent"),
        vocab=dummy_vocab,
        build_vocab_on_init=False
    )
    with pytest.raises(FileNotFoundError):
        _ = dataset[0]

@patch("core.dataset.COCO")
def test_coco_captions_dataset_cached_features(mock_coco, tmp_path, dummy_annotations, dummy_vocab):
    mock_coco.return_value.anns = {1: {"caption": "A test caption.", "image_id": 1}}
    mock_coco.return_value.imgToAnns = {1: [{"caption": "A test caption."}]}
    dataset = CocoCaptionsDataset(
        annotations_file=dummy_annotations,
        use_cached_features=True,
        feature_cache_dir=tmp_path,
        vocab=dummy_vocab,
        build_vocab_on_init=False
    )
    # Create dummy feature file
    torch.save(torch.randn(128), tmp_path / "1.pt")
    feature, caption, length, img_id = dataset[0]
    assert isinstance(feature, torch.Tensor)

@patch("core.dataset.COCO")
def test_coco_captions_dataset_missing_feature(mock_coco, tmp_path, dummy_annotations, dummy_vocab):
    mock_coco.return_value.anns = {1: {"caption": "A test caption.", "image_id": 1}}
    mock_coco.return_value.imgToAnns = {1: [{"caption": "A test caption."}]}
    dataset = CocoCaptionsDataset(
        annotations_file=dummy_annotations,
        use_cached_features=True,
        feature_cache_dir=tmp_path,
        vocab=dummy_vocab,
        build_vocab_on_init=False
    )
    with pytest.raises(FileNotFoundError):
        _ = dataset[0]

def test_collate_fn_with_padding():
    batch = [
        (torch.ones(3), torch.tensor([1,2,3]), 3, 1),
        (torch.ones(3), torch.tensor([1,2,0]), 2, 2),
    ]
    images, captions, lengths, img_ids = collate_fn_with_padding(batch)
    assert images.shape[0] == 2
    assert captions.shape == (2, 3)
    assert lengths.tolist() == [3, 2]
    assert img_ids.tolist() == [1, 2] 