import pytest
from unittest.mock import patch, MagicMock
from core.cache_features import precompute_and_cache_features

@patch("core.cache_features.EncoderCNN")
@patch("core.cache_features.CocoCaptionsDataset")
@patch("core.cache_features.download_and_extract_coco")
@patch("core.cache_features.load_config")
@patch("core.cache_features.DataLoader")
def test_precompute_and_cache_features_success(mock_dataloader, mock_load_config, mock_download, mock_dataset, mock_encoder, tmp_path):
    # Mock config
    mock_load_config.return_value = {
        'general': {'device': 'cpu'},
        'model': {'embed_size': 8, 'encoder_cnn_model': 'resnet50', 'encoder_pretrained': False},
        'dataset': {'image_size': 224, 'vocab_path': str(tmp_path / 'vocab.json'), 'vocab_freq_threshold': 1, 'max_caption_length': 10},
        'feature_caching': {'image_features_dir': str(tmp_path / 'features'), 'feature_cache_batch_size': 2}
    }
    # Mock dataset and dataloader
    mock_download.return_value = (tmp_path, tmp_path, tmp_path)
    mock_dataset.return_value.coco.getImgIds.return_value = [1, 2]
    mock_dataset.return_value.__len__.return_value = 2
    mock_dataloader.return_value = [
        (MagicMock(to=lambda device: MagicMock()), None, [1, 2])
    ]
    # Mock encoder
    mock_encoder.return_value.eval.return_value = None
    mock_encoder.return_value.__call__.return_value = MagicMock(size=lambda: (2, 8), __getitem__=lambda self, idx: MagicMock(cpu=lambda: MagicMock(detach=lambda: MagicMock())))
    # Run
    precompute_and_cache_features()
    # Check that feature directory is created
    assert (tmp_path / 'features' / 'train2014').exists() or (tmp_path / 'features' / 'val2014').exists()

@patch("core.cache_features.EncoderCNN")
@patch("core.cache_features.CocoCaptionsDataset")
@patch("core.cache_features.download_and_extract_coco")
@patch("core.cache_features.load_config")
@patch("core.cache_features.DataLoader")
def test_precompute_and_cache_features_empty_dataset(mock_dataloader, mock_load_config, mock_download, mock_dataset, mock_encoder, tmp_path):
    mock_load_config.return_value = {
        'general': {'device': 'cpu'},
        'model': {'embed_size': 8, 'encoder_cnn_model': 'resnet50', 'encoder_pretrained': False},
        'dataset': {'image_size': 224, 'vocab_path': str(tmp_path / 'vocab.json'), 'vocab_freq_threshold': 1, 'max_caption_length': 10},
        'feature_caching': {'image_features_dir': str(tmp_path / 'features'), 'feature_cache_batch_size': 2}
    }
    mock_download.return_value = (tmp_path, tmp_path, tmp_path)
    mock_dataset.return_value.coco.getImgIds.return_value = []
    mock_dataset.return_value.__len__.return_value = 0
    mock_dataloader.return_value = []
    precompute_and_cache_features()
    # Should not raise, but no features saved
    assert (tmp_path / 'features').exists() 