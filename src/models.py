import torch
import torch.nn as nn
import torchvision.models as models
from torch.nn.utils.rnn import pack_padded_sequence
from typing import Tuple, Union, List
import torch.nn.utils.rnn as rnn_utils # Use alias for clarity

# --- CNN Encoder ---
class EncoderCNN(nn.Module):
    """Convolutional Neural Network for extracting image features."""

    def __init__(self, embed_size: int, cnn_model_name: str = "resnet50", pretrained: bool = True):
        """
        Initializes the CNN encoder.

        Args:
            embed_size: The desired dimension of the output image embedding.
            cnn_model_name: Name of the torchvision model to use (e.g., "resnet50", "resnet101").
            pretrained: Whether to use a model pretrained on ImageNet.
        """
        super().__init__()
        
        # Load a pretrained CNN (e.g., ResNet-50)
        if cnn_model_name == "resnet50":
            cnn_model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1 if pretrained else None)
        elif cnn_model_name == "resnet101":
            cnn_model = models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V1 if pretrained else None)
        # Add more models as needed, e.g., Inception-v3 or VGG
        # elif cnn_model_name == "inception_v3":
        #     cnn_model = models.inception_v3(weights=models.Inception_V3_Weights.IMAGENET1K_V1 if pretrained else None)
        #     # Inception_v3 has an auxiliary classifier, handle if necessary or use only main path
        #     # For feature extraction, usually only the main path up to avgpool is needed.
        else:
            raise ValueError(f"Unsupported CNN model: {cnn_model_name}. Choose from 'resnet50', 'resnet101', etc.")

        # Remove the original classification layer
        modules = list(cnn_model.children())[:-1]  # All layers except the last (fc)
        self.cnn = nn.Sequential(*modules)
        
        # Add a linear layer to map the CNN output to the desired embedding size
        # The input feature size depends on the chosen CNN model
        # For ResNet-50/101, it's 2048 (after the adaptive average pooling)
        # For Inception-v3, it's 2048
        # For VGG16, it's 512 * 7 * 7 if input is 224x224, then adaptive pool makes it 512
        if "resnet" in cnn_model_name:
            in_features = cnn_model.fc.in_features # Get from the original fc layer
        elif "inception" in cnn_model_name:
             in_features = cnn_model.fc.in_features
        else:
            # Fallback or more specific handling needed for other architectures
            # For VGG, it might be cnn_model.classifier[0].in_features if using the classifier features
            # Or, if using features before classifier, it would be different.
            # For now, let's stick to ResNet-like feature extraction from avgpool layer.
            # If using a model without a clear fc.in_features (like VGG raw features),
            # this needs to be determined manually based on the output shape of self.cnn
            # For simplicity, we will assume models that have fc.in_features from global avg pooling
            if hasattr(cnn_model, 'fc') and hasattr(cnn_model.fc, 'in_features'):
                 in_features = cnn_model.fc.in_features
            elif hasattr(cnn_model, 'classifier') and isinstance(cnn_model.classifier, nn.Sequential) and len(cnn_model.classifier) > 0 and hasattr(cnn_model.classifier[0], 'in_features'):
                # Common for VGG style models
                in_features = cnn_model.classifier[0].in_features
            else:
                # This is a fallback, you might need to determine this manually by inspecting the model
                # or running a dummy input through self.cnn and checking its output shape.
                print(f"Warning: Could not automatically determine in_features for {cnn_model_name}. Assuming 2048.")
                in_features = 2048 

        self.linear = nn.Linear(in_features, embed_size)
        self.bn = nn.BatchNorm1d(embed_size, momentum=0.01) # Batch norm for stability

        # Freeze CNN layers if desired (common practice, but can be fine-tuned)
        # for param in self.cnn.parameters():
        #     param.requires_grad = False

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Extracts feature vectors from input images.

        Args:
            images: A batch of images (batch_size, 3, H, W).

        Returns:
            Image features (batch_size, embed_size).
        """
        features = self.cnn(images) # (batch_size, num_features_cnn, 1, 1) for ResNet-like models
        features = features.view(features.size(0), -1) # Flatten: (batch_size, num_features_cnn)
        
        # Map to embed_size
        embedded_features = self.linear(features) # (batch_size, embed_size)
        embedded_features = self.bn(embedded_features) # Apply batch normalization
        # Optional: Add an activation like ReLU here if needed, but often linear output is fine for LSTM init
        # embedded_features = nn.ReLU()(embedded_features)
        return embedded_features


# --- LSTM Decoder ---
class DecoderRNN(nn.Module):
    """Recurrent Neural Network (LSTM) for generating captions."""

    def __init__(self, embed_size: int, hidden_size: int, vocab_size: int, num_layers: int = 1, dropout_prob: float = 0.5):
        """
        Initializes the RNN decoder.

        Args:
            embed_size: Dimension of word embeddings and image features.
            hidden_size: Dimension of LSTM hidden states.
            vocab_size: The size of the vocabulary (number of unique words).
            num_layers: Number of LSTM layers.
            dropout_prob: Dropout probability for LSTM (if num_layers > 1) and after embedding.
        """
        super().__init__()
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.num_layers = num_layers

        # Word embedding layer
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.embedding_dropout = nn.Dropout(dropout_prob)

        # LSTM layer
        # The image features will be the initial input to the LSTM (like the first word)
        # or used to initialize the hidden state.
        # For Show and Tell, image features initialize hidden/cell states.
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True, dropout=dropout_prob if num_layers > 1 else 0)
        
        # Fully connected layer to map LSTM output to vocabulary scores
        self.linear = nn.Linear(hidden_size, vocab_size)

        # Optional: Initialize weights
        # self._init_weights()

        # Layer to initialize LSTM hidden and cell states from image features
        # The image feature (from EncoderCNN) has `embed_size`
        # We need to map this to (num_layers, batch_size, hidden_size) for h0 and c0
        self.init_h = nn.Linear(embed_size, hidden_size * num_layers) # For hidden state
        self.init_c = nn.Linear(embed_size, hidden_size * num_layers) # For cell state
        # Consider adding activation like Tanh for init_h/c, or ensure LSTM handles range.

    # def _init_weights(self):
    #     """Initializes weights for embedding and linear layers."""
    #     self.embedding.weight.data.uniform_(-0.1, 0.1)
    #     self.linear.weight.data.uniform_(-0.1, 0.1)
    #     self.linear.bias.data.fill_(0)

    def init_hidden_state(self, image_features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Initializes LSTM hidden and cell states from image features.

        Args:
            image_features: Features from the EncoderCNN (batch_size, embed_size).
        
        Returns:
            A tuple (h0, c0) for the LSTM.
            h0: (num_layers, batch_size, hidden_size)
            c0: (num_layers, batch_size, hidden_size)
        """
        batch_size = image_features.size(0)
        
        h0_flat = self.init_h(image_features)
        c0_flat = self.init_c(image_features)
        
        h0 = h0_flat.view(batch_size, self.num_layers, self.hidden_size).permute(1, 0, 2).contiguous()
        c0 = c0_flat.view(batch_size, self.num_layers, self.hidden_size).permute(1, 0, 2).contiguous()
        
        return h0, c0

    def forward(self, image_features: torch.Tensor, captions: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for training. Decodes image features and captions to predict next words.

        Args:
            image_features: Features from the EncoderCNN (batch_size, embed_size).
            captions: Ground truth captions (batch_size, max_caption_length).
                      Assumes captions are numericalized and padded, with <start> token.
            lengths: A tensor of actual lengths for each caption in the batch 
                     (batch_size,) - **before padding, including <start>/<end>**.
                     Lengths MUST be sorted in descending order if enforce_sorted=True.

        Returns:
            Predicted word scores for each time step (batch_size, max_caption_length - 1, vocab_size).
        """
        batch_size = image_features.size(0)

        hiddens, cells = self.init_hidden_state(image_features)

        # Input to LSTM excludes the <end> token, so sequence length is max_len - 1
        # Lengths passed to pack_padded_sequence must correspond to the lengths of the sequences being packed.
        # Since we feed captions[:, :-1] (excluding <end>), the lengths should be original_length - 1.
        # Ensure lengths are on CPU for pack_padded_sequence, as it expects CPU tensor for lengths.
        embeddings = self.embedding(captions[:, :-1]) # (batch_size, max_len-1, embed_size)
        embeddings = self.embedding_dropout(embeddings)
        
        # Adjust lengths for packing (input sequence length is original length - 1)
        # Important: lengths must be > 0
        packed_lengths = lengths - 1
        # Move lengths to CPU as expected by pack_padded_sequence
        packed_lengths_cpu = packed_lengths.cpu()

        # Pack sequence
        # enforce_sorted=False allows flexibility but might be slightly less efficient.
        # If collate_fn guarantees sorted lengths, set enforce_sorted=True.
        packed_embeddings = rnn_utils.pack_padded_sequence(embeddings, packed_lengths_cpu, batch_first=True, enforce_sorted=True) # Assuming collate_fn sorts
        
        lstm_out_packed, _ = self.lstm(packed_embeddings, (hiddens, cells))
        
        # Unpack sequence
        # pad_packed_sequence returns a tuple: (padded_sequence, lengths)
        lstm_out, _ = rnn_utils.pad_packed_sequence(lstm_out_packed, batch_first=True, total_length=captions.size(1) - 1) # Pad back to max_len-1
        # lstm_out shape: (batch_size, max_len-1, hidden_size)

        outputs = self.linear(lstm_out) # (batch_size, max_len-1, vocab_size)
        
        return outputs

    def sample(self, image_features: torch.Tensor, max_len: int = 20, start_token_idx: int = 0, end_token_idx: int = 1) -> List[int]:
        """
        Generates a caption for a single image feature during inference/evaluation.

        Args:
            image_features: Features from the EncoderCNN for a single image (1, embed_size).
            max_len: Maximum length of the generated caption.
            start_token_idx: Index of the <start> token in the vocabulary.
            end_token_idx: Index of the <end> token in the vocabulary.

        Returns:
            A list of word indices representing the generated caption.
        """
        assert image_features.size(0) == 1, "Sample method expects batch size of 1 for image_features"
        
        sampled_ids: List[int] = []
        inputs = self.embedding(torch.tensor([start_token_idx]).to(image_features.device)).unsqueeze(0) # (1, 1, embed_size)
        
        hiddens, cells = self.init_hidden_state(image_features) # (num_layers, 1, hidden_size)

        for _ in range(max_len):
            lstm_out, (hiddens, cells) = self.lstm(inputs, (hiddens, cells)) # lstm_out: (1, 1, hidden_size)
            outputs = self.linear(lstm_out.squeeze(1)) # outputs: (1, vocab_size)
            
            # Greedy decoding: get the word with the highest probability
            predicted_idx = outputs.argmax(1) # (1)
            
            sampled_ids.append(predicted_idx.item())
            
            # If <end> token is predicted, stop
            if predicted_idx.item() == end_token_idx:
                break
            
            # Prepare input for the next time step
            inputs = self.embedding(predicted_idx).unsqueeze(0) # (1, 1, embed_size)
            
        return sampled_ids