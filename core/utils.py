import re
from collections import Counter
from typing import List, Dict, Union, Any, Set

# Special tokens - Re-define them here or import from dataset.py if preferred
# For now, defining them here for clarity within this module.
PAD_TOKEN = "<pad>"
START_TOKEN = "<start>"
END_TOKEN = "<end>"
UNKNOWN_TOKEN = "<unk>"


class Vocabulary:
    """Manages the mapping between words and numerical indices."""

    def __init__(self, freq_threshold: int = 5):
        """
        Initializes the vocabulary.

        Args:
            freq_threshold: Words with frequency below this threshold will be mapped to <unk>.
        """
        self.word2idx: Dict[str, int] = {}
        self.idx2word: Dict[int, str] = {}
        self.word_counts: Counter[str] = Counter()
        self.freq_threshold = freq_threshold
        self.idx = 0  # Current index to assign to a new word

        # Add special tokens initially
        self.add_word(PAD_TOKEN, special=True)
        self.add_word(START_TOKEN, special=True)
        self.add_word(END_TOKEN, special=True)
        self.add_word(UNKNOWN_TOKEN, special=True)

    def add_word(self, word: str, special: bool = False) -> None:
        """
        Adds a word to the vocabulary if it's not already present.
        If special is True, the word is added regardless of frequency counts for building.

        Args:
            word: The word to add.
            special: If True, the word is a special token and its count is not incremented.
        """
        if not special:
            self.word_counts[word] += 1

        if word not in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def build_vocabulary(self, sentence_list: List[List[str]]) -> None:
        """
        Builds the vocabulary from a list of tokenized sentences.
        Words below freq_threshold are not added explicitly but will map to <unk>.

        Args:
            sentence_list: A list of sentences, where each sentence is a list of words.
        """
        # First, count all words from the provided sentences
        for sentence in sentence_list:
            for word in sentence:
                self.word_counts[word] += 1
        
        # Re-initialize idx and dictionaries to only include frequent words + special tokens
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

        # Add special tokens first to ensure they have fixed indices (0, 1, 2, 3)
        self.add_word(PAD_TOKEN, special=True)
        self.add_word(START_TOKEN, special=True)
        self.add_word(END_TOKEN, special=True)
        self.add_word(UNKNOWN_TOKEN, special=True)

        # Add words that meet the frequency threshold
        for word, count in self.word_counts.items():
            if count >= self.freq_threshold:
                if word not in self.word2idx: # Should not happen if correctly re-initialized
                    self.word2idx[word] = self.idx
                    self.idx2word[self.idx] = word
                    self.idx += 1
        
        print(f"Vocabulary built. Total unique words (>= {self.freq_threshold} freq): {len(self.word2idx)}")

    def __call__(self, word: str) -> int:
        """Gets the index of a word, defaulting to <unk> if not found."""
        return self.word2idx.get(word, self.word2idx[UNKNOWN_TOKEN])

    def __len__(self) -> int:
        """Returns the total number of unique words in the vocabulary."""
        return len(self.word2idx)

    def get_word(self, index: int) -> Union[str, None]:
        """Gets the word for a given index."""
        return self.idx2word.get(index)


def clean_and_tokenize_text(text: str, token_level: str = 'word') -> List[str]:
    """
    Cleans and tokenizes a given text string.

    Args:
        text: The input string.
        token_level: Granularity of tokenization, currently supports 'word'.
                     Future: 'char', 'subword'.

    Returns:
        A list of tokens.
    """
    if not isinstance(text, str):
        # Handle cases where captions might be non-string (e.g. float if data is malformed)
        # This was observed with some COCO records if not careful.
        text = str(text) 

    text = text.lower()
    # Remove punctuation except for '-' (hyphen) if it's part of a word.
    # This regex keeps hyphenated words, apostrophes for contractions, and alphanumeric.
    # It removes most other punctuation.
    text = re.sub(r'[^a-z0-9\s\'-]', '', text)
    # Standardize whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    if token_level == 'word':
        tokens = text.split(' ')
        return [token for token in tokens if token] # Remove empty strings from split
    # Add char or subword tokenization later if needed
    # elif token_level == 'char':
    #     return list(text)
    else:
        raise ValueError(f"Unsupported token_level: {token_level}")