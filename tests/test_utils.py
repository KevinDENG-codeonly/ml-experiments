import pytest
from collections import Counter

from core.utils import (
    Vocabulary, 
    clean_and_tokenize_text, 
    PAD_TOKEN, START_TOKEN, END_TOKEN, UNKNOWN_TOKEN
)

# --- Fixtures ---
@pytest.fixture
def sample_sentences() -> list[list[str]]:
    return [
        ["a", "cat", "sat"],
        ["a", "dog", "ran"],
        ["a", "cat", "slept", "soundly"],
        ["the", "quick", "brown", "fox"],
    ]

# --- Vocabulary Tests ---
def test_vocabulary_init():
    """Tests basic initialization and presence of special tokens."""
    vocab = Vocabulary(freq_threshold=1)
    assert len(vocab) == 4 # PAD, START, END, UNK
    assert vocab(PAD_TOKEN) == 0
    assert vocab(START_TOKEN) == 1
    assert vocab(END_TOKEN) == 2
    assert vocab(UNKNOWN_TOKEN) == 3
    assert vocab.get_word(0) == PAD_TOKEN
    assert vocab.get_word(1) == START_TOKEN
    assert vocab.get_word(2) == END_TOKEN
    assert vocab.get_word(3) == UNKNOWN_TOKEN
    assert vocab("nonexistent_word") == vocab(UNKNOWN_TOKEN)

def test_vocabulary_add_word():
    """Tests adding words manually (primarily for special tokens)."""
    vocab = Vocabulary()
    initial_len = len(vocab)
    vocab.add_word("test_word", special=False) # special=False updates counts
    assert len(vocab) == initial_len + 1
    assert "test_word" in vocab.word2idx
    assert vocab.word_counts["test_word"] == 1
    vocab.add_word("test_word", special=False)
    assert len(vocab) == initial_len + 1 # Length shouldn't change
    assert vocab.word_counts["test_word"] == 2 # Count should increase

def test_vocabulary_build_no_threshold(sample_sentences):
    """Tests building vocabulary with frequency threshold 1 (all words included)."""
    vocab = Vocabulary(freq_threshold=1)
    vocab.build_vocabulary(sample_sentences)
    
    expected_words = {PAD_TOKEN, START_TOKEN, END_TOKEN, UNKNOWN_TOKEN, 
                      "a", "cat", "sat", "dog", "ran", "slept", "soundly", 
                      "the", "quick", "brown", "fox"}
    assert len(vocab) == len(expected_words)
    for word in expected_words:
        assert word in vocab.word2idx
    assert vocab("cat") != vocab(UNKNOWN_TOKEN)
    assert vocab("a") != vocab(UNKNOWN_TOKEN)
    # Check counts (example)
    assert vocab.word_counts["a"] == 3
    assert vocab.word_counts["cat"] == 2
    assert vocab.word_counts["dog"] == 1

def test_vocabulary_build_with_threshold(sample_sentences):
    """Tests building vocabulary with a frequency threshold > 1."""
    vocab = Vocabulary(freq_threshold=2)
    vocab.build_vocabulary(sample_sentences)

    # Words appearing only once should map to UNKNOWN_TOKEN
    assert vocab("dog") == vocab(UNKNOWN_TOKEN)
    assert vocab("ran") == vocab(UNKNOWN_TOKEN)
    assert vocab("sat") == vocab(UNKNOWN_TOKEN)
    assert vocab("slept") == vocab(UNKNOWN_TOKEN)
    assert vocab("soundly") == vocab(UNKNOWN_TOKEN)
    assert vocab("the") == vocab(UNKNOWN_TOKEN)
    assert vocab("quick") == vocab(UNKNOWN_TOKEN)
    assert vocab("brown") == vocab(UNKNOWN_TOKEN)
    assert vocab("fox") == vocab(UNKNOWN_TOKEN)

    # Words appearing >= 2 times should be present
    assert vocab("a") != vocab(UNKNOWN_TOKEN)
    assert vocab("cat") != vocab(UNKNOWN_TOKEN)

    # Check indices are contiguous after specials
    assert vocab.word2idx[PAD_TOKEN] == 0
    assert vocab.word2idx[START_TOKEN] == 1
    assert vocab.word2idx[END_TOKEN] == 2
    assert vocab.word2idx[UNKNOWN_TOKEN] == 3
    assert vocab.word2idx["a"] == 4
    assert vocab.word2idx["cat"] == 5
    assert len(vocab) == 6 # 4 special + "a" + "cat"

def test_vocabulary_call_and_get_word(sample_sentences):
    """Tests lookup functions __call__ and get_word."""
    vocab = Vocabulary(freq_threshold=2)
    vocab.build_vocabulary(sample_sentences)

    assert vocab("cat") == 5
    assert vocab.get_word(5) == "cat"
    assert vocab("unknown") == vocab(UNKNOWN_TOKEN) # Index 3
    assert vocab.get_word(3) == UNKNOWN_TOKEN
    assert vocab.get_word(100) is None # Index out of bounds

def test_vocabulary_add_and_lookup():
    vocab = Vocabulary(freq_threshold=1)
    vocab.add_word("hello")
    assert vocab("hello") == vocab.word2idx["hello"]
    assert vocab.get_word(vocab("hello")) == "hello"
    # Unknown word returns <unk> index
    unk_idx = vocab.word2idx[UNKNOWN_TOKEN]
    assert vocab("not_in_vocab") == unk_idx

def test_vocabulary_build_vocabulary():
    vocab = Vocabulary(freq_threshold=2)
    sentences = [["a", "b", "a"], ["b", "c"]]
    vocab.build_vocabulary(sentences)
    # Only 'a' and 'b' should be included (freq >= 2)
    assert "a" in vocab.word2idx
    assert "b" in vocab.word2idx
    assert "c" not in vocab.word2idx
    # Special tokens always present
    for token in [PAD_TOKEN, START_TOKEN, END_TOKEN, UNKNOWN_TOKEN]:
        assert token in vocab.word2idx

def test_vocabulary_len():
    vocab = Vocabulary(freq_threshold=1)
    vocab.add_word("x")
    assert len(vocab) == len(vocab.word2idx)

def test_vocabulary_special_tokens():
    vocab = Vocabulary(freq_threshold=1)
    for token in [PAD_TOKEN, START_TOKEN, END_TOKEN, UNKNOWN_TOKEN]:
        idx = vocab(token)
        assert vocab.get_word(idx) == token

# --- clean_and_tokenize_text Tests ---
@pytest.mark.parametrize(
    "input_text, expected_tokens",
    [
        ("A simple sentence.", ["a", "simple", "sentence"]),
        ("  Extra   whitespace!  ", ["extra", "whitespace"]),
        ("It's got punctuation; let's remove it? Yes! 123 numbers.", ["it's", "got", "punctuation", "let's", "remove", "it", "yes", "123", "numbers"]),
        ("Hyphen-ated words.", ["hyphen-ated", "words"]),
        ("", []),
        ("   ", []),
        ("word", ["word"]),
        (123, ["123"]), # Test non-string input
        ("don't won't can't", ["don't", "won't", "can't"]), # Test apostrophes
    ]
)
def test_clean_and_tokenize_text(input_text, expected_tokens):
    """Tests various scenarios for text cleaning and tokenization."""
    assert clean_and_tokenize_text(input_text) == expected_tokens

def test_clean_and_tokenize_unsupported_level():
    """Tests that unsupported token level raises ValueError."""
    with pytest.raises(ValueError):
        clean_and_tokenize_text("some text", token_level="char") # Assuming 'char' is not implemented 

def test_clean_and_tokenize_text_basic():
    text = "Hello, world! This is a test."
    tokens = clean_and_tokenize_text(text)
    assert tokens == ["hello", "world", "this", "is", "a", "test"]

def test_clean_and_tokenize_text_non_string():
    tokens = clean_and_tokenize_text(12345)
    assert tokens == ["12345"]

def test_clean_and_tokenize_text_hyphen_apostrophe():
    text = "state-of-the-art it's"
    tokens = clean_and_tokenize_text(text)
    assert tokens == ["state-of-the-art", "it's"]

def test_clean_and_tokenize_text_unsupported_token_level():
    with pytest.raises(ValueError):
        clean_and_tokenize_text("test", token_level="char") 