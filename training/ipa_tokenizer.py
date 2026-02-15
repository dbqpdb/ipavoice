"""IPA phoneme-level tokenizer with Unicode-aware segmentation.

Segments IPA strings into phoneme tokens, handling combining diacritics,
tie bars (affricates), and modifier letters correctly. Builds a vocabulary
mapping from the full corpus.
"""

from __future__ import annotations

import json
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any

import regex


# --- Special tokens ---

PAD: str = "<pad>"
SOS: str = "<sos>"
EOS: str = "<eos>"
UNK: str = "<unk>"
BLANK: str = "<blank>"
SIL: str = "<sil>"
SPECIAL_TOKENS: list[str] = [PAD, SOS, EOS, UNK, BLANK, SIL]


# --- Normalization maps ---

# Common ASCII substitutions found in IPA transcriptions
_NORMALIZE_MAP: dict[str, str] = {
    "g": "\u0261",      # ASCII g → IPA ɡ
    ":": "\u02D0",      # ASCII colon → IPA length mark ː
    "'": "\u02C8",      # ASCII apostrophe → IPA primary stress ˈ
    "\u035C": "\u0361", # tie bar below → tie bar above (unify)
}

# Characters to strip (transcription conventions, not phonetic content)
_STRIP_CHARS: str = "[]/()"


# --- Regex for phoneme segmentation ---

# Requires the `regex` library for Unicode property escapes (\p{L}, \p{M}).
#
# Token grouping rules:
#   1. Base char + combining marks + (tie bar + base char + combining marks)* + modifiers
#   2. Stress marks (ˈ ˌ) as standalone tokens
#   3. Length marks (ː ˑ) as standalone tokens
#   4. Syllable boundary (.) as standalone token
#   5. Orphaned combining marks (error recovery)
#   6. Whitespace (skipped)
#   7. Any remaining character as fallback

_IPA_TOKEN_RE: regex.Pattern[str] = regex.compile(r"""
    (?:
        [\p{L}\p{So}\p{Nd}]              # base character (letter, symbol, or digit)
        (?:[\p{M}--[\u0361\u035C]])*      # combining marks EXCLUDING tie bars
        (?:
            [\u0361\u035C]                # tie bar (above or below)
            [\p{L}\p{So}]                 # second base character
            (?:[\p{M}--[\u0361\u035C]])*  # its combining marks
        )*
        [\u02B0-\u02C7\u02C9-\u02CB\u02CD-\u02CF\u02D2-\u02FF]*
                                          # trailing modifiers (ʰ ʷ ʲ) excl. stress/length
    )
    | [\u02C8\u02CC]             # stress marks ˈ ˌ
    | [\u02D0\u02D1]             # length marks ː ˑ
    | [.]                         # syllable boundary
    | \p{M}+                      # orphaned combining marks (error case)
    | \s+                         # whitespace (will be filtered)
    | .                           # fallback single character
""", regex.VERBOSE | regex.V1)


def normalize_ipa(text: str) -> str:
    """Normalize an IPA string for consistent tokenization.

    1. Strip transcription brackets/slashes
    2. NFD decomposition (splits precomposed chars like ã → a + ̃)
    3. Apply character-level normalization (ASCII→IPA substitutions)
    4. Strip leading/trailing whitespace
    """
    # Strip transcription convention characters
    for ch in _STRIP_CHARS:
        text = text.replace(ch, "")

    # NFD normalization: decompose precomposed characters
    text = unicodedata.normalize("NFD", text)

    # Apply character-level normalizations
    result: list[str] = []
    for ch in text:
        result.append(_NORMALIZE_MAP.get(ch, ch))

    return "".join(result).strip()


def tokenize(text: str, *, normalize: bool = True) -> list[str]:
    """Segment an IPA string into phoneme-level tokens.

    Args:
        text: IPA transcription string.
        normalize: Whether to apply IPA normalization first.

    Returns:
        List of phoneme tokens (whitespace removed).

    Examples:
        >>> tokenize("t͡ʃʰ")
        ['t͡ʃʰ']
        >>> tokenize("ˈba.na.na")
        ['ˈ', 'b', 'a', '.', 'n', 'a', '.', 'n', 'a']
        >>> tokenize("d͡ʒ")
        ['d͡ʒ']
    """
    if normalize:
        text = normalize_ipa(text)

    if not text:
        return []

    tokens: list[str] = []
    for match in _IPA_TOKEN_RE.finditer(text):
        token: str = match.group()
        # Skip whitespace tokens
        if not token.strip():
            continue
        tokens.append(token)

    return tokens


class IPAVocabulary:
    """IPA phoneme vocabulary with token↔ID mapping.

    Builds from a corpus of IPA strings, filtering by minimum frequency.
    Supports save/load to JSON.
    """

    def __init__(
        self,
        tokens: list[str],
        language_codes: list[str] | None = None,
    ) -> None:
        """Initialize vocabulary from ordered token list.

        Args:
            tokens: Phoneme tokens (excluding special tokens, which are prepended).
            language_codes: Optional language codes to include as conditioning tokens.
        """
        self.special_tokens: list[str] = list(SPECIAL_TOKENS)
        self.phoneme_tokens: list[str] = list(tokens)
        self.language_tokens: list[str] = sorted(language_codes) if language_codes else []

        # Build combined token list: special + phonemes + languages
        all_tokens: list[str] = self.special_tokens + self.phoneme_tokens + self.language_tokens
        self.token_to_id: dict[str, int] = {t: i for i, t in enumerate(all_tokens)}
        self.id_to_token: dict[int, str] = {i: t for i, t in enumerate(all_tokens)}

    @property
    def pad_id(self) -> int:
        return self.token_to_id[PAD]

    @property
    def sos_id(self) -> int:
        return self.token_to_id[SOS]

    @property
    def eos_id(self) -> int:
        return self.token_to_id[EOS]

    @property
    def unk_id(self) -> int:
        return self.token_to_id[UNK]

    @property
    def blank_id(self) -> int:
        return self.token_to_id[BLANK]

    def __len__(self) -> int:
        return len(self.token_to_id)

    def encode(self, text: str) -> list[int]:
        """Encode an IPA string to a list of token IDs."""
        tokens: list[str] = tokenize(text)
        unk: int = self.unk_id
        return [self.token_to_id.get(t, unk) for t in tokens]

    def decode(self, ids: list[int]) -> list[str]:
        """Decode a list of token IDs back to tokens."""
        unk_tok: str = UNK
        return [self.id_to_token.get(i, unk_tok) for i in ids]

    def save(self, path: str | Path) -> None:
        """Save vocabulary to JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data: dict[str, Any] = {
            "special_tokens": self.special_tokens,
            "phoneme_tokens": self.phoneme_tokens,
            "language_tokens": self.language_tokens,
            "token_to_id": self.token_to_id,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> IPAVocabulary:
        """Load vocabulary from JSON."""
        with open(path, encoding="utf-8") as f:
            data: dict[str, Any] = json.load(f)
        vocab: IPAVocabulary = cls(
            tokens=data["phoneme_tokens"],
            language_codes=data.get("language_tokens"),
        )
        return vocab


def build_vocabulary(
    ipa_strings: list[str],
    language_codes: list[str] | None = None,
    min_count: int = 2,
) -> IPAVocabulary:
    """Build an IPA vocabulary from a corpus of transcriptions.

    Args:
        ipa_strings: All IPA transcription strings from the dataset.
        language_codes: Language codes to include as conditioning tokens.
        min_count: Minimum token frequency to include (filters hapax legomena).

    Returns:
        IPAVocabulary with token↔ID mappings.
    """
    # Count all tokens across the corpus
    counter: Counter[str] = Counter()
    for text in ipa_strings:
        tokens: list[str] = tokenize(text)
        counter.update(tokens)

    # Filter by minimum count and sort by frequency (descending)
    filtered: list[tuple[str, int]] = [
        (token, count) for token, count in counter.most_common()
        if count >= min_count
    ]

    total_tokens: int = sum(counter.values())
    filtered_tokens: list[str] = [t for t, _ in filtered]
    dropped: int = len(counter) - len(filtered_tokens)

    print(f"IPA vocabulary statistics:")
    print(f"  Total tokens in corpus: {total_tokens:,}")
    print(f"  Unique tokens: {len(counter):,}")
    print(f"  Tokens with count >= {min_count}: {len(filtered_tokens):,}")
    print(f"  Dropped (hapax): {dropped:,}")
    if language_codes:
        print(f"  Language conditioning tokens: {len(language_codes):,}")
    print(f"  Final vocabulary size: {len(SPECIAL_TOKENS) + len(filtered_tokens) + (len(language_codes) if language_codes else 0):,}")

    # Show top 20 tokens
    print(f"\n  Top 20 tokens:")
    for token, count in filtered[:20]:
        pct: float = 100.0 * count / total_tokens
        # Show Unicode codepoints for non-ASCII tokens
        if any(ord(c) > 127 for c in token):
            codepoints: str = " ".join(f"U+{ord(c):04X}" for c in token)
            print(f"    {token!r:12s} ({codepoints}): {count:>8,} ({pct:.1f}%)")
        else:
            print(f"    {token!r:12s}: {count:>8,} ({pct:.1f}%)")

    return IPAVocabulary(tokens=filtered_tokens, language_codes=language_codes)
