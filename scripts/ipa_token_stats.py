#!/usr/bin/env python3
"""Tally IPA tokens in the training data to identify coverage and gaps."""

from __future__ import annotations

import json
import sys
import unicodedata
from collections import Counter
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from training.ipa_tokenizer import tokenize, normalize_ipa


def get_ipa_category(token: str) -> str:
    """Categorize an IPA token by phonetic class."""
    # Single character analysis
    if len(token) == 1:
        ch = token
        # Stress marks
        if ch in "ˈˌ":
            return "stress"
        # Length marks
        if ch in "ːˑ":
            return "length"
        # Syllable boundary
        if ch == ".":
            return "boundary"
        # Tones (combining or standalone)
        if unicodedata.category(ch) == "Mn" and "TONE" in unicodedata.name(ch, ""):
            return "tone"
        # Vowels (rough heuristic based on common IPA vowels)
        if ch in "aɐɑæɒeɛəɜɞiɪɨoɔøœuʉʊɯʌyʏ":
            return "vowel"
        # Common consonants
        return "consonant"

    # Multi-character tokens (affricates, modified consonants)
    base = token[0]
    # Check for tie bar (affricate)
    if "͡" in token or "͜" in token:
        return "affricate"
    # Check if base is vowel
    if base in "aɐɑæɒeɛəɜɞiɪɨoɔøœuʉʊɯʌyʏ":
        return "vowel"
    return "consonant"


def main() -> None:
    manifest_path = Path("data/manifest.json")

    if not manifest_path.exists():
        print(f"Error: {manifest_path} not found", file=sys.stderr)
        sys.exit(1)

    print("Loading manifest...")
    with open(manifest_path) as f:
        entries = json.load(f)

    print(f"Loaded {len(entries):,} entries")

    # Count tokens
    token_counts: Counter[str] = Counter()
    category_counts: Counter[str] = Counter()
    language_token_counts: dict[str, Counter[str]] = {}
    entries_by_length: Counter[int] = Counter()

    for entry in entries:
        ipa = entry.get("ipa", "")
        lang = entry.get("language_code", "UNK")

        tokens = tokenize(ipa)
        entries_by_length[len(tokens)] += 1

        if lang not in language_token_counts:
            language_token_counts[lang] = Counter()

        for token in tokens:
            token_counts[token] += 1
            language_token_counts[lang][token] += 1
            category_counts[get_ipa_category(token)] += 1

    total_tokens = sum(token_counts.values())
    unique_tokens = len(token_counts)

    print(f"\n{'='*60}")
    print("IPA TOKEN STATISTICS")
    print(f"{'='*60}")
    print(f"Total tokens:  {total_tokens:>12,}")
    print(f"Unique tokens: {unique_tokens:>12,}")
    print(f"Languages:     {len(language_token_counts):>12,}")

    # Category breakdown
    print(f"\n{'-'*60}")
    print("BY CATEGORY")
    print(f"{'-'*60}")
    for cat, count in sorted(category_counts.items(), key=lambda x: -x[1]):
        pct = 100 * count / total_tokens
        print(f"  {cat:12s}: {count:>10,} ({pct:5.1f}%)")

    # Top tokens overall
    print(f"\n{'-'*60}")
    print("TOP 50 TOKENS")
    print(f"{'-'*60}")
    print(f"{'Token':>8s}  {'Count':>10s}  {'%':>6s}  {'Category':>12s}  Unicode")
    print(f"{'-'*8}  {'-'*10}  {'-'*6}  {'-'*12}  {'-'*20}")
    for token, count in token_counts.most_common(50):
        pct = 100 * count / total_tokens
        cat = get_ipa_category(token)
        codepoints = " ".join(f"U+{ord(c):04X}" for c in token)
        print(f"{token:>8s}  {count:>10,}  {pct:>5.1f}%  {cat:>12s}  {codepoints}")

    # Rare tokens (potential issues)
    print(f"\n{'-'*60}")
    print("RARE TOKENS (count <= 10)")
    print(f"{'-'*60}")
    rare_tokens = [(t, c) for t, c in token_counts.items() if c <= 10]
    rare_tokens.sort(key=lambda x: x[1])
    print(f"Total rare tokens: {len(rare_tokens)}")
    print(f"\n{'Token':>8s}  {'Count':>6s}  Unicode")
    print(f"{'-'*8}  {'-'*6}  {'-'*30}")
    for token, count in rare_tokens[:50]:
        codepoints = " ".join(f"U+{ord(c):04X}" for c in token)
        name_parts = []
        for c in token:
            try:
                name_parts.append(unicodedata.name(c, f"U+{ord(c):04X}"))
            except ValueError:
                name_parts.append(f"U+{ord(c):04X}")
        print(f"{token:>8s}  {count:>6,}  {codepoints}")

    # Hapax legomena (appear only once)
    hapax = [t for t, c in token_counts.items() if c == 1]
    print(f"\nHapax legomena (count=1): {len(hapax)}")

    # Entry length distribution
    print(f"\n{'-'*60}")
    print("ENTRY LENGTH DISTRIBUTION (tokens per entry)")
    print(f"{'-'*60}")
    print(f"{'Length':>8s}  {'Count':>10s}  {'%':>6s}")
    print(f"{'-'*8}  {'-'*10}  {'-'*6}")
    total_entries = sum(entries_by_length.values())
    for length in sorted(entries_by_length.keys())[:20]:
        count = entries_by_length[length]
        pct = 100 * count / total_entries
        print(f"{length:>8d}  {count:>10,}  {pct:>5.1f}%")
    if max(entries_by_length.keys()) > 20:
        longer = sum(c for l, c in entries_by_length.items() if l > 20)
        pct = 100 * longer / total_entries
        print(f"{'21+':>8s}  {longer:>10,}  {pct:>5.1f}%")

    # Languages with most unique tokens
    print(f"\n{'-'*60}")
    print("TOP 20 LANGUAGES BY UNIQUE TOKEN COUNT")
    print(f"{'-'*60}")
    lang_unique = [(lang, len(counts)) for lang, counts in language_token_counts.items()]
    lang_unique.sort(key=lambda x: -x[1])
    print(f"{'Language':>8s}  {'Unique':>8s}  {'Total':>10s}")
    print(f"{'-'*8}  {'-'*8}  {'-'*10}")
    for lang, unique in lang_unique[:20]:
        total = sum(language_token_counts[lang].values())
        print(f"{lang:>8s}  {unique:>8,}  {total:>10,}")

    # Save full token counts to file for further analysis
    output_path = Path("data/ipa_token_counts.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "total_tokens": total_tokens,
                "unique_tokens": unique_tokens,
                "token_counts": dict(token_counts.most_common()),
                "category_counts": dict(category_counts),
                "hapax_count": len(hapax),
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"\nFull token counts saved to: {output_path}")


if __name__ == "__main__":
    main()
