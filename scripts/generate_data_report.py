#!/usr/bin/env python3
"""Generate a comprehensive training data report documenting IPA coverage and limitations."""

from __future__ import annotations

import json
import sys
import unicodedata
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from training.ipa_tokenizer import tokenize


# --- IPA Sound Classification ---

# Core IPA vowels (monophthongs)
VOWELS_CLOSE = set("i y ɨ ʉ ɯ u ɪ ʏ ʊ".split())
VOWELS_MID = set("e ø ɘ ɵ ɤ o ə ɛ œ ɜ ɞ ʌ ɔ".split())
VOWELS_OPEN = set("æ ɐ a ɶ ɑ ɒ".split())
ALL_VOWELS = VOWELS_CLOSE | VOWELS_MID | VOWELS_OPEN

# Consonant categories
PLOSIVES = set("p b t d ʈ ɖ c ɟ k ɡ q ɢ ʔ".split())
NASALS = set("m ɱ n ɳ ɲ ŋ ɴ".split())
TRILLS = set("ʙ r ʀ".split())
TAPS = set("ⱱ ɾ ɽ".split())
FRICATIVES = set("ɸ β f v θ ð s z ʃ ʒ ʂ ʐ ç ʝ x ɣ χ ʁ ħ ʕ h ɦ".split())
LATERALS = set("l ɭ ʎ ʟ ɬ ɮ".split())
APPROXIMANTS = set("ʋ ɹ ɻ j ɰ w ʍ ɥ".split())
CLICKS = set("ʘ ǀ ǃ ǂ ǁ".split())
IMPLOSIVES = set("ɓ ɗ ʄ ɠ ʛ".split())
EJECTIVES_BASE = set("pʼ tʼ kʼ sʼ".split())  # Common ejective patterns

# Suprasegmentals
STRESS_MARKS = set("ˈ ˌ".split())
LENGTH_MARKS = set("ː ˑ".split())
TONE_MARKS = set("˥ ˦ ˧ ˨ ˩".split())  # Tone letters

# Common diacritics/modifiers
ASPIRATED_MARKER = "ʰ"
LABIALIZED_MARKER = "ʷ"
PALATALIZED_MARKER = "ʲ"
VELARIZED_MARKER = "ˠ"
PHARYNGEALIZED_MARKER = "ˤ"
NASALIZED_MARKER = "̃"  # Combining tilde
EJECTIVE_MARKER = "ʼ"


@dataclass
class SoundCategory:
    name: str
    description: str
    examples: list[str]
    total_count: int = 0
    unique_count: int = 0
    tokens: dict[str, int] = None

    def __post_init__(self):
        if self.tokens is None:
            self.tokens = {}


def get_base_char(token: str) -> str:
    """Extract the base character from a token (strip diacritics)."""
    # NFD decompose and take first char
    decomposed = unicodedata.normalize("NFD", token)
    if decomposed:
        return decomposed[0]
    return token


def classify_token(token: str) -> tuple[str, str]:
    """Classify a token into category and subcategory."""
    base = get_base_char(token)

    # Suprasegmentals
    if token in STRESS_MARKS or base in STRESS_MARKS:
        return ("suprasegmental", "stress")
    if token in LENGTH_MARKS or base in LENGTH_MARKS:
        return ("suprasegmental", "length")
    if any(c in token for c in TONE_MARKS):
        return ("suprasegmental", "tone")
    # Check for combining tone marks
    if any("TONE" in unicodedata.name(c, "") for c in token if ord(c) > 127):
        return ("suprasegmental", "tone")

    # Check for tie bar (affricate/coarticulation)
    if "͡" in token or "͜" in token:
        return ("consonant", "affricate")

    # Vowels
    if base in ALL_VOWELS:
        # Check for nasalization
        if NASALIZED_MARKER in token or any(unicodedata.name(c, "").startswith("COMBINING TILDE") for c in token):
            return ("vowel", "nasalized")
        return ("vowel", "oral")

    # Consonants by manner
    if base in CLICKS:
        return ("consonant", "click")
    if base in IMPLOSIVES:
        return ("consonant", "implosive")
    if EJECTIVE_MARKER in token:
        return ("consonant", "ejective")
    if base in PLOSIVES:
        return ("consonant", "plosive")
    if base in NASALS:
        return ("consonant", "nasal")
    if base in TRILLS:
        return ("consonant", "trill")
    if base in TAPS:
        return ("consonant", "tap/flap")
    if base in FRICATIVES:
        return ("consonant", "fricative")
    if base in LATERALS:
        return ("consonant", "lateral")
    if base in APPROXIMANTS:
        return ("consonant", "approximant")

    # Modifiers alone
    if token in {ASPIRATED_MARKER, LABIALIZED_MARKER, PALATALIZED_MARKER}:
        return ("modifier", "secondary articulation")

    # Punctuation / boundaries
    if token in {".", "-", " "}:
        return ("boundary", "syllable/word")

    # Default: check if it's a letter
    if unicodedata.category(base).startswith("L"):
        return ("consonant", "other")

    return ("other", "unknown")


def generate_report(manifest_path: Path, output_path: Path) -> None:
    """Generate the training data report."""

    print("Loading manifest...")
    with open(manifest_path) as f:
        entries = json.load(f)

    # Gather statistics
    token_counts: Counter[str] = Counter()
    category_tokens: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    language_stats: dict[str, dict] = defaultdict(lambda: {"entries": 0, "tokens": 0, "unique": set()})

    for entry in entries:
        ipa = entry.get("ipa", "")
        lang = entry.get("language_code", "UNK")
        lang_name = entry.get("language", lang)

        tokens = tokenize(ipa)
        language_stats[lang]["entries"] += 1
        language_stats[lang]["tokens"] += len(tokens)
        language_stats[lang]["name"] = lang_name

        for token in tokens:
            token_counts[token] += 1
            language_stats[lang]["unique"].add(token)
            cat, subcat = classify_token(token)
            category_tokens[(cat, subcat)][token] += 1

    total_tokens = sum(token_counts.values())
    unique_tokens = len(token_counts)

    # Build report
    lines = []
    lines.append("# IPA Voice Training Data Report")
    lines.append("")
    lines.append("This document describes the coverage and limitations of the IPA Voice training data,")
    lines.append("derived from the UCLA Phonetics Lab Archive.")
    lines.append("")

    # Overview
    lines.append("## Overview")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")
    lines.append(f"| Total audio entries | {len(entries):,} |")
    lines.append(f"| Total IPA tokens | {total_tokens:,} |")
    lines.append(f"| Unique IPA tokens | {unique_tokens:,} |")
    lines.append(f"| Languages represented | {len(language_stats):,} |")
    lines.append(f"| Avg tokens per entry | {total_tokens / len(entries):.1f} |")
    lines.append("")

    # Coverage summary by major category
    lines.append("## Sound Category Coverage")
    lines.append("")
    lines.append("### Summary")
    lines.append("")
    lines.append("| Category | Subcategory | Unique Sounds | Total Tokens | % of Data | Coverage |")
    lines.append("|----------|-------------|---------------|--------------|-----------|----------|")

    # Sort categories logically
    category_order = [
        ("vowel", "oral"),
        ("vowel", "nasalized"),
        ("consonant", "plosive"),
        ("consonant", "nasal"),
        ("consonant", "fricative"),
        ("consonant", "affricate"),
        ("consonant", "approximant"),
        ("consonant", "lateral"),
        ("consonant", "trill"),
        ("consonant", "tap/flap"),
        ("consonant", "click"),
        ("consonant", "implosive"),
        ("consonant", "ejective"),
        ("consonant", "other"),
        ("suprasegmental", "stress"),
        ("suprasegmental", "length"),
        ("suprasegmental", "tone"),
        ("modifier", "secondary articulation"),
        ("boundary", "syllable/word"),
        ("other", "unknown"),
    ]

    def coverage_rating(total: int) -> str:
        if total >= 10000:
            return "Excellent"
        elif total >= 1000:
            return "Good"
        elif total >= 100:
            return "Limited"
        elif total >= 10:
            return "Poor"
        else:
            return "Minimal"

    for cat, subcat in category_order:
        if (cat, subcat) in category_tokens:
            counts = category_tokens[(cat, subcat)]
            total = sum(counts.values())
            unique = len(counts)
            pct = 100 * total / total_tokens
            rating = coverage_rating(total)
            lines.append(f"| {cat.title()} | {subcat.title()} | {unique} | {total:,} | {pct:.1f}% | {rating} |")

    lines.append("")

    # Detailed breakdown by category
    lines.append("### Detailed Token Inventory")
    lines.append("")

    for cat, subcat in category_order:
        if (cat, subcat) not in category_tokens:
            continue
        counts = category_tokens[(cat, subcat)]
        if not counts:
            continue

        total = sum(counts.values())
        lines.append(f"#### {cat.title()}: {subcat.title()}")
        lines.append("")
        lines.append(f"Total: {total:,} tokens, {len(counts)} unique")
        lines.append("")
        lines.append("| Token | Count | % | Notes |")
        lines.append("|-------|-------|---|-------|")

        for token, count in counts.most_common(30):
            pct = 100 * count / total_tokens
            # Generate notes
            notes = []
            if count < 10:
                notes.append("rare")
            if count < 100:
                notes.append("limited data")
            # Unicode info for non-ASCII
            if any(ord(c) > 127 for c in token):
                codepoints = "+".join(f"{ord(c):04X}" for c in token)
                notes.append(f"U+{codepoints}")
            notes_str = ", ".join(notes) if notes else "—"
            lines.append(f"| {token} | {count:,} | {pct:.2f}% | {notes_str} |")

        if len(counts) > 30:
            remaining = len(counts) - 30
            lines.append(f"| ... | | | +{remaining} more tokens |")
        lines.append("")

    # Rare and problematic tokens
    lines.append("## Limitations and Known Issues")
    lines.append("")

    lines.append("### Underrepresented Sounds")
    lines.append("")
    lines.append("The following sound classes have limited training data and may not synthesize reliably:")
    lines.append("")

    rare_categories = []
    for (cat, subcat), counts in category_tokens.items():
        total = sum(counts.values())
        if total < 1000 and cat not in ("boundary", "other", "modifier"):
            rare_categories.append((cat, subcat, total, len(counts)))

    rare_categories.sort(key=lambda x: x[2])

    lines.append("| Category | Total Tokens | Unique | Recommendation |")
    lines.append("|----------|--------------|--------|----------------|")
    for cat, subcat, total, unique in rare_categories:
        if total < 100:
            rec = "Unlikely to work correctly"
        elif total < 500:
            rec = "May produce inconsistent results"
        else:
            rec = "Usable with caution"
        lines.append(f"| {cat.title()}: {subcat.title()} | {total:,} | {unique} | {rec} |")
    lines.append("")

    # Hapax legomena
    hapax = [t for t, c in token_counts.items() if c == 1]
    rare_10 = [t for t, c in token_counts.items() if c <= 10]

    lines.append("### Rare Tokens")
    lines.append("")
    lines.append(f"- **Hapax legomena** (appear once): {len(hapax)} tokens")
    lines.append(f"- **Very rare** (≤10 occurrences): {len(rare_10)} tokens")
    lines.append("")
    lines.append("These tokens are essentially noise in the training data. The model cannot learn")
    lines.append("reliable representations for sounds it has seen fewer than ~50 times.")
    lines.append("")

    # Language coverage
    lines.append("## Language Coverage")
    lines.append("")
    lines.append("### Top 30 Languages by Data Volume")
    lines.append("")
    lines.append("| Language | Code | Entries | Tokens | Unique Tokens |")
    lines.append("|----------|------|---------|--------|---------------|")

    lang_by_tokens = sorted(language_stats.items(), key=lambda x: -x[1]["tokens"])
    for lang_code, stats in lang_by_tokens[:30]:
        lines.append(f"| {stats['name']} | {lang_code} | {stats['entries']:,} | {stats['tokens']:,} | {len(stats['unique'])} |")
    lines.append("")

    # Languages with minimal data
    minimal_langs = [(code, stats) for code, stats in language_stats.items() if stats["entries"] < 50]
    lines.append(f"### Languages with Minimal Data (<50 entries): {len(minimal_langs)}")
    lines.append("")
    lines.append("These languages have insufficient data for reliable synthesis.")
    lines.append("")

    # Recommendations
    lines.append("## Usage Recommendations")
    lines.append("")
    lines.append("### Well-Supported")
    lines.append("")
    lines.append("The model should handle these reliably:")
    lines.append("")
    lines.append("- Basic vowels: a, i, u, e, o, ə, ɛ, ɔ, ɑ")
    lines.append("- Common consonants: p, t, k, b, d, ɡ, m, n, ŋ, s, z, f, v, h, l, r, w, j")
    lines.append("- Aspiration: pʰ, tʰ, kʰ")
    lines.append("- Length: ː")
    lines.append("- Primary stress: ˈ")
    lines.append("")

    lines.append("### Use with Caution")
    lines.append("")
    lines.append("Limited training data exists for:")
    lines.append("")
    lines.append("- Click consonants (ǀ, ǃ, ǂ, ǁ) — only present in a few languages")
    lines.append("- Implosives (ɓ, ɗ, ɠ)")
    lines.append("- Ejectives (pʼ, tʼ, kʼ, sʼ)")
    lines.append("- Complex tone contours")
    lines.append("- Rare diacritical combinations")
    lines.append("")

    lines.append("### Likely to Fail")
    lines.append("")
    lines.append("- Tokens appearing <50 times in training")
    lines.append("- Novel diacritical combinations not seen in training")
    lines.append("- Languages with <50 training entries")
    lines.append("")

    # Write report
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines))
    print(f"Report written to: {output_path}")

    # Also save structured data
    json_path = output_path.with_suffix(".json")
    structured_data = {
        "overview": {
            "total_entries": len(entries),
            "total_tokens": total_tokens,
            "unique_tokens": unique_tokens,
            "languages": len(language_stats),
        },
        "category_summary": {
            f"{cat}:{subcat}": {
                "total": sum(counts.values()),
                "unique": len(counts),
            }
            for (cat, subcat), counts in category_tokens.items()
        },
        "token_counts": dict(token_counts.most_common()),
        "language_stats": {
            code: {
                "name": stats["name"],
                "entries": stats["entries"],
                "tokens": stats["tokens"],
                "unique_tokens": len(stats["unique"]),
            }
            for code, stats in language_stats.items()
        },
    }
    with open(json_path, "w") as f:
        json.dump(structured_data, f, indent=2, ensure_ascii=False)
    print(f"Structured data written to: {json_path}")


def main() -> None:
    manifest_path = Path("data/manifest.json")
    output_path = Path("docs/TRAINING_DATA_REPORT.md")

    if not manifest_path.exists():
        print(f"Error: {manifest_path} not found", file=sys.stderr)
        sys.exit(1)

    generate_report(manifest_path, output_path)


if __name__ == "__main__":
    main()
