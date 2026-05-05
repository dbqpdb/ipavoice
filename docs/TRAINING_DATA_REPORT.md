# IPA Voice Training Data Report

This document describes the coverage and limitations of the IPA Voice training data,
derived from the UCLA Phonetics Lab Archive.

## Overview

| Metric | Value |
|--------|-------|
| Total audio entries | 205,408 |
| Total IPA tokens | 1,194,911 |
| Unique IPA tokens | 1,855 |
| Languages represented | 291 |
| Avg tokens per entry | 5.8 |

## Sound Category Coverage

### Summary

| Category | Subcategory | Unique Sounds | Total Tokens | % of Data | Coverage |
|----------|-------------|---------------|--------------|-----------|----------|
| Vowel | Oral | 541 | 496,459 | 41.5% | Excellent |
| Vowel | Nasalized | 135 | 9,889 | 0.8% | Good |
| Consonant | Plosive | 233 | 245,156 | 20.5% | Excellent |
| Consonant | Nasal | 93 | 107,338 | 9.0% | Excellent |
| Consonant | Fricative | 190 | 125,333 | 10.5% | Excellent |
| Consonant | Affricate | 145 | 1,554 | 0.1% | Good |
| Consonant | Approximant | 60 | 28,848 | 2.4% | Excellent |
| Consonant | Lateral | 48 | 41,236 | 3.5% | Excellent |
| Consonant | Trill | 37 | 25,902 | 2.2% | Excellent |
| Consonant | Tap/Flap | 15 | 3,218 | 0.3% | Good |
| Consonant | Click | 22 | 10,761 | 0.9% | Excellent |
| Consonant | Implosive | 10 | 2,334 | 0.2% | Good |
| Consonant | Ejective | 58 | 13,284 | 1.1% | Excellent |
| Consonant | Other | 115 | 12,944 | 1.1% | Excellent |
| Suprasegmental | Stress | 5 | 21,462 | 1.8% | Excellent |
| Suprasegmental | Length | 6 | 27,099 | 2.3% | Excellent |
| Suprasegmental | Tone | 55 | 690 | 0.1% | Limited |
| Modifier | Secondary Articulation | 3 | 8 | 0.0% | Minimal |
| Boundary | Syllable/Word | 2 | 10,201 | 0.9% | Excellent |
| Other | Unknown | 82 | 11,195 | 0.9% | Excellent |

### Detailed Token Inventory

#### Vowel: Oral

Total: 496,459 tokens, 541 unique

| Token | Count | % | Notes |
|-------|-------|---|-------|
| a | 129,901 | 10.87% | — |
| i | 82,426 | 6.90% | — |
| o | 49,963 | 4.18% | — |
| u | 35,676 | 2.99% | — |
| e | 34,952 | 2.93% | — |
| ə | 20,597 | 1.72% | U+0259 |
| á | 18,765 | 1.57% | U+0061+0301 |
| í | 13,720 | 1.15% | U+0069+0301 |
| ɑ | 11,651 | 0.98% | U+0251 |
| à | 7,819 | 0.65% | U+0061+0300 |
| ɛ | 6,803 | 0.57% | U+025B |
| ú | 5,563 | 0.47% | U+0075+0301 |
| é | 5,558 | 0.47% | U+0065+0301 |
| y | 5,409 | 0.45% | — |
| ó | 5,285 | 0.44% | U+006F+0301 |
| ɔ | 5,050 | 0.42% | U+0254 |
| əˀ | 2,838 | 0.24% | U+0259+02C0 |
| ɪ | 2,700 | 0.23% | U+026A |
| ì | 2,576 | 0.22% | U+0069+0300 |
| ɨ | 2,459 | 0.21% | U+0268 |
| æ | 2,373 | 0.20% | U+00E6 |
| è | 2,209 | 0.18% | U+0065+0300 |
| ù | 2,191 | 0.18% | U+0075+0300 |
| ò | 1,725 | 0.14% | U+006F+0300 |
| aʼ | 1,624 | 0.14% | U+0061+02BC |
| ɯ | 1,404 | 0.12% | U+026F |
| ɤ | 1,336 | 0.11% | U+0264 |
| ʌ | 1,222 | 0.10% | U+028C |
| ʊ | 1,173 | 0.10% | U+028A |
| éˀ | 1,122 | 0.09% | U+0065+0301+02C0 |
| ... | | | +511 more tokens |

#### Vowel: Nasalized

Total: 9,889 tokens, 135 unique

| Token | Count | % | Notes |
|-------|-------|---|-------|
| ã | 2,004 | 0.17% | U+0061+0303 |
| ũ | 1,263 | 0.11% | U+0075+0303 |
| ĩ | 1,104 | 0.09% | U+0069+0303 |
| ɛ̃ | 953 | 0.08% | U+025B+0303 |
| ɔ̃ | 556 | 0.05% | U+0254+0303 |
| õ | 496 | 0.04% | U+006F+0303 |
| ɑ̃ | 494 | 0.04% | U+0251+0303 |
| ẽ | 289 | 0.02% | U+0065+0303 |
| ã́ | 263 | 0.02% | U+0061+0303+0301 |
| ã̀ | 229 | 0.02% | U+0061+0303+0300 |
| ə̃ | 219 | 0.02% | U+0259+0303 |
| ĩ̀ | 136 | 0.01% | U+0069+0303+0300 |
| ɯ̃ | 131 | 0.01% | U+026F+0303 |
| ĩ́ | 118 | 0.01% | U+0069+0303+0301 |
| ɔ̃̀ | 110 | 0.01% | U+0254+0303+0300 |
| ɛ̃̀ | 108 | 0.01% | U+025B+0303+0300 |
| ɔ̃́ | 106 | 0.01% | U+0254+0303+0301 |
| ɛ̃́ | 101 | 0.01% | U+025B+0303+0301 |
| ṍ | 96 | 0.01% | limited data, U+006F+0303+0301 |
| ũ̀ | 92 | 0.01% | limited data, U+0075+0303+0300 |
| æ̃ | 87 | 0.01% | limited data, U+00E6+0303 |
| ɪ̃ | 86 | 0.01% | limited data, U+026A+0303 |
| ṹ | 79 | 0.01% | limited data, U+0075+0303+0301 |
| ɛ̃ʰ | 77 | 0.01% | limited data, U+025B+0303+02B0 |
| ẽ́ | 42 | 0.00% | limited data, U+0065+0303+0301 |
| ʌ̃ | 42 | 0.00% | limited data, U+028C+0303 |
| ɐ̃ | 35 | 0.00% | limited data, U+0250+0303 |
| ã̂ | 33 | 0.00% | limited data, U+0061+0303+0302 |
| ʊ̃ | 28 | 0.00% | limited data, U+028A+0303 |
| õ̀ | 26 | 0.00% | limited data, U+006F+0303+0300 |
| ... | | | +105 more tokens |

#### Consonant: Plosive

Total: 245,156 tokens, 233 unique

| Token | Count | % | Notes |
|-------|-------|---|-------|
| t | 50,924 | 4.26% | — |
| k | 41,140 | 3.44% | — |
| p | 29,978 | 2.51% | — |
| b | 25,189 | 2.11% | — |
| ɡ | 18,748 | 1.57% | U+0261 |
| d | 17,991 | 1.51% | — |
| ʔ | 15,062 | 1.26% | U+0294 |
| c | 13,051 | 1.09% | — |
| t̪ | 5,784 | 0.48% | U+0074+032A |
| tʰ | 2,612 | 0.22% | U+0074+02B0 |
| tʲ | 2,531 | 0.21% | U+0074+02B2 |
| kʰ | 2,257 | 0.19% | U+006B+02B0 |
| q | 2,248 | 0.19% | — |
| d̪ | 2,087 | 0.17% | U+0064+032A |
| ʈ | 1,488 | 0.12% | U+0288 |
| kʲ | 1,357 | 0.11% | U+006B+02B2 |
| pʰ | 1,331 | 0.11% | U+0070+02B0 |
| kʷ | 1,083 | 0.09% | U+006B+02B7 |
| ç | 1,057 | 0.09% | U+0063+0327 |
| qʷ | 642 | 0.05% | U+0071+02B7 |
| t̠ | 633 | 0.05% | U+0074+0320 |
| č | 628 | 0.05% | U+0063+030C |
| ɖ | 521 | 0.04% | U+0256 |
| tʲʰ | 387 | 0.03% | U+0074+02B2+02B0 |
| d̠ | 384 | 0.03% | U+0064+0320 |
| pʲ | 371 | 0.03% | U+0070+02B2 |
| t̪ʰ | 334 | 0.03% | U+0074+032A+02B0 |
| tʰʲ | 308 | 0.03% | U+0074+02B0+02B2 |
| ɟ | 303 | 0.03% | U+025F |
| b̥ | 217 | 0.02% | U+0062+0325 |
| ... | | | +203 more tokens |

#### Consonant: Nasal

Total: 107,338 tokens, 93 unique

| Token | Count | % | Notes |
|-------|-------|---|-------|
| n | 49,431 | 4.14% | — |
| m | 37,237 | 3.12% | — |
| ŋ | 7,922 | 0.66% | U+014B |
| ŋ̥ | 2,286 | 0.19% | U+014B+0325 |
| ɲ | 2,260 | 0.19% | U+0272 |
| nˠ | 1,694 | 0.14% | U+006E+02E0 |
| nʲ | 1,532 | 0.13% | U+006E+02B2 |
| n̪ | 844 | 0.07% | U+006E+032A |
| n̩ | 531 | 0.04% | U+006E+0329 |
| ɳ | 349 | 0.03% | U+0273 |
| n̥ | 204 | 0.02% | U+006E+0325 |
| mˀ | 198 | 0.02% | U+006D+02C0 |
| mʲ | 185 | 0.02% | U+006D+02B2 |
| m̩ | 183 | 0.02% | U+006D+0329 |
| ŋʰ | 158 | 0.01% | U+014B+02B0 |
| mʰ | 157 | 0.01% | U+006D+02B0 |
| nˠʰ | 154 | 0.01% | U+006E+02E0+02B0 |
| mʲʰ | 154 | 0.01% | U+006D+02B2+02B0 |
| nʲʰ | 154 | 0.01% | U+006E+02B2+02B0 |
| ǹ | 136 | 0.01% | U+006E+0300 |
| ŋʷ | 109 | 0.01% | U+014B+02B7 |
| ŋʲ | 101 | 0.01% | U+014B+02B2 |
| m̀ | 99 | 0.01% | limited data, U+006D+0300 |
| ḿ | 90 | 0.01% | limited data, U+006D+0301 |
| ñ | 88 | 0.01% | limited data, U+006E+0303 |
| m̥ | 84 | 0.01% | limited data, U+006D+0325 |
| ŋʲʰ | 77 | 0.01% | limited data, U+014B+02B2+02B0 |
| ɴ | 75 | 0.01% | limited data, U+0274 |
| nˀ | 66 | 0.01% | limited data, U+006E+02C0 |
| ṇ | 60 | 0.01% | limited data, U+006E+0323 |
| ... | | | +63 more tokens |

#### Consonant: Fricative

Total: 125,333 tokens, 190 unique

| Token | Count | % | Notes |
|-------|-------|---|-------|
| s | 36,453 | 3.05% | — |
| h | 29,546 | 2.47% | — |
| x | 10,246 | 0.86% | — |
| f | 8,523 | 0.71% | — |
| ʃ | 7,639 | 0.64% | U+0283 |
| v | 6,919 | 0.58% | — |
| z | 6,006 | 0.50% | — |
| ɣ | 3,378 | 0.28% | U+0263 |
| ʒ | 2,688 | 0.22% | U+0292 |
| ðʲ | 1,542 | 0.13% | U+00F0+02B2 |
| ʕ | 1,263 | 0.11% | U+0295 |
| χ | 1,139 | 0.10% | U+03C7 |
| s̪ | 735 | 0.06% | U+0073+032A |
| š | 672 | 0.06% | U+0073+030C |
| sʰ | 650 | 0.05% | U+0073+02B0 |
| β | 604 | 0.05% | U+03B2 |
| ɦ | 562 | 0.05% | U+0266 |
| ʂ | 500 | 0.04% | U+0282 |
| xʷ | 484 | 0.04% | U+0078+02B7 |
| ʁ | 483 | 0.04% | U+0281 |
| χʷ | 475 | 0.04% | U+03C7+02B7 |
| θ | 388 | 0.03% | U+03B8 |
| ð | 369 | 0.03% | U+00F0 |
| ʕʷ | 360 | 0.03% | U+0295+02B7 |
| ž | 318 | 0.03% | U+007A+030C |
| ɣʲ | 245 | 0.02% | U+0263+02B2 |
| ħ | 229 | 0.02% | U+0127 |
| vʲ | 189 | 0.02% | U+0076+02B2 |
| ʃʰ | 173 | 0.01% | U+0283+02B0 |
| fʲ | 172 | 0.01% | U+0066+02B2 |
| ... | | | +160 more tokens |

#### Consonant: Affricate

Total: 1,554 tokens, 145 unique

| Token | Count | % | Notes |
|-------|-------|---|-------|
| k͡p | 257 | 0.02% | U+006B+0361+0070 |
| t͡s | 186 | 0.02% | U+0074+0361+0073 |
| ɡ͡b | 138 | 0.01% | U+0261+0361+0062 |
| t͡ʃ | 92 | 0.01% | limited data, U+0074+0361+0283 |
| b͡d | 79 | 0.01% | limited data, U+0062+0361+0064 |
| d͡ʒ | 66 | 0.01% | limited data, U+0064+0361+0292 |
| p͡t | 65 | 0.01% | limited data, U+0070+0361+0074 |
| m͡n | 62 | 0.01% | limited data, U+006D+0361+006E |
| ŋ͡m | 49 | 0.00% | limited data, U+014B+0361+006D |
| o͡u | 37 | 0.00% | limited data, U+006F+0361+0075 |
| t͡ɬʼ | 31 | 0.00% | limited data, U+0074+0361+026C+02BC |
| t͡sʼ | 30 | 0.00% | limited data, U+0074+0361+0073+02BC |
| t͡sʰ | 29 | 0.00% | limited data, U+0074+0361+0073+02B0 |
| d͡z | 25 | 0.00% | limited data, U+0064+0361+007A |
| e͡ɪ | 23 | 0.00% | limited data, U+0065+0361+026A |
| m͡b | 19 | 0.00% | limited data, U+006D+0361+0062 |
| t͡ʃʰ | 15 | 0.00% | limited data, U+0074+0361+0283+02B0 |
| n͡d | 14 | 0.00% | limited data, U+006E+0361+0064 |
| ɡ͡l | 12 | 0.00% | limited data, U+0261+0361+006C |
| k͡p̚ | 11 | 0.00% | limited data, U+006B+0361+0070+031A |
| ŋ͡m̚ | 10 | 0.00% | limited data, U+014B+0361+006D+031A |
| a͡i | 8 | 0.00% | rare, limited data, U+0061+0361+0069 |
| m͡p | 7 | 0.00% | rare, limited data, U+006D+0361+0070 |
| e͡i | 7 | 0.00% | rare, limited data, U+0065+0361+0069 |
| ɡ͡m | 6 | 0.00% | rare, limited data, U+0261+0361+006D |
| h͡w | 6 | 0.00% | rare, limited data, U+0068+0361+0077 |
| x͡w | 6 | 0.00% | rare, limited data, U+0078+0361+0077 |
| t͡ɬ | 6 | 0.00% | rare, limited data, U+0074+0361+026C |
| k͡s | 6 | 0.00% | rare, limited data, U+006B+0361+0073 |
| n͡ɡ | 6 | 0.00% | rare, limited data, U+006E+0361+0261 |
| ... | | | +115 more tokens |

#### Consonant: Approximant

Total: 28,848 tokens, 60 unique

| Token | Count | % | Notes |
|-------|-------|---|-------|
| w | 17,711 | 1.48% | — |
| j | 9,188 | 0.77% | — |
| ɹ | 583 | 0.05% | U+0279 |
| ɻ | 287 | 0.02% | U+027B |
| ɹ̥ | 149 | 0.01% | U+0279+0325 |
| ǰ | 132 | 0.01% | U+006A+030C |
| ʍ | 132 | 0.01% | U+028D |
| w̃ | 114 | 0.01% | U+0077+0303 |
| j̃ | 50 | 0.00% | limited data, U+006A+0303 |
| ɥ | 43 | 0.00% | limited data, U+0265 |
| ʋ | 43 | 0.00% | limited data, U+028B |
| jˀ | 33 | 0.00% | limited data, U+006A+02C0 |
| ẃ | 26 | 0.00% | limited data, U+0077+0301 |
| j̝ | 26 | 0.00% | limited data, U+006A+031D |
| w̄ | 25 | 0.00% | limited data, U+0077+0304 |
| j˔ | 21 | 0.00% | limited data, U+006A+02D4 |
| ɹ̌ | 20 | 0.00% | limited data, U+0279+030C |
| ɹ̩ | 20 | 0.00% | limited data, U+0279+0329 |
| ɻ̥ | 20 | 0.00% | limited data, U+027B+0325 |
| j̥ | 19 | 0.00% | limited data, U+006A+0325 |
| wˤ | 18 | 0.00% | limited data, U+0077+02E4 |
| wʰ | 15 | 0.00% | limited data, U+0077+02B0 |
| j̵ | 13 | 0.00% | limited data, U+006A+0335 |
| w̥ | 12 | 0.00% | limited data, U+0077+0325 |
| ɰ̥ | 12 | 0.00% | limited data, U+0270+0325 |
| ɹ̣ | 12 | 0.00% | limited data, U+0279+0323 |
| w̝ | 12 | 0.00% | limited data, U+0077+031D |
| j̀ | 11 | 0.00% | limited data, U+006A+0300 |
| ɰ | 10 | 0.00% | limited data, U+0270 |
| j̣ | 9 | 0.00% | rare, limited data, U+006A+0323 |
| ... | | | +30 more tokens |

#### Consonant: Lateral

Total: 41,236 tokens, 48 unique

| Token | Count | % | Notes |
|-------|-------|---|-------|
| l | 30,506 | 2.55% | — |
| ɬ | 4,537 | 0.38% | U+026C |
| lˠ | 2,541 | 0.21% | U+006C+02E0 |
| lʲ | 687 | 0.06% | U+006C+02B2 |
| l̪ | 494 | 0.04% | U+006C+032A |
| ɬ̣ | 465 | 0.04% | U+026C+0323 |
| l̥ | 389 | 0.03% | U+006C+0325 |
| ɭ | 245 | 0.02% | U+026D |
| lˀ | 231 | 0.02% | U+006C+02C0 |
| ʎ | 205 | 0.02% | U+028E |
| ḷ | 168 | 0.01% | U+006C+0323 |
| l̥ˠ | 154 | 0.01% | U+006C+0325+02E0 |
| ɮ | 97 | 0.01% | limited data, U+026E |
| l̩ | 77 | 0.01% | limited data, U+006C+0329 |
| l̥ʲ | 77 | 0.01% | limited data, U+006C+0325+02B2 |
| ɬˀ | 66 | 0.01% | limited data, U+026C+02C0 |
| ʎ̥ | 63 | 0.01% | limited data, U+028E+0325 |
| ḻ | 47 | 0.00% | limited data, U+006C+0331 |
| l̠ | 23 | 0.00% | limited data, U+006C+0320 |
| ḽ | 18 | 0.00% | limited data, U+006C+032D |
| l̴ | 14 | 0.00% | limited data, U+006C+0334 |
| l̓ | 14 | 0.00% | limited data, U+006C+0313 |
| ɬʷ | 14 | 0.00% | limited data, U+026C+02B7 |
| l̉ | 12 | 0.00% | limited data, U+006C+0309 |
| l̥ʰ | 10 | 0.00% | limited data, U+006C+0325+02B0 |
| ɬʰ | 10 | 0.00% | limited data, U+026C+02B0 |
| ʟ | 8 | 0.00% | rare, limited data, U+029F |
| l̆ʲ | 8 | 0.00% | rare, limited data, U+006C+0306+02B2 |
| ľ | 7 | 0.00% | rare, limited data, U+006C+030C |
| l̃ | 7 | 0.00% | rare, limited data, U+006C+0303 |
| ... | | | +18 more tokens |

#### Consonant: Trill

Total: 25,902 tokens, 37 unique

| Token | Count | % | Notes |
|-------|-------|---|-------|
| r | 22,750 | 1.90% | — |
| ʙ | 659 | 0.06% | U+0299 |
| rˠ | 616 | 0.05% | U+0072+02E0 |
| r̪ | 533 | 0.04% | U+0072+032A |
| r̥ | 428 | 0.04% | U+0072+0325 |
| ř̥ | 146 | 0.01% | U+0072+0325+030C |
| ʀ | 139 | 0.01% | U+0280 |
| ṛ | 101 | 0.01% | U+0072+0323 |
| r̭ | 84 | 0.01% | limited data, U+0072+032D |
| r̥ʲ | 77 | 0.01% | limited data, U+0072+0325+02B2 |
| r̀ | 63 | 0.01% | limited data, U+0072+0300 |
| rʷ | 39 | 0.00% | limited data, U+0072+02B7 |
| rʲ | 32 | 0.00% | limited data, U+0072+02B2 |
| ř | 29 | 0.00% | limited data, U+0072+030C |
| ʀ̥ | 28 | 0.00% | limited data, U+0280+0325 |
| rʰ | 28 | 0.00% | limited data, U+0072+02B0 |
| r̩ | 23 | 0.00% | limited data, U+0072+0329 |
| r̄ | 14 | 0.00% | limited data, U+0072+0304 |
| r̥ʰ | 12 | 0.00% | limited data, U+0072+0325+02B0 |
| ṛ̤ | 12 | 0.00% | limited data, U+0072+0323+0324 |
| r̃ | 11 | 0.00% | limited data, U+0072+0303 |
| r̯ | 9 | 0.00% | rare, limited data, U+0072+032F |
| r̓ | 8 | 0.00% | rare, limited data, U+0072+0313 |
| r̠ | 8 | 0.00% | rare, limited data, U+0072+0320 |
| ʀ̩ | 8 | 0.00% | rare, limited data, U+0280+0329 |
| ŕ | 7 | 0.00% | rare, limited data, U+0072+0301 |
| r̆ | 6 | 0.00% | rare, limited data, U+0072+0306 |
| ʙ̥ | 6 | 0.00% | rare, limited data, U+0299+0325 |
| ṛʰ | 5 | 0.00% | rare, limited data, U+0072+0323+02B0 |
| ʀ˂ | 5 | 0.00% | rare, limited data, U+0280+02C2 |
| ... | | | +7 more tokens |

#### Consonant: Tap/Flap

Total: 3,218 tokens, 15 unique

| Token | Count | % | Notes |
|-------|-------|---|-------|
| ɾ | 2,179 | 0.18% | U+027E |
| ɽ | 912 | 0.08% | U+027D |
| ɾ̥ | 62 | 0.01% | limited data, U+027E+0325 |
| ɾʲ | 23 | 0.00% | limited data, U+027E+02B2 |
| ɾ̣ | 8 | 0.00% | rare, limited data, U+027E+0323 |
| ɾ̻ | 8 | 0.00% | rare, limited data, U+027E+033B |
| ɾ̩ | 6 | 0.00% | rare, limited data, U+027E+0329 |
| ɽʰ | 6 | 0.00% | rare, limited data, U+027D+02B0 |
| ɾ̃ | 4 | 0.00% | rare, limited data, U+027E+0303 |
| ɾ̩ʰ | 2 | 0.00% | rare, limited data, U+027E+0329+02B0 |
| ɽ̩ʰ | 2 | 0.00% | rare, limited data, U+027D+0329+02B0 |
| ɽ̩ | 2 | 0.00% | rare, limited data, U+027D+0329 |
| ɽ̈ | 2 | 0.00% | rare, limited data, U+027D+0308 |
| ɽ̃ | 1 | 0.00% | rare, limited data, U+027D+0303 |
| ɾ̪ʰ | 1 | 0.00% | rare, limited data, U+027E+032A+02B0 |

#### Consonant: Click

Total: 10,761 tokens, 22 unique

| Token | Count | % | Notes |
|-------|-------|---|-------|
| ǀ | 4,125 | 0.35% | U+01C0 |
| ǃ | 2,119 | 0.18% | U+01C3 |
| ǁ | 1,485 | 0.12% | U+01C1 |
| ǂ | 1,037 | 0.09% | U+01C2 |
| ǃʼ | 550 | 0.05% | U+01C3+02BC |
| ǀʼ | 416 | 0.03% | U+01C0+02BC |
| ǁʼ | 412 | 0.03% | U+01C1+02BC |
| ʘ | 199 | 0.02% | U+0298 |
| ǂʼ | 190 | 0.02% | U+01C2+02BC |
| ǃʰ | 61 | 0.01% | limited data, U+01C3+02B0 |
| ǀʰ | 30 | 0.00% | limited data, U+01C0+02B0 |
| ǁʰ | 24 | 0.00% | limited data, U+01C1+02B0 |
| ǃʷ | 18 | 0.00% | limited data, U+01C3+02B7 |
| ǂˀ | 17 | 0.00% | limited data, U+01C2+02C0 |
| ʘˀ | 13 | 0.00% | limited data, U+0298+02C0 |
| ǀˀ | 13 | 0.00% | limited data, U+01C0+02C0 |
| ǃˀ | 12 | 0.00% | limited data, U+01C3+02C0 |
| ǁʷ | 9 | 0.00% | rare, limited data, U+01C1+02B7 |
| ʘʰ | 8 | 0.00% | rare, limited data, U+0298+02B0 |
| ǂʰ | 8 | 0.00% | rare, limited data, U+01C2+02B0 |
| ǁˀ | 8 | 0.00% | rare, limited data, U+01C1+02C0 |
| ǀʷ | 7 | 0.00% | rare, limited data, U+01C0+02B7 |

#### Consonant: Implosive

Total: 2,334 tokens, 10 unique

| Token | Count | % | Notes |
|-------|-------|---|-------|
| ɓ | 1,477 | 0.12% | U+0253 |
| ɗ | 767 | 0.06% | U+0257 |
| ɓ̥ | 40 | 0.00% | limited data, U+0253+0325 |
| ɠ | 31 | 0.00% | limited data, U+0260 |
| ɗ̪ | 10 | 0.00% | limited data, U+0257+032A |
| ɗ̣ | 4 | 0.00% | rare, limited data, U+0257+0323 |
| ɓ̰ | 2 | 0.00% | rare, limited data, U+0253+0330 |
| ʄ | 1 | 0.00% | rare, limited data, U+0284 |
| ɓ̤ | 1 | 0.00% | rare, limited data, U+0253+0324 |
| ɓʰ | 1 | 0.00% | rare, limited data, U+0253+02B0 |

#### Consonant: Ejective

Total: 13,284 tokens, 58 unique

| Token | Count | % | Notes |
|-------|-------|---|-------|
| kʼ | 2,087 | 0.17% | U+006B+02BC |
| ʼ | 1,478 | 0.12% | U+02BC |
| sʼ | 1,312 | 0.11% | U+0073+02BC |
| pʼ | 989 | 0.08% | U+0070+02BC |
| mʼ | 878 | 0.07% | U+006D+02BC |
| tʼ | 850 | 0.07% | U+0074+02BC |
| ʃʼ | 847 | 0.07% | U+0283+02BC |
| qʼ | 818 | 0.07% | U+0071+02BC |
| t̪ʼ | 729 | 0.06% | U+0074+032A+02BC |
| nʼ | 524 | 0.04% | U+006E+02BC |
| ɬʼ | 439 | 0.04% | U+026C+02BC |
| xʼ | 330 | 0.03% | U+0078+02BC |
| kʷʼ | 267 | 0.02% | U+006B+02B7+02BC |
| qʷʼ | 266 | 0.02% | U+0071+02B7+02BC |
| wʼ | 204 | 0.02% | U+0077+02BC |
| ʎʼ | 173 | 0.01% | U+028E+02BC |
| t̠ʼ | 145 | 0.01% | U+0074+0320+02BC |
| cʼ | 139 | 0.01% | U+0063+02BC |
| čʼ | 112 | 0.01% | U+0063+030C+02BC |
| ʎ̥ʼ | 87 | 0.01% | limited data, U+028E+0325+02BC |
| qʼʷ | 65 | 0.01% | limited data, U+0071+02BC+02B7 |
| kʼʲ | 46 | 0.00% | limited data, U+006B+02BC+02B2 |
| šʼ | 45 | 0.00% | limited data, U+0073+030C+02BC |
| lʼ | 45 | 0.00% | limited data, U+006C+02BC |
| ĉʼ | 45 | 0.00% | limited data, U+0063+0302+02BC |
| č́ʼ | 45 | 0.00% | limited data, U+0063+030C+0301+02BC |
| žʼ | 39 | 0.00% | limited data, U+007A+030C+02BC |
| cʼʷ | 36 | 0.00% | limited data, U+0063+02BC+02B7 |
| hʼ | 30 | 0.00% | limited data, U+0068+02BC |
| zʼ | 27 | 0.00% | limited data, U+007A+02BC |
| ... | | | +28 more tokens |

#### Consonant: Other

Total: 12,944 tokens, 115 unique

| Token | Count | % | Notes |
|-------|-------|---|-------|
| ʡ | 1,604 | 0.13% | U+02A1 |
| N | 1,454 | 0.12% | — |
| ɩ | 1,158 | 0.10% | U+0269 |
| M | 1,017 | 0.09% | — |
| ʜ | 896 | 0.07% | U+029C |
| H | 745 | 0.06% | — |
| ɷ | 691 | 0.06% | U+0277 |
| ˀ | 531 | 0.04% | U+02C0 |
| S | 434 | 0.04% | — |
| ɕ | 395 | 0.03% | U+0255 |
| K | 389 | 0.03% | — |
| I | 322 | 0.03% | — |
| C | 302 | 0.03% | — |
| P | 274 | 0.02% | — |
| A | 249 | 0.02% | — |
| ɫ | 240 | 0.02% | U+026B |
| Q | 205 | 0.02% | — |
| T | 175 | 0.01% | — |
| ɷ̀ | 153 | 0.01% | U+0277+0300 |
| L | 124 | 0.01% | — |
| O | 122 | 0.01% | — |
| F | 118 | 0.01% | — |
| D | 109 | 0.01% | — |
| ɷ́ | 104 | 0.01% | U+0277+0301 |
| ʑ | 92 | 0.01% | limited data, U+0291 |
| ɩ̀ | 89 | 0.01% | limited data, U+0269+0300 |
| Y | 73 | 0.01% | limited data |
| B | 66 | 0.01% | limited data |
| R | 51 | 0.00% | limited data |
| ⁿ | 49 | 0.00% | limited data, U+207F |
| ... | | | +85 more tokens |

#### Suprasegmental: Stress

Total: 21,462 tokens, 5 unique

| Token | Count | % | Notes |
|-------|-------|---|-------|
| ˈ | 21,181 | 1.77% | U+02C8 |
| ˈʱ | 168 | 0.01% | U+02C8+02B1 |
| ˌ | 109 | 0.01% | U+02CC |
| ˈˀ | 3 | 0.00% | rare, limited data, U+02C8+02C0 |
| ˈ̟ | 1 | 0.00% | rare, limited data, U+02C8+031F |

#### Suprasegmental: Length

Total: 27,099 tokens, 6 unique

| Token | Count | % | Notes |
|-------|-------|---|-------|
| ː | 26,954 | 2.26% | U+02D0 |
| ˑ | 113 | 0.01% | U+02D1 |
| ːʰ | 16 | 0.00% | limited data, U+02D0+02B0 |
| ː˔ | 12 | 0.00% | limited data, U+02D0+02D4 |
| ːʼ | 3 | 0.00% | rare, limited data, U+02D0+02BC |
| ːʲ | 1 | 0.00% | rare, limited data, U+02D0+02B2 |

#### Suprasegmental: Tone

Total: 690 tokens, 55 unique

| Token | Count | % | Notes |
|-------|-------|---|-------|
| ˧ | 255 | 0.02% | U+02E7 |
| ˨ | 135 | 0.01% | U+02E8 |
| ˥ | 59 | 0.00% | limited data, U+02E5 |
| ˩ | 53 | 0.00% | limited data, U+02E9 |
| a̠˧ | 46 | 0.00% | limited data, U+0061+0320+02E7 |
| a˧ | 44 | 0.00% | limited data, U+0061+02E7 |
| i˥ | 10 | 0.00% | limited data, U+0069+02E5 |
| i̠˥ | 7 | 0.00% | rare, limited data, U+0069+0320+02E5 |
| e˥ | 5 | 0.00% | rare, limited data, U+0065+02E5 |
| e̠˥ | 5 | 0.00% | rare, limited data, U+0065+0320+02E5 |
| °˥˧ | 4 | 0.00% | rare, limited data, U+00B0+02E5+02E7 |
| i˧ | 3 | 0.00% | rare, limited data, U+0069+02E7 |
| i̠˧ | 3 | 0.00% | rare, limited data, U+0069+0320+02E7 |
| ɯ˥ | 3 | 0.00% | rare, limited data, U+026F+02E5 |
| o˧˩ | 3 | 0.00% | rare, limited data, U+006F+02E7+02E9 |
| o̠˧˩ | 3 | 0.00% | rare, limited data, U+006F+0320+02E7+02E9 |
| o˥ | 3 | 0.00% | rare, limited data, U+006F+02E5 |
| o̠˥ | 3 | 0.00% | rare, limited data, U+006F+0320+02E5 |
| i˨˩˨ | 2 | 0.00% | rare, limited data, U+0069+02E8+02E9+02E8 |
| i˧˩ | 2 | 0.00% | rare, limited data, U+0069+02E7+02E9 |
| i̠˧˩ | 2 | 0.00% | rare, limited data, U+0069+0320+02E7+02E9 |
| u˥ | 2 | 0.00% | rare, limited data, U+0075+02E5 |
| ɯ˧˩ | 2 | 0.00% | rare, limited data, U+026F+02E7+02E9 |
| ɯ̠˥ | 2 | 0.00% | rare, limited data, U+026F+0320+02E5 |
| ɤ˥ | 2 | 0.00% | rare, limited data, U+0264+02E5 |
| ɤ̠˥ | 2 | 0.00% | rare, limited data, U+0264+0320+02E5 |
| o˧ | 2 | 0.00% | rare, limited data, U+006F+02E7 |
| i˩˥ | 1 | 0.00% | rare, limited data, U+0069+02E9+02E5 |
| i˥˩ | 1 | 0.00% | rare, limited data, U+0069+02E5+02E9 |
| i˦˥ | 1 | 0.00% | rare, limited data, U+0069+02E6+02E5 |
| ... | | | +25 more tokens |

#### Modifier: Secondary Articulation

Total: 8 tokens, 3 unique

| Token | Count | % | Notes |
|-------|-------|---|-------|
| ʷ | 4 | 0.00% | rare, limited data, U+02B7 |
| ʲ | 3 | 0.00% | rare, limited data, U+02B2 |
| ʰ | 1 | 0.00% | rare, limited data, U+02B0 |

#### Boundary: Syllable/Word

Total: 10,201 tokens, 2 unique

| Token | Count | % | Notes |
|-------|-------|---|-------|
| - | 6,963 | 0.58% | — |
| . | 3,238 | 0.27% | — |

#### Other: Unknown

Total: 11,195 tokens, 82 unique

| Token | Count | % | Notes |
|-------|-------|---|-------|
| , | 2,130 | 0.18% | — |
| ³ | 1,643 | 0.14% | U+00B3 |
| ? | 1,449 | 0.12% | — |
| ² | 1,264 | 0.11% | U+00B2 |
| ¹ | 936 | 0.08% | U+00B9 |
| ⁴ | 651 | 0.05% | U+2074 |
| ⁵ | 459 | 0.04% | U+2075 |
| ^ | 317 | 0.03% | — |
| ° | 262 | 0.02% | U+00B0 |
| * | 247 | 0.02% | — |
| + | 178 | 0.01% | — |
| ‿ | 176 | 0.01% | U+203F |
|  | 174 | 0.01% | U+F19D |
| · | 119 | 0.01% | U+00B7 |
| 1 | 113 | 0.01% | — |
| ; | 102 | 0.01% | — |
| 5 | 96 | 0.01% | limited data |
| _ | 72 | 0.01% | limited data |
| ‛ | 72 | 0.01% | limited data, U+201B |
| 3 | 58 | 0.00% | limited data |
| ! | 44 | 0.00% | limited data |
| … | 40 | 0.00% | limited data, U+2026 |
|  | 39 | 0.00% | limited data, U+F24A |
| ̍ | 35 | 0.00% | limited data, U+030D |
| 2 | 35 | 0.00% | limited data |
| ⁶ | 35 | 0.00% | limited data, U+2076 |
|  | 34 | 0.00% | limited data, U+F179 |
| ’ | 32 | 0.00% | limited data, U+2019 |
| ̪ | 31 | 0.00% | limited data, U+032A |
| 9 | 30 | 0.00% | limited data |
| ... | | | +52 more tokens |

## Limitations and Known Issues

### Underrepresented Sounds

The following sound classes have limited training data and may not synthesize reliably:

| Category | Total Tokens | Unique | Recommendation |
|----------|--------------|--------|----------------|
| Suprasegmental: Tone | 690 | 55 | Usable with caution |

### Rare Tokens

- **Hapax legomena** (appear once): 345 tokens
- **Very rare** (≤10 occurrences): 1083 tokens

These tokens are essentially noise in the training data. The model cannot learn
reliable representations for sounds it has seen fewer than ~50 times.

## Language Coverage

### Top 30 Languages by Data Volume

| Language | Code | Entries | Tokens | Unique Tokens |
|----------|------|---------|--------|---------------|
| Dahalo | DAL | 14,618 | 104,862 | 84 |
| Chickasaw | CIC | 10,688 | 103,767 | 58 |
| Gaelic, Scottish | GLA | 20,636 | 100,485 | 81 |
| Pirahã | MYP | 8,953 | 71,127 | 45 |
| Yeyi | YEY | 9,629 | 66,020 | 55 |
| Wari | PAV | 10,938 | 64,476 | 32 |
| Montana Salish | FLA | 9,042 | 59,334 | 65 |
| Apache, Western | APW | 6,373 | 48,310 | 110 |
| Toda | TCX | 11,058 | 45,857 | 58 |
| Sandawe | SAD | 3,875 | 22,681 | 88 |
| Juǀ'hoan (Zhu'oasi) | KTZ | 3,889 | 19,264 | 107 |
| Tsou | TSU | 3,190 | 18,385 | 39 |
| Hadza | HTS | 2,552 | 17,634 | 60 |
| Tongan | TON | 2,125 | 12,150 | 46 |
| Banawa | BNH | 2,121 | 11,021 | 21 |
| Zulu | ZUL | 1,429 | 10,955 | 161 |
| Kapampangan | PAM | 1,635 | 10,743 | 115 |
| Czech | CES | 1,695 | 10,734 | 58 |
| Vietnamese | VIE | 2,402 | 10,061 | 226 |
| Icelandic | ISL | 1,726 | 8,341 | 98 |
| Mon | MNW | 2,051 | 8,258 | 44 |
| Navajo | NAV | 949 | 8,110 | 88 |
| Sotho, Southern | SOT | 1,368 | 8,000 | 61 |
| Assamese | ASM | 1,716 | 7,900 | 33 |
| Khana | OGO | 2,641 | 7,878 | 62 |
| Comanche | COM | 747 | 7,775 | 54 |
| !Xóõ | NMN | 1,544 | 7,368 | 149 |
| Ibibio | IBB | 1,747 | 6,754 | 73 |
| Tee | TKQ | 2,277 | 6,693 | 57 |
| Greek | ELL | 1,024 | 6,572 | 99 |

### Languages with Minimal Data (<50 entries): 98

These languages have insufficient data for reliable synthesis.

## Usage Recommendations

### Well-Supported

The model should handle these reliably:

- Basic vowels: a, i, u, e, o, ə, ɛ, ɔ, ɑ
- Common consonants: p, t, k, b, d, ɡ, m, n, ŋ, s, z, f, v, h, l, r, w, j
- Aspiration: pʰ, tʰ, kʰ
- Length: ː
- Primary stress: ˈ

### Use with Caution

Limited training data exists for:

- Click consonants (ǀ, ǃ, ǂ, ǁ) — only present in a few languages
- Implosives (ɓ, ɗ, ɠ)
- Ejectives (pʼ, tʼ, kʼ, sʼ)
- Complex tone contours
- Rare diacritical combinations

### Likely to Fail

- Tokens appearing <50 times in training
- Novel diacritical combinations not seen in training
- Languages with <50 training entries
