# Sample documents

Real page images used by the tests, the accuracy report and the README examples.

| File | Content | Source and licence |
|---|---|---|
| `sherlock.png` | 1390×925 typeset book page, 33 lines: opening of *A Scandal in Bohemia* | Text by Arthur Conan Doyle (1891), public domain; page rendering added by the project author. Ground truth: `tests/fixtures/real/sherlock.gt.txt`. |
| `noisy_text.png` | 727×343 italic serif text with coffee-stain noise | Appears to derive from the **NoisyOffice** corpus (S. España-Boquera, J. Pastor-Pellicer, M. J. Castro-Bleda, F. Zamora-Martínez; UCI Machine Learning Repository, [doi:10.24432/C5G31N](https://doi.org/10.24432/C5G31N)), licensed **CC BY 4.0**; also distributed as Kaggle's "Denoising Dirty Documents". Resized. |
| `noisy_text2.png` | 727×343 typewriter text on crumpled paper | Same corpus, licence and attribution as `noisy_text.png`. |

Ground-truth transcriptions for all three live in [`tests/fixtures/real/`](../tests/fixtures/real/).
Add new images only if they may be redistributed (public domain, CC0/CC BY, or your own
work without personal data), and record their source here.
