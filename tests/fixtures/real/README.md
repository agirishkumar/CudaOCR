# Real-scan ground truth

Hand transcriptions (by eye, 2026-11-02) of images in [`resources/`](../../../resources/):
`<name>.gt.txt` belongs to `resources/<name>.png`.

| File | Content |
|---|---|
| `noisy_text.gt.txt` | 11 lines, italic serif, coffee-stain background |
| `noisy_text2.gt.txt` | 11 lines, typewriter font, crumpled-paper background |
| `vigorous-before.gt.txt` | one word with speckle noise |
| `sherlock.gt.txt` | 33 lines, clean typeset book page (1390×925): centered headings, drop cap, italic word, curly quotes, em dashes. Public-domain text (A. Conan Doyle, *A Scandal in Bohemia*, 1891); line breaks follow the image |

`sherlock.png` is the most representative page (clean printed document, in scope); the
`noisy_text*` scans stress binarization. Both `noisy_text*` scans are clipped at the right edge mid-word; only characters judged
fully visible are transcribed, so a line's last character may be off by one. Treat CER
differences below ~0.5 points on these pages as noise.

Not transcribed: `sample.png` (a personal document) and `sample.jpg` (a decorative poster —
outside the printed-document scope).
