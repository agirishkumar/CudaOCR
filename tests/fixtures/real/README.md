# Real-scan ground truth

Hand transcriptions (by eye, 2026-11-02) of the images in [`resources/`](../../../resources/)
(sources and licences: [`resources/README.md`](../../../resources/README.md)):
`<name>.gt.txt` belongs to `resources/<name>.png`.

| File | Content |
|---|---|
| `sherlock.gt.txt` | 33 lines, clean typeset book page: centred headings, drop cap, italic word, curly quotes, em dashes; line breaks follow the image |
| `noisy_text.gt.txt` | 11 lines, italic serif, coffee-stain background |
| `noisy_text2.gt.txt` | 11 lines, typewriter font, crumpled-paper background |

Both `noisy_text*` scans are clipped at the right edge mid-word; only characters judged
fully visible are transcribed, so a line's last character may be off by one. Treat CER
differences below ~0.5 points on these pages as noise.
