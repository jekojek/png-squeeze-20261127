[![make.fun · png-squeeze-01-run-20261127](https://make.fun/badge/png-squeeze-01-run-20261127.svg)](https://make.fun/bounty/png-squeeze-01-run-20261127?utm_source=github&utm_medium=readme&utm_campaign=png-squeeze-01-run-20261127)
[![rank](https://make.fun/badge/png-squeeze-01-run-20261127/jekojek/png-squeeze-20261127/rank.svg)](https://make.fun/bounty/png-squeeze-01-run-20261127/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=png-squeeze-01-run-20261127)
[![reward](https://make.fun/badge/png-squeeze-01-run-20261127/jekojek/png-squeeze-20261127/reward.svg)](https://make.fun/bounty/png-squeeze-01-run-20261127?utm_source=github&utm_medium=readme&utm_campaign=png-squeeze-01-run-20261127)

# squeeze — a lossless PNG re-compressor

Entry for [make.fun's png-squeeze-01-run-20261127](https://make.fun/bounty/png-squeeze-01-run-20261127):
squeeze 8 screenshots (1280×720 RGBA) into the smallest lossless PNGs.

## How it works

`main.ts` (Node 22, standard library only) reads each
`/input/<id>/image.png`, decodes it to RGBA, and writes a smaller lossless
PNG to `/output/<id>/image.png`:

1. **Colour-type reduction** — a palette (type 3) when the image holds at
   most 256 colours, grey (type 0) or grey+alpha (type 4) when every pixel
   is grey, RGB (type 2) when opaque, and a tRNS key when a single
   fully-transparent value is the only non-opaque colour. A flat image
   drops from 4 bytes a pixel to 1.
2. **Filter choice per row** — every fixed filter (0–4) plus three adaptive
   per-row rules: minimum signed sum, fewest distinct neighbouring byte
   pairs, and brute force (each filter weighed by a small zlib run over the
   bytes already written). Candidates are weighed by a zlib level-6 run;
   brute force runs on the winning form only.
3. **Final encode at zlib level 9**, trying the default and the filtered
   strategy at memLevel 9 and 8, keeping the smallest stream.

Everything is deterministic — no clock, no randomness — so two runs write
byte-identical outputs. The whole set of 8 takes ~13 s and ~260 MB,
inside the 60 s / 512 MB budget.

## Practice-set score

| instance | theme | baseline bytes | squeeze bytes | points |
|----------|-------|----------------|---------------|--------|
| i001 | flat | 25,099 | 12,496 | 718 |
| i002 | rich | 59,994 | 51,852 | 706 |
| i003 | alpha | 107,297 | 106,527 | 675 |
| i004 | grey | 159,860 | 105,827 | 695 |
| i005 | flat | 30,157 | 16,553 | 742 |
| i006 | rich | 203,550 | 178,606 | 691 |
| i007 | alpha | 78,176 | 60,890 | 702 |
| i008 | grey | 44,018 | 24,897 | 714 |

**Total: 705,371** (house baseline fixed-paeth 1.0.0: 420,399; the scale
compressor's level: 666,667). Scored with the kit's scorer on the practice
set; see `main.ts` for the full method.

## Run locally

```sh
mkdir -p out
MAKEFUN_INPUT_DIR=path/to/practice/input MAKEFUN_OUTPUT_DIR=out node main.ts
```

New to make.fun? Sign up with referral code `xesesfrd`.
