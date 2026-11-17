# Fuzzy–neural character recognizer

Course project for **EEO1003 Fuzzy Logic and Neural Networks**.

The project takes noisy or faded images of the 36 characters A–Z and 0–9 and cleans them with a fuzzy inference system written from scratch. Four classic neural networks then recognise or recall the character. Every experiment runs each network twice on the same noisy images: once on the plain thresholded image and once on the fuzzy-enhanced image.

**Full report:** [report/Fuzzy-Neural-Character-Recognition-Report.pdf](report/Fuzzy-Neural-Character-Recognition-Report.pdf). It covers the theory, design, results and discussion, in 19 pages.

**Study guide:** [GUIDE.md](GUIDE.md), also as a [printable PDF](report/Fuzzy-Neural-Character-Recognition-Guide.pdf). It explains every syllabus topic and how the project uses it, with worked examples, a file-by-file code tour, design decisions and viva questions with answers.

| Syllabus unit | Where it appears |
|---|---|
| 1. ANN models, Hebbian learning, perceptron | `perceptron.py` (perceptron learning rule); Hebbian storage in `hopfield.py`, `bam.py` |
| 2. Hopfield net | `hopfield.py` (discrete, asynchronous, energy function; Hebbian and projection rules) |
| 3. Hamming net, Maxnet, BAM | `hamming.py`, `bam.py` |
| 4. Fuzzy sets, membership functions, fuzzification, defuzzification | `fuzzy.py` (trapezoidal membership functions, fuzzification, INT operator, centroid) |
| 5. Fuzzy rules, fuzzy inference, image-processing application | `fuzzy.py` (9-rule Mamdani filter, min–max inference) |

## Running it

Needs Python 3 with numpy, matplotlib and Pillow (tkinter ships with Python). Nothing else to install.

```
python selftest.py      # 14 correctness checks, about 4 s
python experiments.py   # all experiments and figures in results/, about 40 s
python gui.py           # interactive demo
```

In the demo:
- Click a key, or type a character, to load that glyph.
- Draw on the input grid with the left mouse button; erase with the right.
- Choose a noise type and level.
- Compare the two signal paths, then press **Animate recall** to watch the Hopfield net settle.

## How the fuzzy stage works

For each noisy grayscale image (`fuzzy.enhance`):

1. **Fuzzification.** Each pixel gets a degree of membership in "ink", taken from the image's own intensity range. The 5th–95th percentile is mapped to 0–1, so a faded scan is stretched to full contrast.
2. **INT contrast intensification.** μ′ = 2μ² for μ ≤ 0.5, and 1 − 2(1 − μ)² otherwise. Uncertain pixels are pushed towards paper or ink.
3. **Mamdani rule filter.** The two inputs are the pixel value *p* and the mean of its 8 neighbours *m*, each with low/medium/high membership functions. Nine IF–THEN rules decide the output:
   - an isolated speck on the paper is removed (*p* high, *m* low → low);
   - a hole inside a stroke is filled (*p* low, *m* high → high);
   - an uncertain pixel follows its neighbours.
   
   Inference uses min for AND, min for implication and max for aggregation. Defuzzification is the centroid.
4. **Binarise** at 0.5 for the networks.

## Results (from `python experiments.py`)

- **Cleaner images.** Fuzzy enhancement cuts the pixel error rate at the network input:

  | Noise | Threshold only | Fuzzy |
  |---|---|---|
  | Gaussian, σ = 0.4 | 10.5% | 4.2% |
  | Salt-and-pepper, 30% | 15.0% | 7.0% |
  | Faded scan, σ = 0.25 | 25.4% | 19.7% |

  It removes most specks on the paper and fills holes in strokes (`ink_changes.png`).
- **Better recognition on faded scans.** At σ = 0.25 on a faded scan:

  | Network | Threshold only | Fuzzy |
  |---|---|---|
  | Perceptron | 88% | 94% |
  | Hopfield (projection rule) | 93% | 98% |
  | Hebbian Hopfield | 97% | 99% |
  | BAM | 96% | 99% |

- **Random noise.** All networks average independent pixel noise over 576 inputs, so they stay near 100% with or without enhancement. At the heaviest noise levels, where noise exceeds the ink contrast, the fuzzy filter can no longer separate strokes from noise and the advantage disappears.
- **Hopfield capacity.** Glyphs overlap heavily, so Hebbian storage fails early. Added in a greedy most-different-first order, the Hebbian rule keeps every pattern stable only up to 3 glyphs. The projection (pseudo-inverse) rule stores all 36 (`hopfield_capacity.png`). The greedy selection gave W, Q and D for the Hebbian Hopfield net and BAM, and the projection-rule Hopfield net is used for all 36.

Full numbers are in `results/summary.csv` and `results/image_quality.csv`.

## Files

| File | Purpose |
|---|---|
| `chars.py` | renders the 36 glyphs at 24×24 (Consolas Bold, so 0 and O differ) |
| `noise.py` | Gaussian, salt-and-pepper and faded-scan degradation |
| `fuzzy.py` | the fuzzy enhancement stage |
| `perceptron.py`, `hopfield.py`, `hamming.py`, `bam.py` | the networks |
| `subset.py` | picks the glyphs the Hebbian memories can store stably |
| `models.py` | builds and trains every network with fixed seeds |
| `checkpoint.py` | image-quality check of the fuzzy stage alone |
| `experiments.py` | noise sweeps and figures |
| `selftest.py` | correctness checks |
| `gui.py` | interactive demo |

All randomness uses fixed seeds, so every run gives the same numbers.
