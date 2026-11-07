# VFlash — Detailed Results

[VFlash](README.md) · [Training](TRAINING.md) · [Data Preparation](DATA_PREPARATION.md) · [Evaluation](EVALUATION.md) · [Detailed Results](RESULTS.md)

This page summarizes the published experiments from the [technical report](https://arxiv.org/abs/2611.08449), including target–drafter pairs, downstream systems, and latency measurements.

Each numeric cell reports one value. **Bold** marks the better result within the stated comparison; ↑ means higher is better and ↓ means lower is better.

## Table of Contents

- [Technical report results](#technical-report-results)
- [Robot 3D trajectories](#robot-3d-trajectories)
- [Robot 2D trajectories](#robot-2d-trajectories)
- [Box detection](#box-detection)
- [Dense 3D motion](#dense-3d-motion)
- [Downstream applications](#downstream-applications)
- [Planning latency](#planning-latency)

<a id="technical-report-results"></a>
## Technical Report Results

The report's dense-motion experiments use the released MolmoMotion-AR target with ordinary-text coordinates. That backend, the downstream action models, and the report's latency runtimes are outside this release.

<a id="robot-3d-trajectories"></a>
### Robot 3D Trajectories

Table 4 of the report. ADE and FDE are in meters; SR@5 and SR@10 are fractions of waypoints within 5 cm and 10 cm. The target variants differ in their output supervision and whether wrist-camera views were added during training. All are evaluated with single-view inputs. The README summarizes the **2D+3D target with wrist-camera training**; the other variants are retained below.

#### 3D-only target

| Benchmark | Method | ADE (m) ↓ | FDE (m) ↓ | SR@5 ↑ | SR@10 ↑ |
|:--|:--|--:|--:|--:|--:|
| InternA1 | AR | 0.101 | 0.155 | 0.412 | 0.658 |
| InternA1 | VFlash | **0.065** | **0.098** | **0.638** | **0.820** |
| AgiBot | AR | 0.083 | 0.125 | 0.503 | 0.799 |
| AgiBot | VFlash | **0.042** | **0.067** | **0.731** | **0.914** |
| MolmoBot | AR | 0.066 | 0.104 | 0.526 | 0.804 |
| MolmoBot | VFlash | **0.054** | **0.083** | **0.653** | **0.870** |

#### 2D+3D target

| Benchmark | Method | ADE (m) ↓ | FDE (m) ↓ | SR@5 ↑ | SR@10 ↑ |
|:--|:--|--:|--:|--:|--:|
| InternA1 | AR | 0.153 | 0.233 | 0.273 | 0.473 |
| InternA1 | VFlash | **0.062** | **0.091** | **0.640** | **0.827** |
| AgiBot | AR | 0.105 | 0.151 | 0.398 | 0.698 |
| AgiBot | VFlash | **0.041** | **0.064** | **0.739** | **0.919** |
| MolmoBot | AR | 0.076 | 0.116 | 0.467 | 0.767 |
| MolmoBot | VFlash | **0.055** | **0.085** | **0.640** | **0.862** |

#### 2D+3D target with wrist-camera training

| Benchmark | Method | ADE (m) ↓ | FDE (m) ↓ | SR@5 ↑ | SR@10 ↑ |
|:--|:--|--:|--:|--:|--:|
| InternA1 | AR | 0.130 | 0.196 | 0.360 | 0.554 |
| InternA1 | VFlash | **0.060** | **0.091** | **0.651** | **0.835** |
| AgiBot | AR | 0.084 | 0.129 | 0.456 | 0.749 |
| AgiBot | VFlash | **0.036** | **0.057** | **0.794** | **0.937** |
| MolmoBot | AR | 0.066 | 0.102 | 0.547 | 0.813 |
| MolmoBot | VFlash | **0.052** | **0.080** | **0.671** | **0.877** |

<a id="robot-2d-trajectories"></a>
### Robot 2D Trajectories

Table 2 of the report. DFD, HD, and RMSE use normalized image coordinates. AR and VFlash share the same 4B target. The report also compares with prior models.

| Benchmark | Method | DFD ↓ | HD ↓ | RMSE ↓ |
|:--|:--|--:|--:|--:|
| ShareRobot-Bench | Baseline (AR) | 0.202 | 0.175 | 0.142 |
| ShareRobot-Bench | VFlash | **0.194** | **0.162** | **0.133** |
| VABench-V | Baseline (AR) | 0.130 | 0.125 | 0.093 |
| VABench-V | VFlash | **0.126** | **0.116** | **0.089** |

<a id="box-detection"></a>
### Box Detection

Table 3 of the report. F1 is reported in percent. BPS is boxes per second on one H100. *Direct* accepts every schema-valid box; *AR fallback* regenerates boxes rejected by the spatial-confidence guard. Bold compares all three methods within each benchmark.

| Benchmark | Method | Mean F1 (%) ↑ | F1@0.50 (%) ↑ | F1@0.95 (%) ↑ | BPS ↑ |
|:--|:--|--:|--:|--:|--:|
| COCO | Baseline (AR) | **48.68** | 65.29 | **16.09** | 3.15 |
| COCO | VFlash (direct) | 41.32 | 59.64 | 10.76 | 5.96 |
| COCO | VFlash (AR fallback) | 47.79 | **66.38** | 13.48 | **8.76** |
| Dense200 | Baseline (AR) | 60.10 | 76.70 | **16.95** | 6.73 |
| Dense200 | VFlash (direct) | 39.09 | 55.06 | 6.59 | **22.66** |
| Dense200 | VFlash (AR fallback) | **61.78** | **82.64** | 12.11 | 17.46 |

<a id="dense-3d-motion"></a>
### Dense 3D Motion

Table 5 of the report: PointMotionBench with a frozen MolmoMotion-AR target. ADE and FDE are in meters; PWT is a fraction. The approximately 25× speedup describes single-rollout generation and is separate from the quality scores below.

| Benchmark | Method | ADE (m) ↓ | FDE (m) ↓ | PWT ↑ |
|:--|:--|--:|--:|--:|
| HOT3D | MolmoMotion-AR | **0.109** | 0.217 | **0.444** |
| HOT3D | VFlash | 0.122 | **0.200** | 0.391 |
| WorldTrack | MolmoMotion-AR | **0.143** | **0.261** | **0.445** |
| WorldTrack | VFlash | 0.145 | 0.300 | 0.408 |
| DAVIS | MolmoMotion-AR | 1.227 | 2.108 | **0.153** |
| DAVIS | VFlash | **0.893** | **1.716** | 0.143 |

<a id="downstream-applications"></a>
### Downstream Applications

#### MolmoSpaces manipulation

Tables 6 and 7 of the report. Success rates are percentages over four task suites of the same world action model (WAM). The baseline keeps the WAM's original text encoder. The other variants condition it on the frozen VLM's hidden states for question (Q), sub-task (S), trajectory (T), or image (I) tokens.

| Variant | Plan decoder | Plan | VLM context | Avg. success (%) ↑ |
|:--|:--|:--|:--|--:|
| Baseline | — | — | — | 62.6 |
| AR plans | AR | 2D+3D | Q+S+T | 61.0 |
| Image context | VFlash | — | Q+S+I | 63.7 |
| 3D-only plans | VFlash | 3D | Q+S+T | 69.3 |
| VFlash | VFlash | 2D+3D | Q+S+T | **70.2** |

| Method | Pick (%) ↑ | Pick & Place (%) ↑ | Open (%) ↑ | Close (%) ↑ |
|:--|--:|--:|--:|--:|
| Baseline | 81.1 | 55.7 | 49.3 | 64.2 |
| VFlash | **83.6** | **67.3** | **52.8** | **77.0** |

#### LIBERO family

<p align="center">
  <img src="../../assets/vflash/libero.png" alt="LIBERO family results" width="78%">
  <br>
  <sub><b>LIBERO family.</b> Average success (%) of the same PIv3 action head conditioned on a frozen Qwen3-VL-4B (Baseline), on our VLM's states up to the generated sub-task (w/o Visual Primitives), and on the full visual plan (VFlash).</sub>
</p>

#### Autonomous driving

Table 8 of the report. VFlash uses a frozen planning VLM and a small diffusion action head; RL fine-tunes only the action head. Bold compares methods within the same SFT or RL setting.

| Method | NAVSIM PDMS ↑ |
|:--|--:|
| Qwen-Drive-1.0-SFT | **88.2** |
| VFlash-SFT | 87.1 |
| Qwen-Drive-1.0-RL | **90.7** |
| VFlash-RL | **90.7** |

| Method | Waymo ADE 3s (m) ↓ | Waymo ADE 5s (m) ↓ | Waymo RFS ↑ |
|:--|--:|--:|--:|
| Qwen-Drive-1.0-SFT | **1.19** | **2.65** | 7.78 |
| VFlash-SFT | 1.23 | 2.71 | **7.79** |
| Qwen-Drive-1.0-RL | **1.19** | **2.67** | **7.91** |
| VFlash-RL | 1.22 | 2.70 | 7.86 |

<a id="planning-latency"></a>
### Planning Latency

MolmoSpaces latency per planning step at batch size one:

| Hardware | AR (ms) ↓ | VFlash (ms) ↓ |
|:--|--:|--:|
| Jetson Thor | 3552.7 | **703.3** |
| DGX Spark | 3356.5 | **649.9** |

<p align="center">
  <img src="../../assets/vflash/latency.png" alt="Inference latency" width="92%">
  <br>
  <sub><b>Inference latency</b> per planning step at batch size one on Jetson Thor, DGX Spark and H100, split into preprocessing and vision, prefill, decoding and the action model. On Thor and Spark, VFlash cuts per-step latency 5.1–5.2× for MolmoSpaces and 3.4–3.6× for NAVSIM (3.6× and 2.6× on H100).</sub>
</p>
