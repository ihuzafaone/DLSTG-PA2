# AI651 Assignment 2 — River forecasting proposal (Q1) and "Motion or Memory?" on Moving MNIST (Q2)

Everything needed to regenerate the tables and figures of the report. Environment: conda env `PA1_DL`
(PyTorch 2.10, torchvision 0.25, OpenCV 4.13, matplotlib, scipy). Hardware used: NVIDIA RTX 4060 Laptop GPU (8 GB).

## Layout
| path | content |
|---|---|
| `src/mmdata.py` | Moving MNIST loading, mirror / rotation / symmetry group G, speed windows (eq. 1–2) |
| `src/models.py` | ConvLSTM forecaster (model A = defaults, 480,784 params; model D = other kernel/dilation config); analytic reach |
| `src/train.py` | training protocol (Adam 1e-3, batch 32, 30 epochs, seeds 0/1); models A, B, C, D, DB |
| `src/evalkit.py` | metrics (MSE_h, E^(5), E^(2), C_T, E_T), Blank / Copy-last, symmetrised forecaster (eq. 7), timing |
| `src/table1.py` | Table 1 checks | 
| `src/fig1_flow.py` | Figure 1 (Farneback optical flow) + calibration statistics |
| `src/exp_table2.py` | Table 2, Figure 2 |
| `src/reach_check.py` | autograd check of the one-step reach (A: 8 px, D: 16 px) |
| `src/exp_baseline.py` | Tables 3–4, Figures 3–5, GIFs (model A) |
| `src/diag.py`, `src/analysis_q27.py` | digit-displacement diagnostic (Table S1) and the Q2.7 centroid analysis |
| `src/screen_d.py` | validation-only screening of candidate designs for model D |
| `src/exp_final.py` | Tables 5–6, Figure 6, choice of the final model (validation) |
| `q1/river_prototype.py` | Q1: shape / size / latency check of the proposed forecaster (untrained forward pass) |
| `predictions/` | Q2.5 and Q2.11 predictions, **committed before the corresponding experiments were run** |
| `results/`, `figures/`, `checkpoints/`, `logs/` | all outputs (JSON, figures + GIFs, weights, training logs) |
| `report/` | LaTeX source of the report (`tectonic main.tex`) |

## Reproduce
```bash
conda activate PA1_DL
python src/table1.py; python src/fig1_flow.py
python src/train.py --model A --seed 0; python src/train.py --model A --seed 1
python src/exp_table2.py
# --- predictions/Q2.5.md is committed here, only then:
python src/exp_baseline.py
# --- predictions/Q2.11.md is committed here, only then:
python src/train.py --model B --seed {0,1}; python src/train.py --model C --seed {0,1}
python src/screen_d.py                      # validation split only
python src/train.py --model D --seed {0,1}; python src/train.py --model DB --seed {0,1}
python src/exp_final.py
```
The dataset is downloaded by torchvision (`MovingMNIST(split=None)`, ~819 MB) into `data/` (not committed).
The built-in `split="train"/"test"` options are never used: videos are split by index (0–7999 / 8000–8999 / 9000–9999).
