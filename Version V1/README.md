# FusionCore v1 — PiNet

**Physics-Aware Temporal Modelling for Turbofan Remaining Useful Life Estimation**

A research project developing a hybrid physics-guided neural architecture for Remaining Useful Life (RUL) prediction on the NASA C-MAPSS turbofan engine benchmark.

---

## Project Status

This repository contains the **methodology specification, design documents, and architectural diagrams** for FusionCore v1. The implementation phase is upcoming. What is in the repository now:

- Master Roadmap document defining the full v1 programme (architecture, training regime, evaluation framework, leakage discipline, phase plan)
- Architectural diagrams: PiNet two-branch backbone, dilated causal Temporal Convolutional Network receptive field, Siamese network with triplet loss
- Documentation style specification used to produce the roadmap

Implementation code (PyTorch model, training pipeline, evaluation scripts) is planned for the next phase. This is a deliberate methodology-first approach: the design has been specified to peer-review standard before code is written, so that the implementation has a defined and falsifiable target.

---

## What FusionCore Solves

Predicting Remaining Useful Life of commercial turbofan engines is a safety-critical problem in aerospace maintenance. Underestimating RUL leads to in-service failures and Aircraft-on-Ground events; overestimating it leads to unnecessary maintenance cost. The published gradient-boosting baseline (XGBoost) on the NASA C-MAPSS dataset achieves an RMSE of approximately 14.85 cycles and a NASA Asymmetric Score of 4,336 on the official 707-engine test set. FusionCore v1 asks whether a compact temporal neural architecture, operating on a physics-aware feature manifold, can equal or exceed that baseline while producing a richer operational output: scalar RUL plus a three-class operational risk band (Healthy, Warning, Critical).

The technical hypothesis is that temporal trajectory structure, modelled by a long-memory Temporal Convolutional Network, contains predictive signal that tree-based models cannot fully capture, and that pairing the TCN with a dedicated physics-token branch preserves the gas-path thermodynamic interpretation of degradation in the embedding space.

---

## PiNet Architecture
<img width="2890" height="4979" alt="Physics MLP and TCN-2026-04-29-045101" src="https://github.com/user-attachments/assets/d11da95a-13e6-4107-9b5e-9ed3bfe5c386" />



### PiNet is a Two-Branch Hybrid Architecture:

- **TCN branch** — a 5-layer dilated causal Temporal Convolutional Network with kernel size 3 and dilation factors {1, 2, 4, 8, 16}, producing an effective receptive field of 63 cycles over a 30-cycle input window.
- **Physics token branch** — a small two-layer MLP processing approximately 11 physically interpretable scalars per cycle (gas-path z-scores, virtual thermodynamic sensors, cumulative fatigue indices, the Cox proportional-hazards score, and the inverse-frequency regime weight).

Branch outputs concatenate into a 128-dimensional shared embedding, which feeds two output heads: a scalar RUL regression head clipped to [0, 125] cycles, and a three-class softmax classifying the engine state into Healthy, Warning, or Critical bands.

The two-term primary loss combines the NASA asymmetric RUL score with a class-weighted cross-entropy term for the risk band. A pre-declared advanced ablation extends this with a triplet metric-learning regulariser and a Shannon entropy term, executed on a representative sample subset to manage compute on Apple Silicon hardware.

Full architectural detail, evaluation framework, and phase-by-phase execution plan are in the Master Roadmap document.

---

## Repository Contents

Documents and figures defining the v1 programme. See the file listing in the repository for the current set of artefacts. Anything described here as "planned" is not yet committed.

---

## Methodology Highlights for Reviewers

The v1 design adheres to a strict zero-leakage discipline inherited from FusionCore v0 Phase 3:

- A three-tier partition (Internal Train 567 engines / Internal Validation 142 engines / NASA Official Test 707 engines) where the test set is touched exactly once at programme close and is never used for hyperparameter selection
- Read-only application of all preprocessing artefacts (Frozen Regime Dictionary, z-score parameters, Cox PH coefficients, Yeo-Johnson power transformer for high-skewness virtual sensors) fitted on Internal Train data only
- Composite-key engine grouping enforced at every windowing operation to prevent within-engine row leakage
- Fixed hyperparameters drawn from peer-reviewed C-MAPSS literature, with formal Bayesian Hyperparameter Optimisation deferred to avoid Validation-set contamination
- A documented leakage risk register identifying the live risks under the v1 design and the mitigation against each

The headline evaluation is reported on the NASA official test set, ensuring direct comparability with every published RMSE and NASA Asymmetric Score result in the C-MAPSS Prognostics and Health Management literature.

---

## References

The full bibliography is in the Master Roadmap document. Core references include Saxena and Goebel (2008) for the C-MAPSS benchmark, Heimes (2008) and Li et al. (2018) for prior neural RUL work on the dataset, Bai et al. (2018) for the Temporal Convolutional Network architecture, Cox (1972) for the proportional-hazards model, Schroff et al. (2015) for triplet metric learning, Vovk et al. (2005) for the conformal prediction framework underpinning the planned v2 extension, and Nowlan and Heap (1978) for the Reliability-Centred Maintenance grounding of the operational risk-band thresholds.

---

## Author

**Michele E. J. Maestrini** — MSc Data Science. The FusionCore programme synthesises predictive maintenance methodology with applied deep learning, with a research interest in physics-aware temporal modelling for aerospace prognostics.

---

## Roadmap

- **v0** (complete) — feature manifold construction, XGBoost baseline, SHAP forensic audit on NASA C-MAPSS
- **v1** (current) — PiNet two-branch hybrid architecture; methodology specified, implementation upcoming
- **v2** (planned) — probabilistic extension of the v1 backbone via conformal prediction intervals and risk-band probability calibration
- **v3** (forward-looked) — transfer to NASA N-CMAPSS and full deployment-grade evaluation

---

*This repository is a research artefact. The methodology is documented to peer-review standard so that the implementation, when built, has a falsifiable target rather than a flexible specification.*
