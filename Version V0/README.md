<img width="273" height="121" alt="FusionCore"
src="https://github.com/user-attachments/assets/692079cc-0f95-4386-bf88-2256625293c6" />

# FusionCore: Physics-Aware Predictive Maintenance Framework

**FusionCore $v_0$** is a research-led framework designed to solve the *Structural Leakage* and *Uncertainty Quantification* problems in aerospace prognostics. Moving beyond standard regression, FusionCore implements a rigorous **Zero-Leakage Normalisation** pipeline and a **Forensic Validation Gate** to ensure Remaining Useful Life $(RUL)$ estimations are driven by physical thermodynamic degradation signals rather than temporal metadata artefacts.

---

## 1. Executive Summary

Predictive maintenance in safety-critical domains demands more than low Root Mean Square Error $(RMSE)$; it requires **statistical honesty**, **physical interpretability**, and **quantifiable operational risk**. A model that achieves low $RMSE$ by memorising the passage of time rather than learning degradation physics provides no defensible basis for maintenance decision-making.

FusionCore $v_0$ establishes a mathematically verified ground truth by subjecting State-of-the-Art $(SOTA)$ deep learning architectures to forensic stress testing against the NASA C-MAPSS dataset (Saxena & Goebel, 2008; FD001–FD004). The objective is to construct a model that learns the physics of degradation — explicitly mapping statistical anomalies to empirical aerodynamic efficiency loss and material fatigue — not merely the arithmetic progression of cycle indices.

---

## 2. Core Engineering Challenges and Solutions

Standard deep learning pipelines applied to turbofan prognostics exhibit three structural failure modes. FusionCore remediates each through explicit physical grounding.

### 2.1 The Piecewise-Linear RUL Trap

Assuming engines degrade linearly from cycle 1 violates established fatigue mechanics (Miner, 1945). For the first $\sim$50–75 cycles, sensor noise is stochastic and indistinguishable from healthy operation — there is no discernible degradation trend. A model trained on a raw linear RUL target overfits to early-life noise rather than learning the thermodynamic signatures of genuine deterioration.

FusionCore implements a **piecewise-linear clipped RUL target** (Heimes, 2008), capping the label at:

$$RUL_{\text{clip}} = \min(RUL_{\text{true}},\ 125\ \text{cycles})$$

This plateau represents the incubation period in which sensor behaviour is stationary. The model is trained to output a constant healthy-state prediction across this plateau,reserving its capacity for the degradation region where physical signal is present.

### 2.2 Regime Blindness

Multi-regime subsets (FD002, FD004) span six discrete flight conditions — from sea-level ground idle to cruise at 42,000 ft. Without regime conditioning, altitude-driven pressure offsets in sensors such as $P_{30}$ dominate the signal space and suppress the smaller degradation-driven drift, causing the model to mistake operating-point variation for compressor failure.

FusionCore resolves this via a **Zero-Leakage Normalisation** protocol. $K$-Means clustering ($K = 6$) identifies flight regimes exclusively on the internal training partition. A per-regime $Z$-score transformation then decouples environmental variance from mechanical variance:

$$Z = \frac{X_{\text{raw}} - \mu_{\text{regime}}}{\sigma_{\text{regime}}}$$

All regime parameters ($\mu$, $\sigma$, $K$-Means centroids) are frozen on the training split before any application to validation or test data, enforcing the zero-leakage constraint.

### 2.3 Structural Temporal Leakage

Time-series models trained on row-indexed data can achieve artificially low validation error by memorising cycle position rather than learning thermodynamic state. FusionCore subjects all feature spaces to a **cryptographic sanitisation** phase comprising two tests:

- **Target Null Test:** The $RUL$ target vector is randomly shuffled whilst the feature   matrix $X$ is held intact. A model achieving low error on the corrupted target has   memorised row indices, not physics.
- **Permutation Importance Test:** The `time_cycles` column is randomly shuffled and the   error recomputed. Stable error confirms the model does not rely on explicit cycle count.

---

## 3. The Five-Phase Architecture Pipeline

The repository is structured around a strict five-phase execution plan, each phase producing a versioned notebook and a formal technical analysis document.

| Phase | Title | Core Deliverable |
| --- | --- | --- |
| **1** | EDA & Physics Grounding | Thermodynamic variance isolation; variance threshold $\tau = 1 \times 10^{-5}$; 13-gap registry |
| **2** | Zero-Leakage Normalisation | $K$-Means regime clustering; per-regime $Z$-normalisation; FD00u unified dataset (709 engines, 160,359 rows) |
| **3** | Physics-Aware Feature Engineering | 91-feature matrix; 3 virtual sensors; 3 cumulative fatigue indices; survival analysis (G12); UWL recalibration (G13) |
| **4** | Forensic Stress Test | XGBoost baseline; SHAP attribution audit; leakage verification; SOTA neural network training |
| **5** | SOTA Benchmarking & Evaluation | Aerospace-grade evaluation matrix across all four architectures |

### Phase 3 Feature Construction Summary

The 91-feature matrix $X \in \mathbb{R}^{129{,}331 \times 91}$ is constructed via four sequential steps:

$$|X| = N_{\text{base}} + (N_{\text{active}} \times 3) + N_{\text{virtual}} + N_{\text{fatigue}} + N_{\text{gap}} = 24 + 51 + 3 + 3 + 10 = 91$$

**Thermodynamic Virtual Sensors** encode physics-derived composite quantities that the model would otherwise need to infer implicitly. They are constructed in preference to raw lag features, which carry no physical interpretability:

**Compressor Pressure-Temperature Ratio (CPR):**

$$\text{CPR} = \frac{P_{30}}{T_{30}} = \frac{s7}{s3}$$

Proportional to HPC outlet gas density by the ideal gas law $\left(\rho \propto \frac{P}{T}\right)$. A declining CPR — falling outlet pressure concurrent with rising outlet temperature — is the compound thermodynamic signature of HPC isentropic efficiency loss (Saxena & Goebel, 2008). The original specification $P_{30}/P_{2}$ was revised to $P_{30}/T_{30}$ because $s5\ (P_2)$ is regime-dead in $Z$-normalised space, making the original denominator degenerate. Note: CPR exhibits extreme positive skewness (skew $= 41.20$, excess kurtosis $= 8{,}403.77$); a Yeo-Johnson power transform is mandatory before neural network training in Phase 4.

**Thermal Efficiency Proxy ($E_{\text{thermal}}$):**

$$E_{\text{thermal}} = \frac{\phi \times Ps_{30}}{N_c} = \frac{s12 \times s11}{s9}$$

$$\phi \times Ps_{30} = \frac{\dot{m}_f}{Ps_{30}} \times Ps_{30} = \dot{m}_f$$

Dividing by core speed $N_c$ produces a quantity proportional to fuel consumption per unit rotor work — inversely related to thermodynamic cycle efficiency. A rising $E_{\text{thermal}}$ over engine life is consistent with the fuel burn penalty associated with progressive component degradation (Walsh & Fletcher, 2004). Note:exhibits extreme skewness (skew $= 171.82$, excess kurtosis $= 39{,}215.48$);
Yeo-Johnson transform mandatory before neural network training.

**EGT Margin Drift ($EGT_{\text{drift}}$):**

$$EGT_{\text{drift}}(t) = z_{s4}(t) - \mu_{\text{baseline}}, \quad \mu_{\text{baseline}} = \text{mean}(z_{s4})_{\text{train}} \approx 0$$

Formalises the EGT margin concept central to commercial engine health management practice, where EGT margin erosion relative to a certified limit is the primary dispatch monitoring parameter used by major MRO operators and OEMs. Well-behaved distribution (skew $= 0.470$, excess kurtosis $= -0.102$); no transformation required.

**Cumulative Fatigue Indices** implement a Miner's Rule (Miner, 1945) damage accumulation proxy for sensors $s4$ (EGT), $s7$ ($P_{30}$), and $s9$ ($N_c$):

$$D_i(t) = \sum_{\tau=0}^{t} \max\!\left(z_{s_i}(\tau),\ 0\right)$$

Only positive $Z$-score exceedances are accumulated, consistent with the physical interpretation that damage accrues only when the sensor exceeds the healthy regime baseline. All three indices are verified strictly non-decreasing (monotonicity tolerance $-10^{-10}$) at Feature Engineering Gate 6.

---

## 4. Benchmarked Architectures

All four architectures receive an identical 91-feature input vector. Model selection is not determined by $RMSE$ alone. The evaluation matrix spans four quadrants to reflect the asymmetric operational consequences of prediction error in aviation:

**Quadrant A — Point Estimation:**

$$RMSE = \sqrt{\frac{1}{n}\sum_{i=1}^{n}(d_i)^2}, \qquad MAE = \frac{1}{n}\sum_{i=1}^{n}|d_i|$$

where $d_i = \hat{Y}_i - Y_i$ (predicted minus true RUL).

**Quadrant B — Aerospace Safety (NASA Asymmetric Scoring Function):**

$$S = \sum_{i=1}^{n} s_i, \quad \text{where} \quad s_i = \begin{cases} e^{-d_i/13} - 1 & d_i < 0 \quad \text{(early prediction)} \\\\ e^{\,d_i/10} - 1 & d_i \geq 0 \quad \text{(late prediction)} \end{cases}$$

Late predictions (under-estimated RUL) incur an exponential penalty; early predictions (over-estimated RUL) incur a linear penalty. A lower $S$ indicates a safer model. If Model A achieves lower $RMSE$ but Model B achieves lower $S$, Model B is the operationally superior result.

**Quadrant C — Operational Classification:** The continuous regression output is binned into three maintenance decision classes:

| Class | Condition | Operational Action |
| --- | --- | --- |
| 0 — Healthy | $RUL \geq 60$ | Normal operations |
| 1 — Warning | $30 \leq RUL < 60$ | Trigger supply chain and logistics |
| 2 — Critical | $RUL < 30$ | Aircraft On Ground / immediate maintenance |

Class 2 Recall — the fraction of genuinely critical engines that the model correctly identifies — is the primary safety metric of the evaluation:

$$\text{Class 2 Recall} = \frac{TP}{TP + FN}$$

**Quadrant D — Explainability:** SHAP Variable Importance (XGBoost baseline) and TFT Variable Selection Network weights must confirm that engineered thermodynamic features ($\text{CPR}$, $E_{\text{thermal}}$, cumulative fatigue indices) dominate attribution. If raw sensor noise drives predictions ahead of physics-derived features, Phase 3 feature engineering has failed its mandate.

| **Architecture** | **Role** | **Engineering Rationale** |
| --- | --- | --- |
| **XGBoost** | **The Forensic Baseline** | Gradient Boosted Decision Tree providing a rapid, interpretable $MAE$/$RMSE$ baseline. SHAP values audit whether engineered thermodynamic proxies dominate feature importance over raw sensor readings. |
| **DeepAR** | **The Risk Estimator** | Probabilistic autoregressive RNN (Salinas et al., 2020). Produces a full $RUL$ probability distribution, enabling uncertainty bounding via the $10^{\text{th}}$–$90^{\text{th}}$ percentile interval for in-flight failure risk quantification. |
| **Temporal Fusion Transformer (TFT)** | **The Glass Box** | Utilises a Variable Selection Network (VSN) and gated attention mechanisms to provide per-feature interpretability (Lim et al., 2021). Enables engineers to audit which thermodynamic virtual sensors drive degradation predictions — a direct verification that the physics-aware feature engineering has been learned. |
| **PatchTST** | **The Pattern Recogniser** | Segments sensor time-series into semantic patches under a Channel-Independence assumption (Nie et al., 2023). Captures long-range kinematic degradation trends whilst remaining robust to high-frequency sensor noise that would confuse token-level attention. |

Hyperparameter optimisation for all three neural architectures is conducted via the Optuna framework implementing the Tree-structured Parzen Estimator (TPE) algorithm — a Bayesian search strategy that models the probability of achieving low loss given the hyperparameter configuration, materially more efficient than grid or random search for high-dimensional
spaces.

---

## 5. Author

**Michele Maestrini**

Machine Learning Engineer $\|$ Aerospace Prognostics (PHM) $\|$ Physics-Informed
Time-Series Modelling

[LinkedIn](www.linkedin.com/in/michele-maestrini)
