# Repository Evaluation Report: Temporal Point Process Learning via Numerical Integration

> Comprehensive analysis of scientific merit, code quality, and future potential.

---

## 1. Executive Summary

This repository accompanies the paper *"A machine learning approach for learning temporal point process"* (ComSIS 2022, Vol 19, Issue 2, pp 1007-1022). It proposes using 1D numerical integration (Trapezoid, Euler, Implicit Euler, Simpson's, Gaussian Quadrature) to approximate the integral in the negative maximum likelihood (neML) loss for temporal point processes, enabling gradient-based optimization of parametric intensity functions.

**Verdict: Modest but real contribution with significant limitations. The specific research direction (numerical integration for TPP likelihood) is a solved sub-problem within a rapidly advancing field. It is unlikely to yield breakthrough results on its own, but the underlying skills and domain could be redirected toward more impactful directions.**

---

## 2. Core Principles & Methodology

### 2.1 What This Does

Temporal point processes model the timing of discrete events (e.g., car arrivals, injuries). The key object is the **conditional intensity function** (CIF) `lambda(t|H_t)`, which gives the instantaneous rate of events given history `H_t`.

The log-likelihood of a temporal point process is:

```
L = sum(log(lambda(t_i))) - integral_0^T lambda(s) ds
```

The **first term** is straightforward — evaluate the intensity at observed event times. The **second term** (the compensator integral) is the hard part — for most non-trivial intensity functions, it has no closed-form solution.

### 2.2 The Proposed Solution

The paper's contribution is: **use classical numerical integration schemes to approximate the compensator integral**, then backpropagate through the approximation using PyTorch autograd.

Five quadrature methods are implemented:
- Euler (explicit)
- Implicit Euler
- Trapezoid rule
- Simpson's rule
- Gaussian quadrature (Legendre)

These are applied to four parametric intensity models:
- **PoissonTPP**: constant intensity `lambda(t) = |b|`
- **GausTPP**: Gaussian kernel `lambda(t) = N(t; mu, sigma) + a`
- **HawkesTPP**: self-exciting `lambda(t) = |mu| + |alpha| * sum(exp(-|t - t_i|))`
- **PoissonPolynomialTPP**: quadratic `lambda(t) = |a + bt + ct^2|`

Plus two neural models (LSTM, FCN) that were explored but not prominently featured in results.

### 2.3 Evaluation

Models are trained via MLE, then events are simulated from the learned intensity using **Ogata's thinning algorithm**. Simulated event counts in time bins (size 5, 10, 15) are compared against ground truth using **MAE**.

Two datasets: highway traffic (high-frequency) and ski injuries (low-frequency).

---

## 3. Strengths

### 3.1 Correct Fundamental Idea
The approach of numerically integrating the compensator and backpropagating through it is mathematically sound. It generalizes to any differentiable intensity function, which is a genuine advantage over models requiring closed-form integrals.

### 3.2 Practical Comparison of Integration Schemes
Comparing five numerical integration methods on the same problem is pedagogically valuable. The results show that integration method choice matters (GausTPP+Euler is consistently best, Simpson's can be unstable), which is a useful empirical finding.

### 3.3 Dual-Dataset Validation
Testing on both high-frequency (traffic, ~events/minute) and low-frequency (ski injuries, ~events/day) processes demonstrates some generalization.

### 3.4 Complete Pipeline
The repo contains a full train-simulate-evaluate pipeline with Ogata's thinning, which is non-trivial to implement correctly.

---

## 4. Areas for Improvement

### 4.1 Code Quality Issues

| Issue | Location | Severity |
|-------|----------|----------|
| **Hardcoded magic values** | `HawkesTPP.py:7-8` — parameters initialized to specific values (0.1842, 1.7657) instead of being configurable | Medium |
| **Duplicated code** | `BaselineTraining.py` has `integral()` (L10-92) and `integral_testing_steps()` (L202-284) that are near-identical | High |
| **Duplicated model files** | Models exist in both `/models/` and `/train/models/` | High |
| **Bug in integral_testing_steps** | L284: `return` inside a for-loop causes premature exit after first iteration | Critical |
| **No requirements.txt** | Despite README referencing `pip install -r requirements.txt` | Medium |
| **Bare except clause** | `evaluation.py:125-127` catches all exceptions silently | Medium |
| **Hardcoded pi** | `GausTPP.py:12` uses `3.14` instead of `math.pi` or `torch.pi` | Low |
| **Unused parameter** | `PoissonTPP` has parameter `a` that is never used in `forward()` | Low |
| **Serbian comments** | Comments in Serbian without translation reduce accessibility | Low |
| **No reproducibility** | No random seeds set, no config files, results cannot be exactly reproduced | High |

### 4.2 Methodological Issues

**4.2.1 The "Novelty" Is Standard Practice**

The core claim — using numerical quadrature to approximate the compensator integral — is the *default* approach in the neural TPP literature. Every neural intensity model (Neural Hawkes Process [Mei & Eisner 2017], RMTPP [Du et al. 2016]) that doesn't have a closed-form integral uses Monte Carlo or quadrature approximation. The paper frames this as novel, but it is essentially the standard technique re-derived.

**4.2.2 Weak Baselines**

The models compared are extremely simple:
- Poisson (1 parameter)
- Gaussian (3 parameters)
- Hawkes with exponential kernel (2 parameters)
- Polynomial (3 parameters)

No comparison with any neural TPP method (RMTPP, Neural Hawkes, THP, SAHP) or even classical statistical baselines (kernel density estimation, ETAS). The LSTM and FCN models are implemented but barely appear in results.

**4.2.3 Evaluation Metric Concerns**

MAE on binned event counts is an unusual and weak metric for point processes. Standard metrics include:
- **Log-likelihood on held-out data** (the natural metric)
- **RMSE / MAPE on inter-event times**
- **Calibration plots** (reliability diagrams)
- **QQ-plots of time-rescaled residuals** (testing the time-change theorem)
- **CRPS (Continuous Ranked Probability Score)**

Binned MAE conflates two sources of error: the intensity estimate and the simulation stochasticity. With 10,000 simulations, the simulation variance should wash out, but the metric itself loses temporal resolution.

**4.2.4 No Statistical Significance**

Results report MAE and std, but no confidence intervals, no hypothesis tests, no cross-validation. With only two datasets and ~6 model variants, the experimental evidence is thin.

**4.2.5 Simulation-Based Evaluation Is Indirect**

Evaluating by simulating from the learned model and comparing simulated event counts to ground truth is valid but indirect. Direct evaluation of predictive performance (next-event prediction, intensity estimation quality) would be more informative and standard.

**4.2.6 The GausTPP Model Is Questionable**

`GausTPP` uses a single Gaussian kernel centered at a learned `mu` — this means the intensity peaks once and decays forever. It cannot model periodic or non-stationary processes. That it performs "best" suggests the test windows are short enough that a single-bump intensity suffices, which raises questions about how challenging the evaluation actually is.

### 4.3 Missing Features

- No hyperparameter tuning framework
- No cross-validation
- No learning rate scheduling
- No early stopping (trains for fixed 200 epochs, saves at every epoch)
- No model selection criteria (AIC, BIC)
- No multi-variate/marked point process support
- No standard benchmark datasets (MIMIC-II, StackOverflow, Taxi, etc.)

---

## 5. Results Analysis

### Best Results (bin_size=5):

| Dataset | Best Model | MAE | Worst Model | MAE |
|---------|-----------|-----|-------------|-----|
| Autoput (ski folder) | GausTPP(Euler) | 2.97 | PoissonTPP(Euler) | 4.51 |
| Ski | GausTPP(Euler) | 2.93 | PoissonTPP(Euler) | 5.28 |
| Autoput (autoput folder) | Hawk(Simson) | 5.21 | PoissonTPP | 8.90 |

The absolute MAE values (2.93-5.21 events per 5-unit bin) are difficult to interpret without knowing the average event count per bin. If bins typically have ~10 events, an MAE of 3 is a 30% error rate, which is mediocre. If bins have ~50 events, it's excellent. This context is missing from the results.

The standard deviations are very high relative to means (std/mean ~ 0.75-0.85), indicating high variance across simulations or bins.

---

## 6. Comparison with State of the Art

The temporal point process field has advanced dramatically since 2022:

| Method | Year | Approach | Key Innovation |
|--------|------|----------|----------------|
| RMTPP | 2016 | RNN + exponential intensity | First neural TPP |
| Neural Hawkes | 2017 | Continuous-time LSTM | State-dependent intensity |
| **This work** | **2022** | **Numerical integration + parametric models** | **Quadrature comparison** |
| THP | 2020 | Transformer + attention | Self-attention for event history |
| SAHP | 2020 | Self-attentive Hawkes | Continuous-time attention |
| A-NHP | 2022 | Attentive neural Hawkes | Flexible attention kernels |
| DTPP | 2024 | Decomposable Transformer | Log-normal mixture, no thinning |
| LLM-TPP | 2023-24 | Language models | Few-shot temporal reasoning |
| Mamba-TPP | 2024+ | State-space models | Linear-time sequence modeling |

This work operates at the level of 2016-era methods while the field has moved to transformers, neural ODEs, and now LLMs. The gap is substantial.

---

## 7. Honest Assessment: Fruitful Direction or Dead End?

### 7.1 This Specific Direction: Largely a Dead End

The particular contribution — comparing numerical integration schemes for TPP likelihood — has limited upside:

1. **The problem is solved**: Monte Carlo integration and quadrature-based approaches are well-understood. The field has moved to intensity-free methods (modeling the conditional density directly) that bypass the integral entirely.

2. **No scaling path**: The parametric models (2-3 parameters) cannot compete with neural approaches on complex real-world data. The numerical integration comparison becomes irrelevant when you switch to neural intensity functions that can use specialized techniques (e.g., neural ODE solvers, or the time-change theorem).

3. **Low citation impact**: The paper appears to have minimal citations after 3+ years, suggesting the community did not find the contribution significantly novel.

4. **The "novelty" is standard practice**: Numerical quadrature for the compensator is what everyone does when they don't have a closed-form integral. Framing it as the main contribution is not compelling to the TPP community.

### 7.2 The Broader Domain: Highly Fruitful

Temporal point processes themselves remain an active, vibrant research area. There are genuinely open problems:

1. **Scalable TPPs for massive event streams** (millions of events, real-time) — current transformer-based methods have quadratic complexity
2. **Multi-modal point processes** — combining event sequences with text, images, or graphs
3. **Causal inference with point processes** — understanding interventional effects on event dynamics
4. **Foundation models for event sequences** — pre-trained models that transfer across domains (emerging direction with LLMs)
5. **Spatio-temporal point processes** — adding spatial dimensions to temporal models
6. **Efficient architectures** — applying Mamba/RWKV/state-space models to replace transformers in TPP

### 7.3 Recommendations for the Authors

If continuing in this space, consider:

1. **Pivot to neural TPPs with modern architectures**: Build on the point process expertise but use transformer or state-space model encoders instead of 2-parameter models
2. **Benchmark on standard datasets**: MIMIC-II, StackOverflow, Taxi, Taobao, Reddit — this enables direct comparison with published work
3. **Use standard evaluation**: Log-likelihood, next-event prediction accuracy, calibration metrics
4. **Explore intensity-free methods**: Model the conditional density `p(t|H_t)` directly rather than the intensity, avoiding the integral entirely
5. **Consider applied niches**: The ski injury / traffic domain expertise could be valuable if combined with state-of-the-art methods and real-world deployment
6. **Use existing frameworks**: EasyTPP (ICLR 2024) provides standardized benchmarking infrastructure

---

## 8. Final Verdict

| Dimension | Rating | Notes |
|-----------|--------|-------|
| Mathematical Correctness | 7/10 | Core approach is sound, some implementation bugs |
| Novelty | 3/10 | Numerical quadrature for compensator is standard practice |
| Code Quality | 4/10 | Duplicated code, hardcoded values, no reproducibility |
| Experimental Rigor | 3/10 | 2 datasets, no standard benchmarks, weak metrics, no significance tests |
| Impact Potential | 3/10 | Low citations, field has moved past this approach |
| Domain Relevance | 6/10 | TPPs remain highly relevant; the specific angle is outdated |
| **Overall** | **4/10** | **Honest early-career research with a correct but incremental contribution** |

**Bottom line**: This is solid early-career work that demonstrates understanding of point process fundamentals. The specific research direction (numerical integration comparison) is a dead end — the field has moved to neural and intensity-free approaches that render this comparison moot. However, the *domain expertise* in temporal point processes is valuable and could be redirected toward genuinely open problems like scalable neural TPPs, spatio-temporal modeling, or foundation models for event sequences.

---

*Report generated: 2026-02-25*

## Sources
- [Paper: A machine learning approach for learning temporal point process (ResearchGate)](https://www.researchgate.net/publication/360596077_A_machine_learning_approach_for_learning_temporal_point_process)
- [Advances in TPPs: Bayesian, Deep, and LLM Approaches (2025 survey)](https://arxiv.org/html/2501.14291v1)
- [Extensive Survey on Deep Temporal Point Process](https://arxiv.org/html/2110.09823v5)
- [Decomposable Transformer Point Processes (NeurIPS 2024)](https://proceedings.neurips.cc/paper_files/paper/2024/file/a1f0d96171c8e79019ae35ee439c2938-Paper-Conference.pdf)
- [TPP Paper Collection (GitHub)](https://github.com/yangalan123/TemporalPointProcessPapers)
- [Connected Papers graph for this work](https://www.connectedpapers.com/main/83e29eabdfc9a1d1a7bcc786585e74c6ce983f8f/A-machine-learning-approach-for-learning-temporal-point-process/graph)
