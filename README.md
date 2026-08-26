# EVALUATION OF FEDERATED LEARNING MODELS FOR PRIVACY-PRESERVING ELECTRONIC HEALTH RECORDS IN THE UK HEALTHCARE SYSTEM

**MRes Computing and Artificial Intelligence — University of Greater Manchester**
**MRES7015 Research Project · 2026**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![Flower](https://img.shields.io/badge/Flower-1.5.0-green)](https://flower.dev/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)](https://pytorch.org/)
[![Opacus](https://img.shields.io/badge/Opacus-1.4%2B-purple)](https://opacus.ai/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Overview

This repository contains the complete simulation code, preprocessing pipeline, and experimental results for the MRes dissertation:

> **"Evaluation of Federated Learning Models for Privacy-Preserving Electronic Health Records in the UK Healthcare System"**

The dissertation investigates whether federated learning (FL) can enable NHS Trusts to collaboratively train clinical prediction models without patient data ever leaving Trust boundaries. Four FL algorithms — FedAvg, FedProx, SCAFFOLD, and Per-FedAvg — are compared across three data heterogeneity conditions simulating NHS Trust diversity. Three privacy mechanisms — Differential Privacy, Secure Aggregation, and Homomorphic Encryption — are evaluated.

**Supervisor:** Dr Pradeep Hewage, University of Greater Manchester

---

## Key Findings

| Finding | Result |
|---|---|
| **Recommended algorithm** | **FedProx** — lowest training loss in all 3 conditions |
| FedProx IID convergence | 0.2484 → **0.1617** (−34.9%) |
| FedProx moderate non-IID | **0.1072** — lowest of all 12 runs |
| FedProx extreme non-IID | **0.1574** — most stable (σ=0.0026) |
| DP clinical utility | **All 5 epsilon configurations AUC > 0.80** |
| Best DP result (ε=0.5) | AUC = **0.8762** (above centralised baseline) |
| MIA attack AUC | **0.3983** — below random chance → LOW risk |
| HE overhead (full model) | **21.1ms** per aggregation round |
| Corrected baseline AUC | **0.8541** (after discharge_location leakage removal) |

---

## Repository Structure

```
federated-learning-nhs-ehr/
│
├── notebooks/
│   ├── 01_Preprocessing_Pipeline.ipynb    ← Data cleaning, feature engineering,
│   │                                          leakage detection, Dirichlet partitioning
│   └── 02_FL_Simulation.ipynb             ← FL simulation (4 algorithms × 3 conditions),
│                                              DP, MIA, HE evaluation
│
├── results/
│   ├── fedavg_iid.json                    ← Round-by-round training loss (50 rounds)
│   ├── fedprox_iid.json
│   ├── scaffold_iid.json
│   ├── perfedavg_iid.json
│   ├── fedavg_mod_noniid.json
│   ├── fedprox_mod_noniid.json
│   ├── scaffold_mod_noniid.json
│   ├── perfedavg_mod_noniid.json
│   ├── fedavg_ext_noniid.json
│   ├── fedprox_ext_noniid.json
│   ├── scaffold_ext_noniid.json
│   ├── perfedavg_ext_noniid.json
│   ├── dp_results.json                    ← DP evaluation (5 epsilon values)
│   ├── he_results.json                    ← HE timing results
│   ├── mia_results.json                   ← Membership inference attack results
│   └── test_evaluation.json               ← Corrected baseline model metrics
│
├── requirements.txt                       ← All Python dependencies with versions
├── .gitignore                             ← Excludes MIMIC-IV data (cannot be shared)
├── LICENSE                                ← MIT License
└── README.md                              ← This file
```

---

## Dataset — MIMIC-IV v3.1

> ⚠️ **The MIMIC-IV dataset is NOT included in this repository.** It contains de-identified patient records and requires a credentialled data use agreement.

**To access MIMIC-IV:**
1. Create an account at [PhysioNet](https://physionet.org/)
2. Complete the required training course: *"CITI Data or Specimens Only Research"*
3. Sign the Data Use Agreement (DUA)
4. Request access to [MIMIC-IV v3.1](https://physionet.org/content/mimiciv/3.1/)
5. Download and place in `data/mimic-iv-3.1/`

**Citation:**
> Johnson, A.E.W. et al. (2023) 'MIMIC-IV, a Freely Accessible Electronic Health Record Dataset', *Scientific Data*, 10(1), article 1. https://doi.org/10.1038/s41597-022-01899-x

> Goldberger, A.L. et al. (2000) 'PhysioBank, PhysioToolkit, and PhysioNet', *Circulation*, 101(23), pp.e215–e220. https://doi.org/10.1161/01.CIR.101.23.e215

---

## Installation

### Prerequisites
- Python 3.10 or later
- Google Colab (recommended — notebooks were developed and run on Colab)
- Google Drive (for data storage during simulation)

### Install dependencies

```bash
pip install -r requirements.txt
```

Or install individually:

```bash
pip install flwr[simulation]==1.5.0
pip install opacus==1.4.0
pip install tenseal==0.3.14
pip install torch>=2.0.0
pip install scikit-learn>=1.3.0
pip install pandas>=2.0.0
pip install numpy>=1.24.0
pip install matplotlib>=3.7.0
pip install ray>=2.6.0
```

---

## How to Run

### Notebook 01 — Data Preprocessing

```
Open notebooks/01_Preprocessing_Pipeline.ipynb in Google Colab
```

**What it does:**
1. Loads MIMIC-IV tables (admissions, icustays, patients, chartevents, labevents)
2. Engineers clinical features (vital signs and lab result aggregations)
3. Detects and removes data leakage (discharge_location, MI score = 0.1476)
4. Applies median imputation for missing values
5. Applies StandardScaler normalisation
6. Performs stratified 80/20 train/test split
7. Creates Dirichlet-partitioned NHS Trust datasets (α = 1.0, 0.5, 0.1)
8. Saves cleaned arrays: `X_train.npy`, `X_test.npy`, `y_train.npy`, `y_test.npy`

**Output:** Cleaned numpy arrays and Dirichlet partition files saved to Google Drive

---

### Notebook 02 — FL Simulation

```
Open notebooks/02_FL_Simulation.ipynb in Google Colab
```

**What it does:**

**Stage 1: Algorithm Comparison**
- Runs 12 FL simulations: 4 algorithms × 3 heterogeneity conditions
- 10 simulated NHS Trust nodes, 50 communication rounds each
- Records round-by-round training loss → saved to `/results/`

**Stage 2: Differential Privacy Evaluation**
- Applies DP-SGD (Opacus) across 5 epsilon values: 0.5, 1.0, 2.0, 5.0, 10.0
- δ = 1×10⁻⁵, 15 training epochs per configuration
- Evaluates AUC-ROC on corrected test set

**Stage 3: Membership Inference Attack**
- Shadow model attack (Shokri et al., 2017 methodology)
- n = 3,613 members + 3,613 non-members

**Stage 4: Homomorphic Encryption**
- CKKS scheme (TenSEAL) at 3 model sizes: 1,000 / 5,000 / 11,970 parameters
- Records encryption/decryption time and numerical error

**Output:** All results saved as JSON files in `/results/`

---

## Model Architecture — ClinicalNN

```python
class ClinicalNN(nn.Module):
    def __init__(self, input_dim=15):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 2)         # Binary: survived / died
        )
    def forward(self, x):
        return self.net(x)
```

**Total parameters:** 11,970
**Input features:** 15 clinical variables (after leakage removal)
**Task:** In-hospital mortality prediction (binary classification)
**Loss function:** Weighted cross-entropy (class weights to address imbalance)

---

## FL Algorithms Compared

| Algorithm | Reference | Key mechanism |
|---|---|---|
| **FedAvg** | McMahan et al. (2017) | Weighted model averaging |
| **FedProx** ★ | Li et al. (2020) | + Proximal term μ‖w − wᵍ‖² |
| **SCAFFOLD** | Karimireddy et al. (2020) | Control variates for drift correction |
| **Per-FedAvg** | Fallah et al. (2020) | MAML-based personalisation |

**★ FedProx recommended for NHS multi-Trust FL deployment**

---

## Important Note — Data Leakage Correction

During preprocessing, the variable `discharge_location` was identified as a data leakage source (Mutual Information score = 0.1476 — approximately 5× higher than any genuine clinical feature). This variable encodes the mortality outcome (values include "DIED", "HOSPICE") and was recorded after the patient outcome was determined.

**Impact:**
- Uncorrected baseline AUC-ROC: **0.9933** ← inflated, DO NOT USE
- Corrected baseline AUC-ROC: **0.8541** ← honest, used in all analysis

Both values are documented in the dissertation. All FL and DP evaluations use the corrected dataset.

---

## Known Limitations

1. **AUC tracking during FL simulation:** The `evaluate()` metrics were inaccessible from Flower 1.5.0 simulation history. Training loss is reported as the convergence metric. The correct extraction call (`history.metrics_distributed.get('auc', [])`) is documented in Notebook 02 for future replication.

2. **MIMIC-IV is a US dataset:** Results are indicative of NHS-relevant FL behaviour but require validation on real NHS Trust data.

3. **Google Colab environment:** All simulations ran on Colab CPU. A `protobuf==3.20.3` pin is required to resolve a version conflict with `grpcio-status`.

4. **DP evaluation is single-partition:** DP was evaluated on a single partition rather than the full federated configuration. Cross-client DP evaluation is recommended as future work.

---

## Citation

If you use this code in your research, please cite:

```
Oluwasegun (2026) Federated Learning for Privacy-Preserving NHS EHR:
Simulation Code and Results [online]. GitHub. University of Greater
Manchester MRes Dissertation MRES7015. Available from:
https://github.com/[your-username]/federated-learning-nhs-ehr
[Accessed: insert date].
```

**BibTeX:**
```bibtex
@misc{oluwasegun2026fl,
  author    = {Oluwasegun},
  title     = {Federated Learning for Privacy-Preserving NHS EHR:
               Simulation Code and Results},
  year      = {2026},
  publisher = {GitHub},
  note      = {MRes Dissertation, University of Greater Manchester, MRES7015},
  url       = {https://github.com/[your-username]/federated-learning-nhs-ehr}
}
```

---

## References

- McMahan, H.B. et al. (2017) 'Communication-Efficient Learning of Deep Networks from Decentralized Data', *AISTATS 2017*, PMLR 54. https://doi.org/10.48550/arXiv.1602.05629
- Li, T. et al. (2020) 'Federated Optimization in Heterogeneous Networks', *MLSys 2020*. https://doi.org/10.48550/arXiv.1812.06127
- Karimireddy, S.P. et al. (2020) 'SCAFFOLD: Stochastic Controlled Averaging for Federated Learning', *ICML 2020*, PMLR 119. https://doi.org/10.48550/arXiv.1910.06378
- Fallah, A. et al. (2020) 'Personalized Federated Learning with Theoretical Guarantees', *NeurIPS 2020*. https://doi.org/10.48550/arXiv.2002.07948
- Abadi, M. et al. (2016) 'Deep Learning with Differential Privacy', *ACM CCS 2016*. https://doi.org/10.1145/2976749.2978318
- Shokri, R. et al. (2017) 'Membership Inference Attacks Against Machine Learning Models', *IEEE S&P 2017*. https://doi.org/10.1109/SP.2017.41
- Beutel, D.J. et al. (2022) 'Flower: A Friendly Federated Learning Research Framework'. https://doi.org/10.48550/arXiv.2007.14390
- Hsieh, K. et al. (2020) 'The Non-IID Data Quagmire of Decentralized Machine Learning', *ICML 2020*. https://doi.org/10.48550/arXiv.1910.00189

---

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

The MIMIC-IV dataset is subject to PhysioNet's data use agreement and is **not** included or redistributable under this licence.

---

**University of Greater Manchester · MRes Computing and Artificial Intelligence · MRES7015 · 2026**
