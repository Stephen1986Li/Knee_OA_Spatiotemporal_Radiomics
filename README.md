# Spatiotemporal Radiomics Pipeline for Knee Osteoarthritis Progression

[![Academic-Specification](https://img.shields.io/badge/Academic-Nature%20Medicine%20Standard-blue.svg)](https://github.com/Stephen1986Li/Knee_OA_Spatiotemporal_Radiomics)
[![Python-Version](https://img.shields.io/badge/Python-3.8%2B-green.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

## 📌 Overview
This repository hosts the official standard English analytic pipeline for our multi-center study on **Knee Osteoarthritis (OA)** progression. We developed a novel spatiotemporal radiomics framework to decode the microstructural "silent window" preceding terminal joint failure. 

By leveraging a retrospective countdown design and a **-24-month Sentinel Window**, this pipeline identifies early structural bio-signatures and evaluates the therapeutic potential of pharmacological interventions (e.g., Statins) in mitigating OA progression.

---

## 📂 Repository Structure
The codebase is organized into three core modules, representing the sequential logic of the manuscript:

### 1. Discovery Phase (`/1_discovery_phase`)
* **`1_1_cohort_demographics.py`**: Automated Table 1 generator for baseline population characteristics across derivation and validation cohorts. 
* **`1_2_sentinel_apex_isolation.py`**: Executes the retrospective chronological sweep to isolate the optimal prognostic apex (-24m Sentinel Window). 
* **`1_3_pharmacological_medwas.py`**: High-throughput Medication-Wide Association Study (MedWAS) to screen protective clinical drug exposures.

### 2. Refinement Phase (`/2_refinement_phase`)
* **`2_1_sentinel_shap_decoding.py`**: Decodes local structural risk vectors using regularized PHReg and SHAP value importance.
* **`2_2_hierarchical_clustering.py`**: Evaluates spatial redundancy and feature colinearity within the Sentinel Window.
* **`2_3_clinico_radiomic_heatmap.py`**: Spearman rank correlation mapping between microstructural evolution and WOMAC pain progression.
* **`2_4_quadrant_mapping.py`**: Generates the Phenotypic Quadrant Map to categorize structural vs. symptomatic OA subtypes.

### 3. Mechanism & Validation (`/3_mechanism_and_validation`)
* **`3_1_statin_dose_response.py`**: Evaluates the Spearman slope between Statin therapy duration and structural preservation.
* **`3_2_simple_mediation_structural.py`**: Classic mediation analysis investigating the impact of radiomics on terminal joint failure risk.
* **`3_3_simple_mediation_pain.py`**: Causal chain analysis linking structural changes to longitudinal symptomatic progression.
* **`3_4_sequential_mediation_chain.py`**: Multi-stage sequential mediation analytics unlocking structural-to-symptomatic cascade mechanisms.
* **`4_1_longitudinal_gee_trajectory.py`**: 8-year retrospective trajectory tracing using Generalized Estimating Equations (GEE) with interaction terms.
* **`5_1_single_cohort_validation.py`**: Single-center internal validation metrics computing standard AUC, DCA, and calibration offsets.
* **`5_2_multicenter_split_validation.py`**: External blind validation on the locked Site C cohort to establish geographic generalization.

---

## 🚀 Key Methodological Features
* **Sentinel Window Alignment**: A rigorous "Hard-Alignment" protocol to synchronize multi-center timepoints relative to the joint failure event.
* **Unpolluted Validation**: Data standardizers are fit exclusively on the derivation pool and applied to the locked validation site to prevent information leakage.
* **Mechanism-Driven**: Sequential and simple mediation models to provide biological plausibility for pharmacological efficacy.

---

## 🛠️ Installation & Usage
To replicate the environment and execute the scripts:

1. **Clone the repository**:
   ```bash
   git clone [https://github.com/Stephen1986Li/Knee_OA_Spatiotemporal_Radiomics.git](https://github.com/Stephen1986Li/Knee_OA_Spatiotemporal_Radiomics.git)
   cd Knee_OA_Spatiotemporal_Radiomics
2. Install dependencies:
   pip install -r requirements.txt
3. Run analysis:
   Scripts are intended to be run sequentially from Phase 1 to Phase 3.

📧 Contact
Shengfa Li, MD, PhD Corresponding Author Arthroplasty and Sport Medicine Department,
Chengdu Third People's Hospital
Email: 759529552@qq.com

📄 License
This project is licensed under the MIT License.
