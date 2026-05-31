import pandas as pd
import numpy as np
from scipy.stats import ttest_ind, chi2_contingency
import os
import warnings

# ==============================================================================
# 1. ENVIRONMENT & PATH INITIALIZATION (Targeting V15 Multicenter Dataset)
# ==============================================================================
BASE_DIR = r'C:\在研课题\radiomics_time_window\2026-05-12'
INPUT_PATH = os.path.join(BASE_DIR, 'final_merged_with_scores_v15_sites.csv')
OUTPUT_DIR = os.path.join(BASE_DIR, 'Phase_5_Validation_Final_v4_Split_standalone')

if not os.path.exists(OUTPUT_DIR): 
    os.makedirs(OUTPUT_DIR)

warnings.filterwarnings('ignore')

print("⏳ Loading V15 dataset and initializing baseline demographics statistical engine...")
df = pd.read_csv(INPUT_PATH, low_memory=False)

# Strict string trimming to prevent categorical indexing failures
df['V00SITE'] = df['V00SITE'].astype(str).str.strip()
df['sex'] = df['sex'].astype(str).str.strip()

# Physical split of cohorts (Enforcing data isolation for Site C validation)
df_deriv = df[df['V00SITE'] != 'C'].copy()
df_valid = df[df['V00SITE'] == 'C'].copy()

# ==============================================================================
# 2. HIGH-PRECISION FORMATTING ASSISTANT FUNCTIONS
# ==============================================================================
def fmt_p_value(p):
    if p < 0.001: 
        return "< 0.001"
    return f"{p:.3f}"

table1_records = []

# ==============================================================================
# 3. COMPUTATIONAL LAYER: COMPREHENSIVE RECRUITMENT FREQUENCIES
# ==============================================================================
n_total = len(df)
n_deriv = len(df_deriv)
n_valid = len(df_valid)
table1_records.append({
    'Characteristics': 'Total Patients, N',
    'Total Population (N = 4,229)': f"{n_total}",
    'Derivation Cohort (N = 2,944)': f"{n_deriv}",
    'Validation Cohort (N = 1,285)': f"{n_valid}",
    'P-value': 'Reference'
})

# ==============================================================================
# 4. COMPUTATIONAL LAYER: CONTINUOUS ADJUSTERS (Welch's T-Test Protocol)
# ==============================================================================
cont_vars = [
    ('Age (years)', 'age'),
    ('Body Mass Index (kg/m²)', 'bmi'),
    ('Follow-up Duration (months)', 'time_t0')
]

for label, col in cont_vars:
    v_tot_mean, v_tot_std = df[col].mean(), df[col].std()
    v_der_mean, v_der_std = df_deriv[col].mean(), df_deriv[col].std()
    v_val_mean, v_val_std = df_valid[col].mean(), df_valid[col].std()
    
    # Independent two-sample t-test under unequal variance assumptions
    stat, p_val = ttest_ind(df_deriv[col].dropna(), df_valid[col].dropna(), equal_var=False)
    
    table1_records.append({
        'Characteristics': f"{label}, Mean ± SD",
        'Total Population (N = 4,229)': f"{v_tot_mean:.2f} ± {v_tot_std:.2f}",
        'Derivation Cohort (N = 2,944)': f"{v_der_mean:.2f} ± {v_der_std:.2f}",
        'Validation Cohort (N = 1,285)': f"{v_val_mean:.2f} ± {v_val_std:.2f}",
        'P-value': fmt_p_value(p_val)
    })

# ==============================================================================
# 5. COMPUTATIONAL LAYER: CATEGORICAL PROFILE DISPOSITIONS (Pearson Chi-Square)
# ==============================================================================
cat_vars = [
    ('Biological Sex', 'sex'),
    ('Baseline Kellgren-Lawrence (KL) Grade', 'xrkl'),
    ('Clinical Outcome (End-stage Failure)', 'status')
]

for label, col in cat_vars:
    table1_records.append({
        'Characteristics': f"{label}, n (%)",
        'Total Population (N = 4,229)': '', 'Derivation Cohort (N = 2,944)': '', 'Validation Cohort (N = 1,285)': '', 'P-value': ''
    })
    
    categories = sorted(df[col].dropna().unique())
    contingency_matrix = []
    
    for cat in categories:
        c_der = len(df_deriv[df_deriv[col] == cat])
        c_val = len(df_valid[df_valid[col] == cat])
        contingency_matrix.append([c_der, c_val])
    
    try:
        chi2, p_val, dof, expected = chi2_contingency(contingency_matrix)
        p_str = fmt_p_value(p_val)
    except:
        p_str = "N/A"
        
    for idx, cat in enumerate(categories):
        count_tot = len(df[df[col] == cat])
        count_der = len(df_deriv[df_deriv[col] == cat])
        count_val = len(df_valid[df_valid[col] == cat])
        
        pct_tot = (count_tot / n_total) * 100
        pct_der = (count_der / n_deriv) * 100
        pct_val = (count_val / n_valid) * 100
        
        p_display = p_str if idx == 0 else ""
        
        cat_label = str(cat)
        if col == 'status':
            cat_label = "Joint Failure (Event)" if int(cat) == 1 else "Censored"
            
        table1_records.append({
            'Characteristics': f"  {cat_label}",
            'Total Population (N = 4,229)': f"{count_tot} ({pct_tot:.1f}%)",
            'Derivation Cohort (N = 2,944)': f"{count_der} ({pct_der:.1f}%)",
            'Validation Cohort (N = 1,285)': f"{count_val} ({pct_val:.1f}%)",
            'P-value': p_display
        })

# ==============================================================================
# 6. CSV REPORT EXPORTation & TERMINAL PREVIEW
# ==============================================================================
df_table1 = pd.DataFrame(table1_records)
csv_out_path = os.path.join(OUTPUT_DIR, 'Table_1_Population_Characteristics.csv')
df_table1.to_csv(csv_out_path, index=False)

print("\n🚀 [Success: Cohort Baseline Demographics Table Compiled Perfectly]")
print(f"➡️ Consolidated CSV exported to: {csv_out_path}\n")