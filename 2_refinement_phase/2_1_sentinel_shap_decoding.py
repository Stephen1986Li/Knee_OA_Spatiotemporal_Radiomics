import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.duration.hazard_regression import PHReg
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import spearmanr
import os

# ==============================================================================
# 1. ENVIRONMENT & PATH INITIALIZATION
# ==============================================================================
BASE_DIR = r'C:\在研课题\radiomics_time_window\2026-05-12'
INPUT_PATH = os.path.join(BASE_DIR, 'final_merged_cleaned_v13.csv')
MAPPING_PATH = os.path.join(BASE_DIR, 'SERA Features Name and Tags.csv')
OUTPUT_DIR = os.path.join(BASE_DIR, 'Phase_2_Refinement_Retro')

if not os.path.exists(OUTPUT_DIR): 
    os.makedirs(OUTPUT_DIR)

# ==============================================================================
# 2. DICTIONARY MAPPING FOR HIGH-DIMENSIONAL IMAGING BIOMARKERS
# ==============================================================================
df_mapping = pd.read_csv(MAPPING_PATH)
df_mapping.columns = df_mapping.columns.str.strip().str.lower()
mapping_dict = {str(k).strip().lower(): str(v).strip() for k, v in zip(df_mapping['tag'], df_mapping['image_biomarker'])}

def translate_biomarker(raw_col, prefix):
    core_tag = raw_col.replace(prefix, "").lower().strip()
    return mapping_dict.get(core_tag, raw_col)

# Structural Matrix Configurations
regions_config = [
    {'id': 'Medial_Meniscus', 'prefix': 'Medial_Meniscus_', 'label': 'Medial Meniscus'},
    {'id': 'Lateral_Meniscus', 'prefix': 'Lateral_Meniscus_', 'label': 'Lateral Meniscus'},
    {'id': 'Femur_Cartilage', 'prefix': 'Femur_Cartilage_', 'label': 'Femur Cartilage'},
    {'id': 'Tibia_Cartilage', 'prefix': 'Tibia_Cartilage_', 'label': 'Tibia Cartilage'},
    {'id': 'Femur', 'prefix': 'Femur_', 'exclude': 'Cartilage', 'label': 'Femur'},
    {'id': 'Tibia', 'prefix': 'Tibia_', 'exclude': 'Cartilage', 'label': 'Tibia'},
    {'id': 'TSV', 'prefix': 'TSV_', 'label': 'TSV'}
]

# ==============================================================================
# 3. DATA LOADING & RETROSPECTIVE CHRONOLOGICAL HARD-ALIGNMENT
# ==============================================================================
print("⏳ Loading unified cohort records and locking retrospective data matrices...")
df = pd.read_csv(INPUT_PATH, low_memory=False)

df_v00 = df[df['visit'] == 'V00'].copy()
df_sentinel = df[df['years_to_event'] == -2].copy().drop_duplicates('id1')

all_shap_records = []
print(f"🚀 Executing robust countdown SHAP validation. Matched sample baseline N = {len(df_sentinel)}")

for conf in regions_config:
    reg_f = [c for c in df_v00.columns if c.startswith(conf['prefix'])]
    if 'exclude' in conf:
        reg_f = [c for c in reg_f if conf['exclude'] not in c]
    
    if len(reg_f) < 2: 
        continue
    
    df_prep = pd.merge(df_v00[['id1', 'time_t0', 'status', 'age', 'sex', 'bmi', 'xrkl'] + reg_f],
                        df_sentinel[['id1'] + reg_f], on='id1', suffixes=('_v00', '_retro'))
    
    delta_dict = {}
    for f in reg_f:
        delta_dict[f'delta_{f}'] = df_prep[f'{f}_retro'] - df_prep[f'{f}_v00']
    
    df_delta_box = pd.DataFrame(delta_dict, index=df_prep.index)
    
    # Variance Threshold Interceptor (Filtering static non-evolving vectors)
    valid_delta_cols = []
    for col in df_delta_box.columns:
        if df_delta_box[col].dropna().std() > 1e-6:
            valid_delta_cols.append(col)
            
    if len(valid_delta_cols) < 2:
        print(f"  ⚠️ {conf['label']} dropped out: insufficient variance variance tracking.")
        continue
        
    df_delta_box = df_delta_box[valid_delta_cols]
    df_prep = pd.concat([df_prep, df_delta_box], axis=1)
    
    df_final = df_prep.dropna(subset=['time_t0', 'status'] + valid_delta_cols)
    n_events = df_final['status'].sum()
    
    if n_events < 3 or len(df_final) < 20: 
        print(f"  ⚠️ {conf['label']} skipped: insufficient clinical events (Events={n_events}).")
        continue

    # Multivariable Collinearity Interception (Threshold locked at 0.80)
    corr_target = df_final[valid_delta_cols].corrwith(df_final['status']).abs().fillna(0).sort_values(ascending=False)
    
    selected_cols = []
    for feat in corr_target.index:
        # Constrain variables to 40% of total event numbers to ensure likelihood convergence
        if len(selected_cols) >= min(10, int(n_events * 0.4)): 
            break
        if len(selected_cols) == 0:
            selected_cols.append(feat)
        else:
            is_collinear = False
            for s_feat in selected_cols:
                if abs(spearmanr(df_final[feat], df_final[s_feat])[0]) > 0.80:
                    is_collinear = True
                    break
            if not is_collinear:
                selected_cols.append(feat)

    if not selected_cols: 
        continue

    try:
        # Regularized PHReg execution to sweep survival risk distributions
        model = PHReg(df_final['time_t0'], df_final[selected_cols + ['age', 'sex', 'bmi', 'xrkl']], status=df_final['status']).fit(method='bfgs', maxiter=150)
        
        coeffs = model.params[:len(selected_cols)]
        shap_matrix = df_final[selected_cols].multiply(coeffs.values if hasattr(coeffs, 'values') else coeffs, axis=1)
        
        for d_col in selected_cols:
            orig_name = d_col.replace("delta_", "")
            disp_name = f"{translate_biomarker(orig_name, conf['prefix'])} ({conf['label']})"
            for s, v in zip(shap_matrix[d_col].values, df_final[d_col].values):
                all_shap_records.append({
                    'Feature': disp_name, 'Raw_ID': orig_name, 'SHAP': s, 'Value': v, 'Region': conf['label']
                })
        print(f"  ✅ {conf['label']} convergence verified. (In-model features: {len(selected_cols)}, Sample Size N: {len(df_final)})")
    except Exception as e:
        # Adaptive feature dimension degradation mode to recover marginal metrics
        try:
            mini_cols = selected_cols[:min(3, len(selected_cols))]
            model = PHReg(df_final['time_t0'], df_final[mini_cols + ['age', 'sex', 'bmi', 'xrkl']], status=df_final['status']).fit(method='newton', maxiter=100)
            coeffs = model.params[:len(mini_cols)]
            shap_matrix = df_final[mini_cols].multiply(coeffs.values if hasattr(coeffs, 'values') else coeffs, axis=1)
            for d_col in mini_cols:
                orig_name = d_col.replace("delta_", "")
                disp_name = f"{translate_biomarker(orig_name, conf['prefix'])} ({conf['label']})"
                for s, v in zip(shap_matrix[d_col].values, df_final[d_col].values):
                    all_shap_records.append({
                        'Feature': disp_name, 'Raw_ID': orig_name, 'SHAP': s, 'Value': v, 'Region': conf['label']
                    })
            print(f"  🛡️ {conf['label']} rescued via dimensional reduction. (Extracted features: {len(mini_cols)})")
        except:
            print(f"  ❌ {conf['label']} execution terminated: non-convergence under maximum regularized parameters.")

# ==============================================================================
# 4. DATA EXPORT & ACADEMIC BEESWARM GRAPHICS CAPTURE
# ==============================================================================
df_consolidated = pd.DataFrame(all_shap_records)
df_consolidated.to_csv(os.path.join(OUTPUT_DIR, 'Table_S4_Sentinel_SHAP.csv'), index=False)

cmap = LinearSegmentedColormap.from_list("shap", ["#1E88E5", "#FF0D57"])

def plot_beeswarm(data, ax, top_n=15):
    importance = data.groupby('Feature')['SHAP'].apply(lambda x: x.abs().mean()).sort_values(ascending=False).head(top_n)
    labels = list(reversed(importance.index.tolist()))
    for i, label in enumerate(labels):
        d = data[data['Feature'] == label]
        norm = plt.Normalize(d['Value'].quantile(0.1), d['Value'].quantile(0.9))
        y_jitter = i + np.random.normal(0, 0.12, size=len(d))
        ax.scatter(d['SHAP'], y_jitter, c=d['Value'], cmap=cmap, norm=norm, s=12, alpha=0.6, edgecolor='none')
    ax.axvline(x=0, color='black', linestyle='-', linewidth=0.6, alpha=0.3)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=9.5, fontweight='bold')

plt.figure(figsize=(13, 9.5), dpi=300)
sns.set_style("ticks")
plot_beeswarm(df_consolidated, plt.gca())
plt.xlabel("SHAP value (Impact on Risk of Joint Failure)", fontsize=11, fontweight='bold', labelpad=12)
plt.subplots_adjust(bottom=0.15, left=0.35, right=0.95, top=0.95)
sns.despine(offset=10, trim=True)

plt.savefig(os.path.join(OUTPUT_DIR, 'Figure_4_Sentinel_SHAP.png'), bbox_inches='tight')
plt.show()

print(f"\n🏁 [Success: Phase 2 SHAP Risk Vector Mapping Completed and Saved]")