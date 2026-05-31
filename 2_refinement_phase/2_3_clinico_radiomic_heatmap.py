import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
import os

# ==============================================================================
# 1. ENVIRONMENT ENVIRONMENT SETUP & STORAGE MATRIX COUPLING
# ==============================================================================
base_dir = r'C:\在研课题\radiomics_time_window\2026-05-12'
shap_data_path = os.path.join(base_dir, 'Phase_2_Refinement_Retro', 'Table_S4_Sentinel_SHAP.csv')
raw_data_path = os.path.join(base_dir, 'final_merged_cleaned_v13.csv')
output_dir = os.path.join(base_dir, 'Phase_2_Refinement_Retro')

if not os.path.exists(output_dir): 
    os.makedirs(output_dir)

print("⏳ Executing non-parametric Spearman rank correlation loops across locked temporal windows...")
df_shap = pd.read_csv(shap_data_path)
df_raw = pd.read_csv(raw_data_path, low_memory=False)
clinical_var = 'womkp'

# Isolate Top 20 metrics ranked via global cumulative SHAP parameters
top_features_list = df_shap.groupby('Feature')['SHAP'].apply(lambda x: x.abs().mean()).sort_values(ascending=False).head(20).index.tolist()
feature_mapping = df_shap[df_shap['Feature'].isin(top_features_list)][['Feature', 'Raw_ID']].drop_duplicates('Feature')
raw_ids = feature_mapping['Raw_ID'].tolist()

# Locking backward discrete timeline endpoints (-24m vs Baseline)
df_v00 = df_raw[df_raw['visit'] == 'V00'][['id1', clinical_var] + raw_ids].drop_duplicates('id1')
df_sentinel = df_raw[df_raw['years_to_event'] == -2][['id1', clinical_var] + raw_ids].drop_duplicates('id1')

df_m = pd.merge(df_v00, df_sentinel, on='id1', suffixes=('_v00', '_retro24')).dropna()

# Rapid block dictionary setup to isolate macroscopic clinical symptom metrics
delta_dict = {}
delta_dict['Delta_WOMAC_Pain'] = df_m[f'{clinical_var}_retro24'] - df_m[f'{clinical_var}_v00']

for _, row in feature_mapping.iterrows():
    f_name = row['Feature']
    r_id = row['Raw_ID']
    delta_dict[f_name] = df_m[f"{r_id}_retro24"] - df_m[f"{r_id}_v00"]

delta_matrix = pd.DataFrame(delta_dict, index=df_m.index)

s6_records = []
for _, row in feature_mapping.iterrows():
    feat_name = row['Feature']
    rho, p_val = spearmanr(delta_matrix[feat_name], delta_matrix['Delta_WOMAC_Pain'])
    s6_records.append({
        'Feature': feat_name,
        'Spearman_rho': rho, 
        'P_value': p_val,
        'N': len(delta_matrix)
    })

df_s6 = pd.DataFrame(s6_records)
s6_out_path = os.path.join(output_dir, 'Table_S6_Clinico_Radiomic_Correlation.csv')
df_s6.to_csv(s6_out_path, index=False)
print(f"✅ Table S6 Clinico-Radiomic metrics successfully outputted: {s6_out_path}")

# ==============================================================================
# 4. RENDERING CLINICAL PROGRESSION HEATMAP
# ==============================================================================
plot_matrix = df_s6.set_index('Feature')[['Spearman_rho']]

plt.figure(figsize=(10, 12), dpi=300)
sns.set_style("white")

ax = sns.heatmap(plot_matrix, 
                 cmap='RdBu_r', 
                 annot=True, 
                 fmt=".3f", 
                 vmin=-0.4, vmax=0.4, 
                 linewidths=1.2, 
                 cbar_kws={"label": "Spearman's Rho (ρ)"},
                 annot_kws={"size": 10, "weight": "bold"})

# Vertical linear significance asterisk placement
for i, feat in enumerate(plot_matrix.index):
    p = df_s6[df_s6['Feature'] == feat]['P_value'].values[0]
    if p < 0.05:
        marker = '*' if p >= 0.01 else '**'
        if p < 0.001: 
            marker = '***'
        plt.text(1.12, i + 0.5, marker, va='center', ha='left', color='red', fontsize=14, fontweight='bold')

plt.ylabel('Top 20 Sentinel Biomarkers (SHAP Ranked)', fontweight='bold', fontsize=12)

note_text = ("Figure S6: Clinico-radiomic association within the exact Sentinel Window.\n"
             "Heatmap shows Spearman's rank correlation coefficient (ρ) between delta-radiomics (-24m vs Baseline)\n"
             "and retrospective pain progression via unpolluted Hard Alignment. (*p<0.05, **p<0.01, ***p<0.001).")
plt.figtext(0.5, 0.02, note_text, wrap=True, horizontalalignment='center', fontsize=10.5, style='italic', fontweight='bold')

plt.subplots_adjust(bottom=0.15)
plt.savefig(os.path.join(output_dir, 'Figure_S6_Regional_SHAP_Hard.png'), bbox_inches='tight')
plt.close()

print(f"🏁 [Success: Phase 2 Heatmap Correlation Metrics Pipeline Finished]")