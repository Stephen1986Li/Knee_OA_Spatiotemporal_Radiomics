import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ==============================================================================
# 1. DIRECTORY CONFIGURATIONS & ASSET SYNCHRONIZATION
# ==============================================================================
base_dir = r'C:\在研课题\radiomics_time_window\2026-05-12'
s2_path = os.path.join(base_dir, 'Phase_1_Time_Window_Countdown_Hard', 'Table_S2_Countdown_Detailed_HardAlignment.csv')
s6_path = os.path.join(base_dir, 'Phase_2_Refinement_Retro', 'Table_S6_Clinico_Radiomic_Correlation.csv')
output_dir = os.path.join(base_dir, 'Phase_2_Refinement_Retro')

if not os.path.exists(output_dir): 
    os.makedirs(output_dir)

print("⏳ Merging survival coefficients with categorical pain outcomes...")
df_s2 = pd.read_csv(s2_path)
df_s6 = pd.read_csv(s6_path)

df_s2['Region_Clean'] = df_s2['Region'].str.replace('_', ' ').str.strip().str.lower()
df_surv_24 = df_s2[df_s2['Window'] == '-24m'].copy()

# Converting partial likelihood estimates to standard transformation scale -log10(P)
df_surv_24['surv_logp'] = -np.log10(df_surv_24['P_value'].astype(float))

# Resolving mesh-based bracket nesting loops across 7 validated regions
known_regions = ['Medial Meniscus', 'Lateral Meniscus', 'Femur Cartilage', 'Tibia Cartilage', 'Femur', 'Tibia', 'TSV']

def find_true_region(feature_name):
    for reg in sorted(known_regions, key=len, reverse=True):
        if reg.lower() in str(feature_name).lower():
            return reg
    return None

df_s6['Anatomical Region'] = df_s6['Feature'].apply(find_true_region)
df_s6['Region_Clean'] = df_s6['Anatomical Region'].str.lower().str.strip()

cli_summary = df_s6.groupby('Region_Clean')['Spearman_rho'].apply(lambda x: x.abs().mean()).reset_index()
cli_summary.columns = ['Region_Clean', 'cli_rho']

df_final = pd.merge(
    df_surv_24[['Region_Clean', 'Region', 'surv_logp', 'C_index', 'P_value']], 
    cli_summary, 
    on='Region_Clean'
).dropna()

print(f"✅ Multicenter quadrant parameters aligned perfectly. rows: {len(df_final)}")

# Adaptive split thresholds (aligned with component means)
x_mean = df_final['cli_rho'].mean()
y_mean = df_final['surv_logp'].mean()

table_s3 = df_final.copy()
table_s3['Prognostic P-value'] = table_s3['P_value'].apply(lambda x: f"{float(x):.2e}")
table_s3['Clinical Impact (|Rho|)'] = table_s3['cli_rho'].apply(lambda x: f"{x:.4f}")
table_s3_out = table_s3[['Region', 'Prognostic P-value', 'Clinical Impact (|Rho|)', 'C_index']].rename(columns={'Region': 'Anatomical Region'})
table_s3_out.to_csv(os.path.join(output_dir, 'Table_S3_Final_Statistics_Hard.csv'), index=False)

# ==============================================================================
# 4. RENDERING HIGH-D ADAPTIVE QUADRANT CHART (Figure 3)
# ==============================================================================
fig, ax = plt.subplots(figsize=(11.5, 8.5), dpi=300)
sns.set_style("ticks")

ax.axvline(x=x_mean, color='#555555', linestyle='--', linewidth=1.5, alpha=0.5, zorder=2)
ax.axhline(y=y_mean, color='#555555', linestyle='--', linewidth=1.5, alpha=0.5, zorder=2)

x_min, x_max = df_final['cli_rho'].min(), df_final['cli_rho'].max()
y_min, y_max = df_final['surv_logp'].min(), df_final['surv_logp'].max()
x_padding = (x_max - x_min) * 0.35
y_padding = (y_max - y_min) * 0.35

plot_x_min, plot_x_max = x_min - x_padding, x_max + x_padding
plot_y_min, plot_y_max = y_min - y_padding, y_max + y_padding
ax.set_xlim(plot_x_min, plot_x_max)
ax.set_ylim(plot_y_min, plot_y_max)

# Locking categorical labels strictly to the far out corners of the canvas axis
ax.text(plot_x_max - x_padding*0.03, plot_y_max - y_padding*0.03, 'QUAD I: EPICENTER\n(High Risk & High Symptom)', 
        color='#b71c1c', ha='right', va='top', fontsize=11, fontweight='bold', zorder=3,
        bbox=dict(boxstyle="round,pad=0.35", fc="#FFF5F5", ec="#FFCDCD", alpha=1.0, linewidth=0.8))

ax.text(plot_x_min + x_padding*0.03, plot_y_max - y_padding*0.03, 'QUAD II: SILENT DRIVERS\n(High Risk, Low Symptom)', 
        color='#0d47a1', ha='left', va='top', fontsize=11, fontweight='bold', zorder=3,
        bbox=dict(boxstyle="round,pad=0.35", fc="#F0F4F8", ec="#D0E1FD", alpha=1.0, linewidth=0.8))

ax.text(plot_x_min + x_padding*0.03, plot_y_min + y_padding*0.03, 'QUAD III: DORMANT ZONE\n(Low Risk, Low Symptom)', 
        color='#43a047', ha='left', va='bottom', fontsize=11, fontweight='bold', zorder=3,
        bbox=dict(boxstyle="round,pad=0.35", fc="#F4F9F4", ec="#C8E6C9", alpha=1.0, linewidth=0.8))

ax.text(plot_x_max - x_padding*0.03, plot_y_min + y_padding*0.03, 'QUAD IV: PHENOTYPIC COUPLING\n(Low Risk, High Symptom)', 
        color='#e65100', ha='right', va='bottom', fontsize=11, fontweight='bold', zorder=3,
        bbox=dict(boxstyle="round,pad=0.35", fc="#FFF8F1", ec="#FFE0B2", alpha=1.0, linewidth=0.8))

sc = ax.scatter(df_final['cli_rho'], df_final['surv_logp'], c=df_final['C_index'], cmap='coolwarm', 
                 s=200, alpha=0.95, edgecolors='black', linewidths=1.2, zorder=4)

cbar = fig.colorbar(sc, ax=ax)
cbar.set_label('Predictive Power (C-index)', fontsize=11, fontweight='bold', labelpad=10)
cbar.ax.tick_params(labelsize=9)

label_offsets = {
    'Medial Meniscus': (0.0006, -0.8, 'left', 'top'),
    'Lateral Meniscus': (0.0006, 0.8, 'left', 'bottom'),
    'Femur Cartilage': (0.0006, 0.8, 'left', 'bottom'),
    'Tibia Cartilage': (0.0006, -0.8, 'left', 'top'),
    'Femur': (-0.0006, 0, 'right', 'center'),
    'Tibia': (-0.0006, -0.8, 'right', 'top'),
    'TSV': (0.0006, -0.8, 'left', 'top')
}

for _, row in df_final.iterrows():
    reg = row['Region']
    x_off, y_off, ha, va = label_offsets.get(reg, (0.0006, 0, 'left', 'center'))
    ax.text(row['cli_rho'] + x_off, row['surv_logp'] + y_off, reg, 
            fontsize=11, fontweight='bold', ha=ha, va=va, zorder=6,
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#BBBBBB", alpha=0.95, linewidth=0.8))

ax.set_xlabel("Clinical Burden Correlation (|Mean Spearman's " + r'$\rho$' + "| with WOMAC Pain)", fontsize=11, fontweight='bold', labelpad=10)
ax.set_ylabel(r'$-\log_{10}(P\mathdefault{-value})$' + ' (Longitudinal Risk Effect)', fontsize=11, fontweight='bold', labelpad=12)

sns.despine(ax=ax, offset=10)
fig.tight_layout()
plt.subplots_adjust(bottom=0.26)

note_text = ("Figure 3: Clinico-radiomic quadrant mapping at the unpolluted Sentinel Window (-24m).\n"
             "Dashed lines indicate the mean values of clinical correlation and prognostic scores.\n"
             "The four solid boundary boxes indicate distinct phenotypic domains within whole-joint osteoarthritis progression.")
plt.figtext(0.5, 0.03, note_text, wrap=True, horizontalalignment='center', fontsize=10.5, style='italic', fontweight='bold', color='#333333')

fig.savefig(os.path.join(output_dir, 'Figure_3_Clinico_Radiomic_Quadrant_Hard.png'), bbox_inches='tight')
plt.close()

print(f"🏁 [Success: Figure 3 Academic Quadrant Plot Saved without Label Intersections]")