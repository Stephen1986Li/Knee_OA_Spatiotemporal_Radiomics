import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os

# ==============================================================================
# 1. ENVIRONMENT ENVIRONMENT SETUP & PATH INITIALIZATION
# ==============================================================================
BASE_DIR = r'C:\在研课题\radiomics_time_window\2026-05-12'
INPUT_PATH = os.path.join(BASE_DIR, 'final_merged_cleaned_v13.csv')
OUTPUT_DIR = os.path.join(BASE_DIR, 'Phase_3_Mechanism_Retro_Mediation')

if not os.path.exists(OUTPUT_DIR): 
    os.makedirs(OUTPUT_DIR)

df = pd.read_csv(INPUT_PATH, low_memory=False)

# Define 7 validated anatomical ROIs consistent with prior blocks
regions = ['Medial Meniscus', 'Lateral Meniscus', 'Femur Cartilage', 'Tibia Cartilage', 'Femur', 'Tibia', 'TSV']

print("⏳ Executing Statin chronological dose-response slope screening under retrospective hard alignment...")

dose_results = []

for reg in regions:
    reg_key = reg.replace(' ', '_')
    feat_cols = [c for c in df.columns if c.startswith(reg_key)]
    if reg in ['Femur', 'Tibia']: 
        feat_cols = [c for c in feat_cols if 'Cartilage' not in c]
    
    if not feat_cols: 
        continue
        
    # Extract unpolluted baseline features
    df_v00 = df[df['visit'] == 'V00'][['id1', 'Statins_Status', 'Statins_Duration'] + feat_cols].copy().drop_duplicates('id1')
    
    # Restrospective hard-alignment window lock at the -24m pre-failure apex
    df_retro = df[df['years_to_event'] == -2][['id1'] + feat_cols].copy().drop_duplicates('id1')
    df_merged = pd.merge(df_v00, df_retro, on='id1', suffixes=('_v00', '_retro24')).dropna()
    
    # Vectorized matrix block computation for delta progressions
    delta_dict = {f'delta_{f}': df_merged[f'{f}_retro24'] - df_merged[f'{f}_v00'] for f in feat_cols}
    df_delta_matrix = pd.DataFrame(delta_dict, index=df_merged.index)
    
    df_merged['Progression_Index'] = df_delta_matrix.mean(axis=1)
    
    # Temporal filtering: restrict evaluation exclusively to actively exposed cohorts
    df_users = df_merged[df_merged['Statins_Status'] == 1].dropna(subset=['Statins_Duration', 'Progression_Index'])
    
    if len(df_users) > 15:
        rho, p_val = stats.spearmanr(df_users['Statins_Duration'], df_users['Progression_Index'])
        
        # Robust Fisher Z transformation bound limits
        n = len(df_users)
        se = 1.06 / np.sqrt(n - 3) if n > 3 else 0.1
        z = np.arctanh(rho) if abs(rho) < 1 else np.arctanh(rho * 0.99)
        rho_low, rho_high = np.tanh(z - 1.96*se), np.tanh(z + 1.96*se)
        
        dose_results.append({
            'Anatomical Region': reg,
            'Duration_Rho': rho,
            '95% CI Lower': rho_low,
            '95% CI Upper': rho_high,
            'P_value': p_val,
            'N': n
        })

df_table5 = pd.DataFrame(dose_results)

# ==============================================================================
# 2. METRICS TABLE 5 CONSOLIDATION EXPORT
# ==============================================================================
df_table5_out = df_table5.copy()
df_table5_out['95% CI'] = df_table5_out.apply(lambda x: f"({x['95% CI Lower']:.4f}, {x['95% CI Upper']:.4f})", axis=1)
df_table5_out['P-value'] = df_table5_out['P_value'].apply(lambda x: f"{x:.3f}" if x >= 0.001 else f"{x:.2e}")
df_table5_out = df_table5_out[['Anatomical Region', 'Duration_Rho', '95% CI', 'P-value', 'N']]
df_table5_out.to_csv(os.path.join(OUTPUT_DIR, 'Table_5_Dose_Response_HardAlignment.csv'), index=False)
print("📊 [Success: Standard Table 5 Dose-Response Bottom metrics Logged]")

# ==============================================================================
# 3. RENDERING FIGURE 5 (Academic Standalone Horizontal Bar Panels)
# ==============================================================================
fig, ax = plt.subplots(figsize=(10.5, 6.5), dpi=300)
sns.set_style("ticks")

df_plot = df_table5.sort_values('Duration_Rho', ascending=True)
colors = ['#2E7D32' if (p < 0.05 and r < 0) else '#BDBDBD' for p, r in zip(df_plot['P_value'], df_plot['Duration_Rho'])]

bars = ax.barh(df_plot['Anatomical Region'], df_plot['Duration_Rho'], 
                color=colors, edgecolor='#333333', linewidth=1.0, alpha=0.9)

for i, bar in enumerate(bars):
    p = df_plot.iloc[i]['P_value']
    x_pos = bar.get_width()
    offset = 0.004 if x_pos >= 0 else -0.004
    p_str = f"P = {p:.3f}" if p >= 0.001 else f"P = {p:.2e}"
    
    ax.text(x_pos + offset, bar.get_y() + bar.get_height()/2, 
             p_str, va='center', ha='left' if x_pos >= 0 else 'right', fontsize=10, 
             fontweight='bold' if p < 0.05 else 'normal',
             color='#2E7D32' if (p < 0.05 and x_pos < 0) else '#424242')

ax.axvline(x=0, color='#D32F2F', linestyle='-', linewidth=1.2, alpha=0.7)
ax.set_xlabel("Spearman's Rank Correlation Coefficient (ρ)\n[Negative Correlation Implies Mitigation of Microstructural Progression]", fontsize=11, fontweight='bold', labelpad=10)

x_bound_min = df_plot['Duration_Rho'].min() - 0.05
x_bound_max = max(df_plot['Duration_Rho'].max() + 0.05, 0.05)
ax.set_xlim(x_bound_min, x_bound_max)
sns.despine(ax=ax, offset=10, trim=True)

fig.tight_layout()
plt.subplots_adjust(bottom=0.28, left=0.18, right=0.95, top=0.95)

note_text = ("Figure 5: Dose-response profile of Statin therapy duration against delta-radiomics microstructural changes.\n"
             "All cumulative changes were strictly mapped within the pre-failure Sentinel Window (-24m vs. Baseline).\n"
             "Green horizontal bars denote regions showing statistically robust structural preservation efficiency (*p < 0.05).")
plt.figtext(0.5, 0.03, note_text, wrap=True, horizontalalignment='center', fontsize=10, style='italic', fontweight='bold', color='#222222')

fig.savefig(os.path.join(OUTPUT_DIR, 'Figure_5_Dose_Response_Hard_Perfect.png'), bbox_inches='tight')
plt.close()

print("🏁 [Success: Figure 5 Academic Dose-Response Asset Dispatched]")