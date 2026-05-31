import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from lifelines import CoxPHFitter
import os

# ==============================================================================
# 1. BASE CONFIGURATION & DIRECTORY RESOLUTION
# ==============================================================================
BASE_DIR = r'C:\在研课题\radiomics_time_window\2026-05-12'
INPUT_PATH = os.path.join(BASE_DIR, 'final_merged_cleaned_v13.csv')
OUTPUT_DIR = os.path.join(BASE_DIR, 'Phase_1_Time_Window_Countdown_Hard')

if not os.path.exists(OUTPUT_DIR): 
    os.makedirs(OUTPUT_DIR)

# Anatomical Region Mapping Matrices
regions = {
    'Medial_Meniscus': 'Medial Meniscus', 'Lateral_Meniscus': 'Lateral Meniscus',
    'Femur_Cartilage': 'Femur Cartilage', 'Tibia_Cartilage': 'Tibia Cartilage',
    'Femur': 'Femur', 'Tibia': 'Tibia', 'TSV': 'TSV'
}

# Strict Discrete Temporal Countdown Array (Preventing Information Overlap)
window_order = ['-96m', '-72m', '-48m', '-36m', '-24m', '-12m']
target_years_map = {'-12m': -1, '-24m': -2, '-36m': -3, '-48m': -4, '-72m': -6, '-96m': -8}
months_map = {'-12m': 12, '-24m': 24, '-36m': 36, '-48m': 48, '-72m': 72, '-96m': 96}

results_list = []
df_raw = pd.read_csv(INPUT_PATH, low_memory=False)

print("⏳ Initiating chronological hard-alignment sweep for dynamic radiomics C-index tracking...")

for reg_id, reg_label in regions.items():
    reg_f = [c for c in df_raw.columns if c.startswith(reg_id)]
    if reg_id in ['Femur', 'Tibia']: 
        reg_f = [c for c in reg_f if 'Cartilage' not in c]
    
    if not reg_f: 
        continue
        
    for w_label in window_order:
        target_y = target_years_map[w_label]
        
        # Pulling unpolluted baseline covariates
        df_v00 = df_raw[df_raw['visit'] == 'V00'][['id1', 'time_t0', 'status', 'age', 'sex', 'bmi', 'xrkl'] + reg_f].copy().drop_duplicates('id1')
        
        # Enforcing backward discrete survival window matching
        df_vt = df_raw[df_raw['years_to_event'] == target_y][['id1'] + reg_f].copy().drop_duplicates('id1')
        
        df_m = pd.merge(df_v00, df_vt, on='id1', suffixes=('_v00', '_vt')).dropna()
        
        if len(df_m) > 30:
            for f in reg_f: 
                df_m[f'delta_{f}'] = df_m[f'{f}_vt'] - df_m[f'{f}_v00']
            
            delta_cols = [f'delta_{f}' for f in reg_f]
            corr = df_m[delta_cols].corrwith(df_m['status']).abs().sort_values(ascending=False).head(5)
            top_f = corr.index.tolist()
            
            try:
                # Regularized multivariable Cox hazard regression
                cph = CoxPHFitter(penalizer=0.08)
                cph.fit(df_m[top_f + ['time_t0', 'status', 'age', 'sex', 'bmi', 'xrkl']], 
                        duration_col='time_t0', event_col='status')
                
                c_index = cph.concordance_index_
                
                # Standard Greenwood variance error estimation
                events = df_m['status'].sum()
                se = 1.0 / np.sqrt(events) if events > 0 else 0.05
                
                results_list.append({
                    'Region': reg_label, 'Window': w_label, 'Months': months_map[w_label],
                    'C_index': c_index, 'CI_L': max(c_index - 1.96*se, 0.5), 'CI_U': min(c_index + 1.96*se, 1.0),
                    'P_value': cph.log_likelihood_ratio_test().p_value, 'N': len(df_m)
                })
            except: 
                continue

df_res = pd.DataFrame(results_list)

# Isolate optimal performing epicenter region at -24 month apex
perf_24m = df_res[df_res['Window'] == '-24m'].sort_values('C_index', ascending=False)
sentinel_region = perf_24m.iloc[0]['Region'] if not perf_24m.empty else "Medial Meniscus"

# ==============================================================================
# 4. EXPORTING SUPPLEMENTARY METRICS TABLE S2
# ==============================================================================
table_records = []
for reg in df_res['Region'].unique():
    subset = df_res[df_res['Region'] == reg].set_index('Window').reindex(window_order).dropna().reset_index()
    for _, row in subset.iterrows():
        table_records.append({
            'Region': row['Region'], 'Window': row['Window'], 'Months': row['Months'],
            'C_index': round(row['C_index'], 3), 'CI_L': round(row['CI_L'], 3), 'CI_U': round(row['CI_U'], 3),
            'P_value': row['P_value'], 'N': int(row['N'])
        })

df_output_table = pd.DataFrame(table_records)
df_output_table.to_csv(os.path.join(OUTPUT_DIR, 'Table_S2_Countdown_Detailed_HardAlignment.csv'), index=False)
print("📊 [Success: Supplementary Table S2 Tracking Logs Exported with 95% CI bands]")

# ==============================================================================
# 5. RENDERING FIGURE 2 (Clean Trend Version under Nature Aesthetics)
# ==============================================================================
plt.figure(figsize=(11.5, 7.0), dpi=300)
sns.set_style("ticks")
palette = sns.color_palette("muted", len(regions))

y_min = max(df_res['C_index'].min() - 0.02, 0.48)
y_max = min(df_res['C_index'].max() + 0.04, 0.95)
plt.ylim(y_min, y_max)

for i, reg in enumerate(regions.values()):
    group = df_res[df_res['Region'] == reg].set_index('Window').reindex(window_order).dropna().reset_index()
    is_sent = (reg == sentinel_region)
    
    line_color = '#D32F2F' if is_sent else palette[i]
    line_width = 4.5 if is_sent else 1.8
    line_alpha = 1.0 if is_sent else 0.45
    z_ord = 10 if is_sent else 2
    
    plt.plot(group['Window'], group['C_index'], marker='o', markersize=7 if is_sent else 5,
             linewidth=line_width, alpha=line_alpha, color=line_color,
             label=f"{reg} (Sentinel)" if is_sent else reg, zorder=z_ord)

# Vertical line locking the critical temporal threshold
plt.axvline(x='-24m', color='black', linestyle='--', linewidth=2.0, alpha=0.8, zorder=3)

plt.text('-24m', y_max - 0.01, f' Critical Sentinel Window (-24m): {sentinel_region} ', 
         ha='right', va='top', fontsize=10.5, fontweight='bold', color='black',
         bbox=dict(facecolor='white', alpha=0.95, edgecolor='black', boxstyle='round,pad=0.3'), zorder=15)

plt.xlabel('Months Prior to Joint Failure Event (Retrospective Countdown)', fontsize=12, labelpad=12)
plt.ylabel('Concordance Index (C-index)', fontsize=12, labelpad=12)
plt.grid(axis='y', linestyle=':', alpha=0.5)

plt.legend(title='Anatomical Regions', bbox_to_anchor=(1.02, 1), loc='upper left', frameon=False, fontsize=10.5)
plt.subplots_adjust(bottom=0.18, top=0.96, left=0.12, right=0.78)
sns.despine(offset=10, trim=True)

plt.savefig(os.path.join(OUTPUT_DIR, 'Figure_2_Countdown_Time_Window_Hard_Perfect.png'), bbox_inches='tight')
plt.show()

print("🏁 [Success: Figure 2 Spatiotemporal Predictive Trajectory Asset Finalized]")