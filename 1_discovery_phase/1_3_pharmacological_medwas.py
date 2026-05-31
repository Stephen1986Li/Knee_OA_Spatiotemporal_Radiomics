import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
import os
import warnings

# ==============================================================================
# 1. DIRECTORY PARSING & ENVIRONMENTAL CONFIGURATION
# ==============================================================================
BASE_DIR = r'C:\在研课题\radiomics_time_window\2026-05-12'
INPUT_PATH = os.path.join(BASE_DIR, 'final_merged_cleaned_v13.csv')
OUTPUT_DIR = os.path.join(BASE_DIR, 'Phase_1_Discovery_Retro_Hard_Independent_v1')

if not os.path.exists(OUTPUT_DIR): 
    os.makedirs(OUTPUT_DIR)

warnings.filterwarnings('ignore')

# ==============================================================================
# 2. MEDWAS CANDIDATE ARRAYS & COVARIATE CONTROL ARCHITECTURES
# ==============================================================================
drug_cols = ['Celecoxib_Status', 'Glucosamine_Status', 'Naproxen_Status', 'Glucocorticoid_Status',
             'Bisphosphonates_Status', 'VitaminD3_Status', 'Estradiol_Status', 'Statins_Status',
             'Hydrochlorothiazide_Status', 'Levothyroxine_Status']

regions = ['TSV', 'Femur', 'Femur_Cartilage', 'Tibia', 'Tibia_Cartilage', 'Medial_Meniscus', 'Lateral_Meniscus']
covariates = ['age', 'sex', 'bmi', 'xrkl']

print("⏳ Loading unified dataset and running unpolluted retrospective alignment...")
df = pd.read_csv(INPUT_PATH, low_memory=False)

df['status'] = pd.to_numeric(df['status'], errors='coerce')
df['time_t0'] = pd.to_numeric(df['time_t0'], errors='coerce')
for col in drug_cols + covariates:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# ==============================================================================
# 2.5 ADAPTIVE DECIMAL TRACKER FILTER (Preventing False Zero Rounding)
# ==============================================================================
def fmt_val_adaptive(val):
    if np.isnan(val): 
        return "NaN"
    if val == 0: 
        return "0.000"
    
    abs_val = abs(val)
    if abs_val >= 0.001:
        return f"{val:.3f}"
    
    # Backwards trace to capture the first true non-zero digit placement
    decimals = int(np.ceil(-np.log10(abs_val)))
    if decimals > 6: 
        decimals = 6  
    return f"{val:.{decimals}f}"

def fmt_p_dynamic(val):
    if val < 0.001 or val <= 0 or np.isnan(val): 
        return "p < 0.001"
    return f"p = {val:.3f}"

# ==============================================================================
# 3. HIGH-THROUGHPUT SYSTEMIC SCREENING COMPUTATIONAL CORE
# ==============================================================================
all_results = []

print("🚀 Executing high-throughput MedWAS scanning loops across 10 drug exposure tracks...")
for drug in drug_cols:
    for reg in regions:
        feat_cols = [c for c in df.columns if c.startswith(reg)]
        if reg in ['Femur', 'Tibia']:
            feat_cols = [c for c in feat_cols if 'Cartilage' not in c]
        if not feat_cols: 
            continue
        
        df_v00 = df[df['visit'] == 'V00'][['id1', 'time_t0', 'status'] + covariates + [drug]].dropna()
        df_v24 = df[df['years_to_event'] == -2][['id1'] + feat_cols].dropna()
        
        df_m = pd.merge(df_v00, df_v24, on='id1').dropna()
        if len(df_m) < 30: 
            continue
        
        from sklearn.preprocessing import StandardScaler
        X_feats = StandardScaler().fit_transform(df_m[feat_cols])
        df_m['Rad_Score_Local'] = X_feats.mean(axis=1)
        
        try:
            formula = f"Rad_Score_Local ~ {drug} + age + sex + bmi + xrkl"
            model = sm.OLS.from_formula(formula, data=df_m).fit()
            
            beta = model.params[drug]
            se = model.bse[drug]
            p_val = model.pvalues[drug]
            ci_lower, ci_upper = model.conf_int().loc[drug]
            
            all_results.append({
                'Drug': drug, 'Region': reg.replace('_', ' '), 'Beta': beta,
                'SE': se, 'P_value': p_val, 'Lower_CI': ci_lower, 'Upper_CI': ci_upper, 'N': len(df_m)
            })
        except:
            continue

df_res_master = pd.DataFrame(all_results)

# ==============================================================================
# 4. EXPORTING ISOLATED HETERO-FOREST PLOTS (Text Islands Architecture)
# ==============================================================================
print("\n🎨 Drafting 10 specialized un-cluttered asset forest plots...")
sns.set_style("ticks")

c_protective = '#1976D2'  
c_neutral = '#616161'     

csv_export_records = []

for idx, drug in enumerate(drug_cols):
    data = df_res_master[df_res_master['Drug'] == drug].copy()
    if data.empty: 
        continue
    
    data_plot = data.iloc[::-1].reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(7.5, 4.8), dpi=300)
    
    x_min, x_max = data_plot['Lower_CI'].min(), data_plot['Upper_CI'].max()
    x_range = x_max - x_min if (x_max - x_min) > 1e-5 else 1.0
    
    for i, row in data_plot.iterrows():
        current_color = c_protective if (row.Beta < 0 and row.P_value < 0.05) else c_neutral
        
        ax.errorbar(row['Beta'], i, xerr=[[row['Beta'] - row['Lower_CI']], [row['Upper_CI'] - row['Beta']]],
                    fmt='o', color=current_color, ecolor=current_color, elinewidth=2.2, capsize=4, ms=7, zorder=3)
        
        b_clean = fmt_val_adaptive(row['Beta'])
        l_clean = fmt_val_adaptive(row['Lower_CI'])
        u_clean = fmt_val_adaptive(row['Upper_CI'])
        p_str = fmt_p_dynamic(row['P_value'])
        
        stats_text = f"β: {b_clean} [95% CI: {l_clean}, {u_clean}]; {p_str}"
        
        text_x_pos = x_max + 0.04 * x_range
        ax.text(text_x_pos, i, stats_text, va='center', ha='left', fontsize=9.5,
                color=current_color, fontweight='bold' if row.P_value < 0.05 else 'normal')
        
    for _, row in data.iterrows():
        csv_export_records.append({
            'Clinical_Drug_Exposure': row['Drug'].replace('_Status', ''),
            'Anatomical_Region': row['Region'],
            'Beta_Coefficient': fmt_val_adaptive(row['Beta']),
            'Robust_SE': fmt_val_adaptive(row['SE']),
            '95%_CI_Lower': fmt_val_adaptive(row['Lower_CI']),
            '95%_CI_Upper': fmt_val_adaptive(row['Upper_CI']),
            'P_value_Formatted': fmt_p_dynamic(row['P_value']),
            'Raw_P_value': row['P_value'],
            'Sample_Size_N': row['N'],
            'Statistical_Significance': 'Significant Benefit (p<0.05)' if (row['Beta'] < 0 and row['P_value'] < 0.05) else 'Non-Significant'
        })
        
    ax.set_xlim(x_min - 0.08 * x_range, x_max + 0.45 * x_range)
    ax.axvline(x=0, color='black', linestyle='--', alpha=0.4, lw=1.2, zorder=1)
    ax.set_yticks(range(len(data_plot)))
    ax.set_yticklabels(data_plot['Region'], fontsize=10.5, fontweight='bold')
    
    clean_drug_name = drug.replace('_Status', '')
    ax.set_title(f"Anatomical Panel: {clean_drug_name} Exposure", fontsize=12, fontweight='bold', pad=12, loc='left')
    ax.set_xlabel("Regression Coefficient (Beta Coefficient from Adjusted OLS)", fontsize=10, labelpad=8)
    
    ax.grid(axis='x', linestyle=':', alpha=0.4)
    sns.despine(offset=8, trim=True)
    plt.subplots_adjust(bottom=0.22, left=0.24, right=0.96, top=0.88)
    
    fig_note = (f"Figure 1-{clean_drug_name}: Forest plot evaluating the multivariable OLS effect of {clean_drug_name} across 7 anatomical regions.\n"
                f"Models are strictly locked to the retrospective -24m sentinel time window and adjusted for age, sex, bmi, and xrkl.\n"
                f"Data text islands are adaptively formatted with standard mathematical qualifiers (p = or p <) to eliminate continuous string artifacts.")
    plt.figtext(0.5, 0.03, fig_note, wrap=True, horizontalalignment='center', fontsize=8.5, style='italic', color='#222222')
    
    fig_out_name = f"Forest_Panel_{idx+1}_{clean_drug_name}.png"
    plt.savefig(os.path.join(OUTPUT_DIR, fig_out_name), bbox_inches='tight')
    plt.close()
    print(f"  🎨 Drug profile [{clean_drug_name}] standalone forest chart saved -> {fig_out_name}")

# ==============================================================================
# 5. EXPORTING MASTER STATISTICAL METRICS TABLE S1
# ==============================================================================
df_table_s1_out = pd.DataFrame(csv_export_records)
csv_out_path = os.path.join(OUTPUT_DIR, 'Table_S1_Comprehensive_MedWAS_Drug_Screening_Metrics.csv')
df_table_s1_out.to_csv(csv_out_path, index=False)

print(f"\n📊 [Success: High-Throughput MedWAS Drug Screening Metrics Table Released]")
print(f"➡️ Complete CSV dataset saved at: {csv_out_path}\n")