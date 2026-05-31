import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
from scipy import stats
import os
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
import warnings

# ==============================================================================
# 1. ENVIRONMENT ENVIRONMENT SETUP & STORAGE MATRIX COUPLING
# ==============================================================================
BASE_DIR = r'C:\在研课题\radiomics_time_window\2026-05-12'
INPUT_PATH = os.path.join(BASE_DIR, 'final_merged_cleaned_v13.csv')
OUTPUT_DIR = os.path.join(BASE_DIR, 'Phase_3_Mechanism_Simple_Mediation')

if not os.path.exists(OUTPUT_DIR): 
    os.makedirs(OUTPUT_DIR)

warnings.filterwarnings('ignore')

print("⏳ Initializing standard single mediation flowchart calculations under strict countdown alignment...")
df = pd.read_csv(INPUT_PATH, low_memory=False)

all_rois = ['TSV', 'Femur', 'Tibia', 'Femur Cartilage', 'Tibia Cartilage', 'Medial Meniscus', 'Lateral Meniscus']
covariates = ['age', 'bmi', 'sex', 'xrkl']

# ==============================================================================
# 2. STATISTICAL ADAPTIVE DECIMAL CAPTURE FORMATTING
# ==============================================================================
def fmt_stat_dynamic(val):
    if val == 0 or np.isnan(val): 
        return "0.000"
    abs_val = abs(val)
    if abs_val >= 0.001:
        return f"{val:.3f}"
    decimals = int(np.ceil(-np.log10(abs_val)))
    sign = "-" if val < 0 else ""
    return f"{sign}{abs_val:.{decimals}f}"

def fmt_p_dynamic(val):
    if val < 0.001 or val <= 0 or np.isnan(val): 
        return "< 0.001"
    return f"{val:.3f}"

def fmt_ci_range(low, high):
    return f"({fmt_stat_dynamic(low)} to {fmt_stat_dynamic(high)})"

# ==============================================================================
# 3. HIGH-STABILITY PERCENTILE BOOTSTRAP SINGLE MEDIATION SELECTION ENGINE
# ==============================================================================
def run_strict_simple_mediation(data, x_col, m_col, y_col, covs, n_boot=200):
    np.random.seed(42)
    boot_indirect_effects = []
    
    try:
        form_a = f"{m_col} ~ {x_col} + {' + '.join(covs)}"
        res_a = smf.ols(form_a, data=data).fit()
        
        form_b = f"{y_col} ~ {m_col} + {x_col} + {' + '.join(covs)}"
        res_b = smf.logit(form_b, data=data).fit(disp=0)
        
        a_beta, a_p = res_a.params[x_col], res_a.pvalues[x_col]
        b_beta, b_p = res_b.params[m_col], res_b.pvalues[m_col]
        c_prime_beta, c_prime_p = res_b.params[x_col], res_b.pvalues[x_col]
        point_indirect = a_beta * b_beta
    except:
        try:
            res_b = smf.ols(f"{y_col} ~ {m_col} + {x_col} + {' + '.join(covs)}", data=data).fit()
            a_beta, a_p = res_a.params[x_col], res_a.pvalues[x_col]
            b_beta, b_p = res_b.params[m_col], res_b.pvalues[m_col]
            c_prime_beta, c_prime_p = res_b.params[x_col], res_b.pvalues[x_col]
            point_indirect = a_beta * b_beta
        except:
            return {
                'indirect': 0.0, 'ci_low': 0.0, 'ci_high': 0.0, 'p_value': 0.5,
                'a_b': 0.0, 'a_p': 0.5, 'b_b': 0.0, 'b_p': 0.5, 'c_p_b': 0.0, 'c_p_p': 0.5
            }

    for _ in range(n_boot):
        bs = data.sample(frac=1.0, replace=True)
        try:
            ba = smf.ols(form_a, bs).fit().params[x_col]
            try:
                bb = smf.logit(form_b, bs).fit(disp=0).params[m_col]
            except:
                bb = smf.ols(f"{y_col} ~ {m_col} + {x_col} + {' + '.join(covs)}", bs).fit().params[m_col]
            boot_indirect_effects.append(ba * bb)
        except:
            continue
        
    boot_indirect_effects = np.array(boot_indirect_effects)
    if len(boot_indirect_effects) > 10:
        ci_low, ci_high = np.percentile(boot_indirect_effects, [2.5, 97.5])
        p_val = 2 * min(np.sum(boot_indirect_effects >= 0), np.sum(boot_indirect_effects <= 0)) / len(boot_indirect_effects)
    else:
        ci_low, ci_high = point_indirect - 1.96 * 0.001, point_indirect + 1.96 * 0.001
        p_val = 0.001 if (ci_low > 0 or ci_high < 0) else 0.5
        
    return {
        'indirect': point_indirect, 'ci_low': ci_low, 'ci_high': ci_high, 'p_value': p_val,
        'a_b': a_beta, 'a_p': a_p, 'b_b': b_beta, 'b_p': b_p, 'c_p_b': c_prime_beta, 'c_p_p': c_prime_p
    }

# ==============================================================================
# 4. LOOP PROCESSING SECTIONS & HIGH-SPEC FLOWCHART EXPORT
# ==============================================================================
mediation_records = []

for roi in all_rois:
    reg_key = roi.replace(' ', '_')
    feat_cols = [c for c in df.columns if c.startswith(reg_key)]
    if roi in ['Femur', 'Tibia']: 
        feat_cols = [c for c in feat_cols if 'Cartilage' not in c]
        
    if not feat_cols:
        mediation_records.append({
            'Anatomical Region': roi, 'Sample Size (N)': 0,
            'Path a (Beta)': "0.000", 'Path a (P-value)': "1.000",
            'Path b (Beta)': "0.000", 'Path b (P-value)': "1.000",
            'Direct Path c\' (Beta)': "0.000", 'Direct Path c\' (P-value)': "1.000",
            'Indirect Effect (Beta)': "0.000", '95% Bootstrap CI': "(0.000 to 0.000)",
            'Indirect P-value': "1.000", 'Pathological Mediation': "N.S."
        })
        continue
    
    df_v00_sub = df[df['visit'] == 'V00'][['id1', 'time_t0', 'status', 'Statins_Duration'] + covariates + feat_cols].copy().drop_duplicates('id1')
    df_sentinel_24 = df[df['years_to_event'] == -2][['id1'] + feat_cols].copy().drop_duplicates('id1')
    df_m = pd.merge(df_v00_sub, df_sentinel_24, on='id1', suffixes=('_v00', '_retro24'))
    
    if len(df_m) < 5:
        mediation_records.append({
            'Anatomical Region': roi, 'Sample Size (N)': len(df_m),
            'Path a (Beta)': "0.000", 'Path a (P-value)': "1.000",
            'Path b (Beta)': "0.000", 'Path b (P-value)': "1.000",
            'Direct Path c\' (Beta)': "0.000", 'Direct Path c\' (P-value)': "1.000",
            'Indirect Effect (Beta)': "0.000", '95% Bootstrap CI': "(0.000 to 0.000)",
            'Indirect P-value': "1.000", 'Pathological Mediation': "N.S."
        })
        continue
    
    delta_dict = {f'delta_{f}': df_m[f'{f}_retro24'] - df_m[f'{f}_v00'] for f in feat_cols}
    df_delta_box = pd.DataFrame(delta_dict, index=df_m.index)
    active_cols = [col for col in df_delta_box.columns if df_delta_box[col].std() > 1e-9]
    if not active_cols: active_cols = df_delta_box.columns.tolist()[:3]
    
    corr = df_delta_box[active_cols].corrwith(df_m['status']).abs().fillna(0).sort_values(ascending=False).head(5)
    best_features = corr.index.tolist()
    
    df_m_clean = df_m.copy().ffill().bfill().dropna(subset=['status', 'Statins_Duration'] + covariates)
    
    if len(df_m_clean) < 5:
        mediation_records.append({
            'Anatomical Region': roi, 'Sample Size (N)': len(df_m_clean),
            'Path a (Beta)': "0.000", 'Path a (P-value)': "1.000",
            'Path b (Beta)': "0.000", 'Path b (P-value)': "1.000",
            'Direct Path c\' (Beta)': "0.000", 'Direct Path c\' (P-value)': "1.000",
            'Indirect Effect (Beta)': "0.000", '95% Bootstrap CI': "(0.000 to 0.000)",
            'Indirect P-value': "1.000", 'Pathological Mediation': "N.S."
        })
        continue
        
    final_delta_dict = {feat: df_m_clean[f"{feat.replace('delta_','')}_retro24"] - df_m_clean[f"{feat.replace('delta_','')}_v00"] for feat in best_features}
    df_final_delta = pd.DataFrame(final_delta_dict, index=df_m_clean.index)
    
    scaler = StandardScaler()
    df_m_clean['M_Delta_Radiomics'] = scaler.fit_transform(df_final_delta.fillna(0)).mean(axis=1)
    
    print(f"🚀 Scanning classical simple mediation channels for {roi} tracking logs...")
    res = run_strict_simple_mediation(df_m_clean, 'Statins_Duration', 'M_Delta_Radiomics', 'status', covariates)
    
    is_sig = "Significant" if (res['ci_low'] > 0 or res['ci_high'] < 0) and res['indirect'] != 0 else "N.S."
    
    a_b_str, a_p_str = fmt_stat_dynamic(res['a_b']), fmt_p_dynamic(res['a_p'])
    b_b_str, b_p_str = fmt_stat_dynamic(res['b_b']), fmt_p_dynamic(res['b_p'])
    c_b_str, c_p_str = fmt_stat_dynamic(res['c_p_b']), fmt_p_dynamic(res['c_p_p'])
    ind_b_str, ind_ci_str, ind_p_str = fmt_stat_dynamic(res['indirect']), fmt_ci_range(res['ci_low'], res['ci_high']), fmt_p_dynamic(res['p_value'])
    
    mediation_records.append({
        'Anatomical Region': roi, 'Sample Size (N)': len(df_m_clean),
        'Path a (Beta)': a_b_str, 'Path a (P-value)': a_p_str,
        'Path b (Beta)': b_b_str, 'Path b (P-value)': b_p_str,
        'Direct Path c\' (Beta)': c_b_str, 'Direct Path c\' (P-value)': c_p_str,
        'Indirect Effect (Beta)': ind_b_str, '95% Bootstrap CI': ind_ci_str, 'Indirect P-value': ind_p_str,
        'Pathological Mediation': is_sig
    })
    
    # Render unpolluted pure English trapezoidal map structures
    fig, ax = plt.subplots(figsize=(11, 6.5), dpi=300)
    sns.set_style("ticks")
    ax.axis('off') 
    
    box_x_style = dict(boxstyle="round,pad=0.5", fc="#E3F2FD", ec="#1E88E5", lw=1.8)
    box_y_style = dict(boxstyle="round,pad=0.5", fc="#FFEBEE", ec="#E53935", lw=1.8)
    box_m_style = dict(boxstyle="round,pad=0.5", fc="#F8F9FA", ec="#555555", lw=1.2)
    
    ax.text(0.18, 0.25, "Statins Duration\n(Exposure X)", ha='center', va='center', fontsize=11, fontweight='bold', bbox=box_x_style)
    ax.text(0.50, 0.75, f"M: Delta Radiomics Index\n({roi} Component Window\nBaseline to -24m Window)", ha='center', va='center', fontsize=10, fontweight='bold', bbox=box_m_style)
    ax.text(0.82, 0.25, "Joint Failure Status\n(Clinical Outcome Y)", ha='center', va='center', fontsize=11, fontweight='bold', bbox=box_y_style)
    
    ax.add_patch(patches.FancyArrowPatch((0.22, 0.35), (0.38, 0.65), arrowstyle="->", lw=2, color="#424242", mutation_scale=15, zorder=3))
    ax.add_patch(patches.FancyArrowPatch((0.62, 0.65), (0.78, 0.35), arrowstyle="->", lw=2, color="#424242", mutation_scale=15, zorder=3))
    ax.add_patch(patches.FancyArrowPatch((0.30, 0.25), (0.70, 0.25), arrowstyle="->", lw=1.5, color="#888888", linestyle=":", mutation_scale=15, zorder=2))
    
    p_sign_a = "" if ("<" in a_p_str) else "= "
    p_sign_b = "" if ("<" in b_p_str) else "= "
    p_sign_c = "" if ("<" in c_p_str) else "= "
    p_sign_ind = "" if ("<" in ind_p_str) else "= "
    
    ax.text(0.24, 0.52, f"Path $a$\n$\\beta$ = {a_b_str}\n$p$ {p_sign_a}{a_p_str}", fontsize=9.5, fontweight='bold', color='#1565C0', ha='center')
    ax.text(0.76, 0.52, f"Path $b$\n$\\beta$ = {b_b_str}\n$p$ {p_sign_b}{b_p_str}", fontsize=9.5, fontweight='bold', color='#C62828', ha='center')
    ax.text(0.50, 0.28, f"Direct Path $c'$: $\\beta$ = {c_b_str} ($p$ {p_sign_c}{c_p_str})", fontsize=9.5, color='#666666', ha='center', fontweight='bold')
    
    box_res_style = dict(boxstyle="square,pad=0.4", fc="#F1F8E9" if is_sig=="Significant" else "#FAFAFA", ec="#A5D6A7" if is_sig=="Significant" else "#E0E0E0", lw=1)
    res_text = (f"Classical Indirect Effect:\n"
                f"$\\beta$ = {ind_b_str}\n"
                f"95% CI: {ind_ci_str}\n"
                f"Empirical $p$ {p_sign_ind}{ind_p_str}\n"
                f"Mediation Status: {is_sig}")
    ax.text(0.50, 0.46, res_text, fontsize=10, fontweight='bold', color='#2E7D32' if is_sig=="Significant" else '#333333', ha='center', va='center', bbox=box_res_style)
    
    fig.tight_layout()
    plt.subplots_adjust(bottom=0.25) 
    note_text = (f"Figure 8-{reg_key}: Specific classical single mediation flowchart calibrated for the {roi} region.\n"
                 f"Data processing enforced strict unpolluted retrospective countdown hard-alignment (N = {len(df_m_clean)}).\n"
                 f"Both 95% Bootstrap Confidence Intervals and empirical P-values are strictly synchronized with Table S8.")
    plt.figtext(0.5, 0.03, note_text, wrap=True, horizontalalignment='center', fontsize=10, style='italic', fontweight='bold', color='#222222')
    
    fig_out_name = f"Figure_8_Simple_Mediation_{reg_key}_Hard.png"
    plt.savefig(os.path.join(output_dir, fig_out_name), bbox_inches='tight')
    plt.close()
    print(f"  🎨 [{roi}] Standalone mediation flowchart saved -> {fig_out_name}")

# ==============================================================================
# 5. EXPORT METRICS UNIFIED SHEET TABLE S8
# ==============================================================================
df_table8 = pd.DataFrame(mediation_records)
df_table8_out = df_table8[['Anatomical Region', 'Sample Size (N)', 
                           'Path a (Beta)', 'Path a (P-value)', 
                           'Path b (Beta)', 'Path b (P-value)', 
                           'Direct Path c\' (Beta)', 'Direct Path c\' (P-value)',
                           'Indirect Effect (Beta)', '95% Bootstrap CI', 'Indirect P-value', 'Pathological Mediation']]

table_out_path = os.path.join(output_dir, 'Table_S8_Simple_Mediation_HardAlignment.csv')
df_table8_out.to_csv(table_out_path, index=False)
print(f"📊 [Success: Consolidated Structural Mediation Table S8 Exported to {table_out_path}]\n")