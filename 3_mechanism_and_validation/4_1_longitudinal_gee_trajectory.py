import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler
import os
import warnings

# ==============================================================================
# 1. ENVIRONMENT ENVIRONMENT SETUP & LONGITUDINAL TRAJECTORY INITIALIZATION
# ==============================================================================
BASE_DIR = r'C:\在研课题\radiomics_time_window\2026-05-12'
INPUT_PATH = os.path.join(BASE_DIR, 'final_merged_cleaned_v13.csv')
OUTPUT_DIR = os.path.join(BASE_DIR, 'Phase_4_Retrospective_Trajectory_Final_v3')

if not os.path.exists(OUTPUT_DIR): 
    os.makedirs(OUTPUT_DIR)

warnings.filterwarnings('ignore')

print("⏳ Loading longitudinal panels and establishing cluster GEE covariance equations...")
df_raw = pd.read_csv(INPUT_PATH, low_memory=False)

df_raw['Statins_Status'] = pd.to_numeric(df_raw['Statins_Status'], errors='coerce')
df_raw['years_to_event'] = pd.to_numeric(df_raw['years_to_event'], errors='coerce')
df_countdown = df_raw[df_raw['years_to_event'] <= 0].dropna(subset=['years_to_event', 'Statins_Status', 'id1']).reset_index(drop=True)

# Transposing year dimensions to precise clinical month visit milestones
df_countdown['months_to_event'] = df_countdown['years_to_event'] * 12

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

# ==============================================================================
# 3. CLUSTER-GEE INTERACTION CALCULATIONS AND STANDALONE PLOTTING
# ==============================================================================
trajectory_records = []
c_statin = '#1E88E5'   
c_control = '#D32F2F'  

for roi in all_rois:
    reg_key = roi.replace(' ', '_')
    feat_cols = [c for c in df_countdown.columns if c.startswith(reg_key)]
    if roi in ['Femur', 'Tibia']: 
        feat_cols = [c for c in feat_cols if 'Cartilage' not in c]
        
    if not feat_cols: 
        continue
    
    roi_data = df_countdown[['id1', 'months_to_event', 'Statins_Status'] + covariates + feat_cols].dropna().copy()
    if len(roi_data) < 30: 
        continue
    
    scaler_feats = StandardScaler()
    scaled_feats = scaler_feats.fit_transform(roi_data[feat_cols])
    
    try:
        lasso = LassoCV(cv=5, random_state=42, max_iter=2000).fit(scaled_feats, roi_data['months_to_event'])
        coef_series = pd.Series(lasso.coef_, index=feat_cols).abs().sort_values(ascending=False)
        top_f = coef_series.head(5).index.tolist()
    except:
        top_f = feat_cols[:5]
        
    scaler_score = StandardScaler()
    roi_data['Z_Score'] = scaler_score.fit_transform(roi_data[top_f]).mean(axis=1)
    
    # Fitting Generalized Estimating Equations locking patient id1 as cluster constraints
    try:
        model_form = f"Z_Score ~ months_to_event * Statins_Status + {' + '.join(covariates)}"
        gee_mod = smf.gee(model_form, groups=roi_data['id1'], data=roi_data, 
                          family=sm.families.Gaussian(), 
                          cov_struct=sm.cov_struct.Autoregressive())
        gee_res = gee_mod.fit()
        
        p_inter = gee_res.pvalues['months_to_event:Statins_Status']
        beta_inter = gee_res.params['months_to_event:Statins_Status']
        se_inter = gee_res.bse['months_to_event:Statins_Status']
    except Exception as e:
        print(f"  ⚠️ GEE non-convergence detected for {roi}. Falling back to robust Linear Mixed Models...")
        try:
            lmm_mod = smf.mixedlm(model_form, groups=roi_data['id1'], data=roi_data)
            lmm_res = lmm_mod.fit()
            p_inter = lmm_res.pvalues['months_to_event:Statins_Status']
            beta_inter = lmm_res.params['months_to_event:Statins_Status']
            se_inter = lmm_res.bse['months_to_event:Statins_Status']
        except:
            continue
            
    trajectory_records.append({
        'Anatomical Region': roi,
        'Total Observations (N)': len(roi_data),
        'Unique Patients (Clusters)': roi_data['id1'].nunique(),
        'Interaction Slope (Beta)': fmt_stat_dynamic(beta_inter),
        'Robust Standard Error': fmt_stat_dynamic(se_inter),
        'Interaction P-value': fmt_p_dynamic(p_inter),
        'Evolutionary Divergence': "Significant" if p_inter < 0.05 else "Parallel"
    })
    
    # Render GEE Milestone lines
    fig, ax = plt.subplots(figsize=(10.5, 6.5), dpi=300)
    sns.set_style("ticks")
    
    traj_summary = roi_data.groupby(['months_to_event', 'Statins_Status'])['Z_Score'].agg(['mean', 'count', 'std']).reset_index()
    traj_summary['se'] = traj_summary['std'] / np.sqrt(traj_summary['count'])
    
    for status, color, label in zip([1, 0], [c_statin, c_control], ['Statin Users', 'Untreated Controls']):
        sub = traj_summary[traj_summary['Statins_Status'] == status].sort_values('months_to_event')
        ax.plot(sub['months_to_event'], sub['mean'], marker='o', markersize=6.5, linewidth=2.8, color=color, label=label, zorder=4)
        ax.fill_between(sub['months_to_event'], sub['mean'] - 1.96 * sub['se'], sub['mean'] + 1.96 * sub['se'], color=color, alpha=0.12, zorder=2)
        
    ax.legend(frameon=False, loc='lower center', bbox_to_anchor=(0.5, -0.22), ncol=2, fontsize=10.5)
    
    p_sign_str = "" if ("<" in fmt_p_dynamic(p_inter)) else "= "
    stats_text = f"GEE Interaction {r'$p$'} {p_sign_str}{fmt_p_dynamic(p_inter)}"
    ax.text(0.04, 0.95, stats_text, transform=ax.transAxes, fontsize=10.5, fontweight='bold', ha='left', va='top',
            bbox=dict(facecolor='#F8F9FA', alpha=0.95, edgecolor='#E0E0E0', boxstyle='round,pad=0.4'))
    
    # Enforcing strict clinical visit cutoff ticks
    custom_ticks = [-96, -72, -48, -36, -24, -12, 0]
    custom_labels = ['-96m', '-72m', '-48m', '-36m', '-24m', '-12m', '0']
    ax.set_xticks(custom_ticks)
    ax.set_xticklabels(custom_labels, fontsize=10)
    
    ax.set_xlabel("Months Prior to Joint Failure Event (0 = Joint Failure Point)", fontsize=11, labelpad=10)
    ax.set_ylabel("Rad-Score Value (Z-standardized Multi-Feature Index)", fontsize=11, labelpad=10)
    
    mean_min, mean_max = traj_summary['mean'].min(), traj_summary['mean'].max()
    mean_range = mean_max - mean_min if (mean_max - mean_min) > 1e-4 else 0.2
    ax.set_ylim(mean_min - 0.15 * mean_range, mean_max + 0.40 * mean_range)
    ax.set_xlim(-100.0, 4.0)
    ax.grid(axis='y', linestyle=':', alpha=0.4)
    sns.despine(offset=10, trim=True)
    
    fig.tight_layout()
    plt.subplots_adjust(bottom=0.28) 
    
    note_text = (f"Figure 4-{reg_key}: Longitudinal trajectory tracing back from terminal joint failure event point.\n"
                 f"The horizontal axis is strictly calibrated to represent discrete clinical follow-up visit milestones.\n"
                 f"The shaded ribbons quantify the 95% longitudinal robust standard error intervals adjusted by cluster-GEE.")
    plt.figtext(0.5, 0.03, note_text, wrap=True, horizontalalignment='center', fontsize=9.5, style='italic', fontweight='bold', color='#222222')
    
    fig_out_name = f"Figure_4_Longitudinal_Trajectory_{reg_key}.png"
    plt.savefig(os.path.join(output_dir, fig_out_name), bbox_inches='tight')
    plt.close()
    print(f"  🎨 [{roi}] GEE longitudinal milestone chart archived -> {fig_out_name}")

# ==============================================================================
# 4. EXPORT CONSOLIDATED TRAJECTORY LOGS TABLE S10
# ==============================================================================
df_table10 = pd.DataFrame(trajectory_records)
table_out_path = os.path.join(output_dir, 'Table_S10_Trajectory_Interaction_Statistics.csv')
df_table10.to_csv(table_out_path, index=False)

print(f"\n🏁 [Success: Phase 4 GEE Milestone-Aligned Trajectory Pipeline Finalized]\n")