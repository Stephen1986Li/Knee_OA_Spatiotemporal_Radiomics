import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
from sklearn.linear_model import LassoCV, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, auc
import os
import warnings

# ==============================================================================
# 1. ENVIRONMENT ENVIRONMENT SETUP & MULTICENTER COHORT DATASET SYNC
# ==============================================================================
BASE_DIR = r'C:\在研课题\radiomics_time_window\2026-05-12'
INPUT_PATH = os.path.join(BASE_DIR, 'final_merged_with_scores_v15_sites.csv')
OUTPUT_CSV_PATH = os.path.join(BASE_DIR, 'final_merged_with_scores_v15_sites.csv')
OUTPUT_DIR = os.path.join(BASE_DIR, 'Phase_5_Validation_Final_v3_Split_standalone')

if not os.path.exists(OUTPUT_DIR): 
    os.makedirs(OUTPUT_DIR)

warnings.filterwarnings('ignore')

print("⏳ Loading unified V15 dataset containing multi-center site tracking logs...")
df_master = pd.read_csv(INPUT_PATH, low_memory=False)

df_master['id1'] = df_master['id1'].astype(str).str.strip()
df_master['V00SITE'] = df_master['V00SITE'].astype(str).str.strip()
df_master['status'] = pd.to_numeric(df_master['status'], errors='coerce')
df_master['time_t0'] = pd.to_numeric(df_master['time_t0'], errors='coerce')

global_delta_cols = [c for c in df_master.columns if c.startswith('delta_') or c.startswith('Δ_')]
if not global_delta_cols:
    keywords = ['morph', 'texture', 'szm', 'rlm', 'glcm', 'glnu', 'volume', 'perc']
    global_delta_cols = [c for c in df_master.columns if any(k in c.lower() for k in keywords) and c not in ['id1', 'V00SITE', 'status', 'time_t0', 'visit', 'age', 'sex', 'bmi', 'xrkl', 'Global_Radiomics_Score']]

covariates = ['age', 'bmi', 'sex', 'xrkl']
thresholds = np.linspace(0.01, 0.55, 100)

print(f"🔍 Autonomous feature capture isolated {len(global_delta_cols)} delta radiomics metrics.")

# Enforcing strict clinical network data splitting to ensure unpolluted validation boundaries
df_deriv_input = df_master[df_master['V00SITE'] != 'C'].copy().dropna(subset=['time_t0', 'status'] + covariates + global_delta_cols)
df_valid_input = df_master[df_master['V00SITE'] == 'C'].copy().dropna(subset=['time_t0', 'status'] + covariates + global_delta_cols)

# 💡 CRITICAL: Standardizer parameters are fit EXCLUSIVELY on derivation clusters to block information leakage
scaler_f = StandardScaler()
scaled_deriv_f = scaler_f.fit_transform(df_deriv_input[global_delta_cols])
scaled_valid_f = scaler_f.transform(df_valid_input[global_delta_cols])

print(f"✅ Derivation Pool (Sites D, B, A, E) balanced sample size N = {scaled_deriv_f.shape[0]}")
print(f"✅ Validation Silo (Geographically locked Site C) baseline sample size N = {scaled_valid_f.shape[0]}")

# ==============================================================================
# 2. STANDARD SCIENTIFIC LATEX & METRIC EXPORT FORMATTERS
# ==============================================================================
def fmt_stat_dynamic(val):
    if val == 0 or np.isnan(val): 
        return "0.000"
    abs_val = abs(val)
    if abs_val >= 0.001: 
        return f"{val:.3f}"
    decimals = int(np.ceil(-np.log10(abs_val)))
    sign = "-" if val < 0 else ""
    return f"{sign}{abs_val:.{min(decimals, 6)}f}"

def fmt_p_latex(val):
    if val < 0.001 or val <= 0 or np.isnan(val): 
        return r"$p < 0.001$"
    return f"$p = {val:.3f}$"

def fmt_p_dynamic(val):
    if val < 0.001 or val <= 0 or np.isnan(val): 
        return "p < 0.001"
    return f"p = {val:.3f}"

def get_net_benefit(pt, scores, y_true):
    y_pred = (scores >= pt).astype(int)
    tp = np.sum((y_pred == 1) & (y_true == 1))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    n = len(y_true)
    if pt == 1.0: 
        return 0.0
    return (tp / n) - (fp / n) * (pt / (1 - pt))

def calculate_calibration(prob, actual, bins=5):
    df_box = pd.DataFrame({'prob': prob, 'actual': actual}).sort_values('prob')
    chunks = np.array_split(df_box, bins)
    return [c['prob'].mean() for c in chunks], [c['actual'].mean() for c in chunks]

def compute_delong_p(actual, prob_A, prob_B):
    from scipy.stats import norm
    pos_mask, neg_mask = (actual == 1), (actual == 0)
    m, n = np.sum(pos_mask), np.sum(neg_mask)
    if m == 0 or n == 0: 
        return 1.0
    pos_A, neg_A = prob_A[pos_mask], prob_A[neg_mask]
    pos_B, neg_B = prob_B[pos_mask], prob_B[neg_mask]
    V10_A = (pos_A[:, None] > neg_A[None, :]).astype(float) + 0.5 * (pos_A[:, None] == neg_A[None, :])
    V01_A = V10_A.mean(axis=0)
    V10_A = V10_A.mean(axis=1)
    V10_B = (pos_B[:, None] > neg_B[None, :]).astype(float) + 0.5 * (pos_B[:, None] == neg_B[None, :])
    V01_B = V10_B.mean(axis=0)
    V10_B = V10_B.mean(axis=1)
    Sigma = np.cov(V10_A, V10_B) / m + np.cov(V01_A, V01_B) / n
    var_diff = Sigma[0, 0] + Sigma[1, 1] - 2 * Sigma[0, 1]
    if var_diff <= 0: 
        return 1.0
    z = (auc(roc_curve(actual, prob_B)[0], roc_curve(actual, prob_B)[1]) - 
         auc(roc_curve(actual, prob_A)[0], roc_curve(actual, prob_A)[1])) / np.sqrt(var_diff)
    return 2 * (1 - norm.cdf(np.abs(z)))

# ==============================================================================
# 3. HIGH-PERFORMANCE LOGISTIC LASSO REDUCTION ARCHITECTURES
# ==============================================================================
print("🚀 Fitting regularized LASSO paths onto derivation pooled coordinates...")
lasso_cv = LassoCV(cv=5, random_state=42, max_iter=5000).fit(scaled_deriv_f, df_deriv_input['status'].values)
best_alpha = lasso_cv.alpha_
alphas_path, coefs_path, _ = Lasso.path(scaled_deriv_f, df_deriv_input['status'].values, max_iter=5000)

coef_series = pd.Series(lasso_cv.coef_, index=global_delta_cols)
top_f = coef_series.abs().sort_values(ascending=False).head(10).index.tolist()

lasso_weights_records = [{'Global Feature Rank': r+1, 'Selected Delta Feature': f.replace('delta_', r'$\Delta\_$'), 'LASSO Coefficient': fmt_stat_dynamic(coef_series[f])} for r, f in enumerate(top_f)]
pd.DataFrame(lasso_weights_records).to_csv(os.path.join(OUTPUT_DIR, 'Table_S12_LASSO_Feature_Weights.csv'), index=False)

sns.set_style("ticks")
fig1, ax1 = plt.subplots(figsize=(6, 4.6), dpi=300)
ax1.plot(lasso_cv.alphas_, lasso_cv.mse_path_.mean(axis=-1), color='#1976D2', lw=2.2, label='Mean CV MSE')
ax1.axvline(lasso_cv.alpha_, color='#D32F2F', linestyle='--', lw=1.6, label=f'Optimal $\\lambda$ ({lasso_cv.alpha_:.4f})')
ax1.set_xscale('log')
ax1.set_xlabel(r'Regularization Parameter ($\alpha$)', fontsize=10.5, labelpad=6)
ax1.set_ylabel('Mean Squared Error (Cross-Validation MSE)', fontsize=10.5, labelpad=6)
ax1.set_title('LASSO Diagnostic 1: Cross-Validation MSE Curve', fontsize=11, fontweight='bold', pad=10)
ax1.legend(frameon=False, loc='best')
sns.despine(ax=ax1, offset=6, trim=True)
fig1.savefig(os.path.join(OUTPUT_DIR, 'Lasso_Diagnostic_1_MSE_Path.png'), bbox_inches='tight')
plt.close(fig1)

fig2, ax2 = plt.subplots(figsize=(6, 4.6), dpi=300)
for i in range(coefs_path.shape[0]):
    ax2.plot(alphas_path, coefs_path[i, :], alpha=0.4, lw=1.1)
ax2.axvline(lasso_cv.alpha_, color='#D32F2F', linestyle='--', lw=1.6)
ax2.set_xscale('log')
ax2.set_xlabel(r'Regularization Parameter ($\alpha$)', fontsize=10.5, labelpad=6)
ax2.set_ylabel('Coefficient Weights', fontsize=10.5, labelpad=6)
ax2.set_title('LASSO Diagnostic 2: Coefficient Convergence Path', fontsize=11, fontweight='bold', pad=10)
ax2.axhline(0, color='black', linestyle=':', alpha=0.4)
sns.despine(ax=ax2, offset=6, trim=True)
fig2.savefig(os.path.join(OUTPUT_DIR, 'Lasso_Diagnostic_2_Coefficient_Trace.png'), bbox_inches='tight')
plt.close(fig2)

fig3, ax3 = plt.subplots(figsize=(7.2, 5.0), dpi=300)
df_l_w = pd.DataFrame({'Feature': top_f, 'Weight': [coef_series[f] for f in top_f]}).sort_values('Weight', key=abs, ascending=True)
bars_colors = ['#1976D2' if w < 0 else '#D32F2F' for w in df_l_w['Weight']]
ax3.barh(range(len(df_l_w)), df_l_w['Weight'], color=bars_colors, height=0.55, alpha=0.85, edgecolor='#444444', lw=0.6, zorder=3)
ax3.set_yticks(range(len(df_l_w)))
ax3.set_yticklabels([f.replace('delta_', r'$\Delta$ ').replace('_', ' ') if f.startswith('delta_') else r'$\Delta$ ' + f.replace('_', ' ') for f in df_l_w['Feature']], fontsize=9.5, fontweight='bold')
ax3.set_xlabel('LASSO Regression Coefficients Weight Amount', fontsize=10.5, labelpad=6)
ax3.set_ylabel(r'$\Delta\text{radiomics features}$', fontsize=11, fontweight='bold', pad=8)
ax3.set_title(r'LASSO Diagnostic 3: Selected $\Delta\text{radiomics features}$ Weights Distribution', fontsize=11, fontweight='bold', pad=10)
ax3.axvline(0, color='black', lw=1.1, zorder=2)
ax3.grid(axis='x', linestyle=':', alpha=0.4)
sns.despine(ax=ax3, offset=6, trim=True)
fig3.savefig(os.path.join(OUTPUT_DIR, 'Lasso_Diagnostic_3_Feature_Weights.png'), bbox_inches='tight')
plt.close(fig3)

# ==============================================================================
# 4. DOUBLE-STANDARDIZATION TECHNIQUE FOR HARSH OVERFLOW BLOCKING
# ==============================================================================
np.random.seed(42)
scaler_s = StandardScaler()
df_deriv_input['Global_Radiomics_Score'] = scaler_s.fit_transform(df_deriv_input[top_f]).mean(axis=1) + np.random.normal(0, 1e-5, size=len(df_deriv_input))
df_valid_input['Global_Radiomics_Score'] = scaler_s.transform(df_valid_input[top_f]).mean(axis=1) + np.random.normal(0, 1e-5, size=len(df_valid_input))

scaler_score_final = StandardScaler()
df_deriv_input['Global_Radiomics_Score'] = scaler_score_final.fit_transform(df_deriv_input['Global_Radiomics_Score'].values.reshape(-1, 1))
df_valid_input['Global_Radiomics_Score'] = scaler_score_final.transform(df_valid_input['Global_Radiomics_Score'].values.reshape(-1, 1))

df_master_final_backed = pd.concat([df_deriv_input, df_valid_input]).sort_index()
df_master_final_backed.to_csv(OUTPUT_CSV_PATH, index=False)

# ==============================================================================
# 5. INTEGRATED COX SURVIVAL RISK MODEL CALIBRATION
# ==============================================================================
print("⏳ Adjusting multivariable cluster proportional hazard risk models...")
cph_std = CoxPHFitter(penalizer=0.25).fit(df_deriv_input[['time_t0', 'status'] + covariates], duration_col='time_t0', event_col='status')
cph_rad = CoxPHFitter(penalizer=0.25).fit(df_deriv_input[['time_t0', 'status', 'Global_Radiomics_Score'] + covariates], duration_col='time_t0', event_col='status')

cox_weights_records = []
for feat in covariates:
    cox_weights_records.append({'Model Type': 'Baseline Clinical Model', 'Feature Name': feat, 'Coefficient Weight (Beta)': fmt_stat_dynamic(cph_std.summary.loc[feat, 'coef']), 'Hazard Ratio (HR)': fmt_stat_dynamic(cph_std.summary.loc[feat, 'exp(coef)']), 'Wald P-value': fmt_p_dynamic(cph_std.summary.loc[feat, 'p'])})
for feat in ['Global_Radiomics_Score'] + covariates:
    cox_weights_records.append({'Model Type': 'Integrated Model (Clin. + Radiomics)', 'Feature Name': feat, 'Coefficient Weight (Beta)': fmt_stat_dynamic(cph_rad.summary.loc[feat, 'coef']), 'Hazard Ratio (HR)': fmt_stat_dynamic(cph_rad.summary.loc[feat, 'exp(coef)']), 'Wald P-value': fmt_p_dynamic(cph_rad.summary.loc[feat, 'p'])})
pd.DataFrame(cox_weights_records).to_csv(os.path.join(OUTPUT_DIR, 'Table_S13_Cox_Model_Coefficients.csv'), index=False)

prob_std_deriv = 1 - cph_std.predict_survival_function(df_deriv_input, times=[96]).T.values.flatten()
prob_rad_deriv = 1 - cph_rad.predict_survival_function(df_deriv_input, times=[96]).T.values.flatten()
prob_std_valid = 1 - cph_std.predict_survival_function(df_valid_input, times=[96]).T.values.flatten()
prob_rad_valid = 1 - cph_rad.predict_survival_function(df_valid_input, times=[96]).T.values.flatten()

# ==============================================================================
# 6. EMPIRICAL BOOTSTRAP SAMPLING ENGINE FOR 95% CONFIDENCE INTERVALS
# ==============================================================================
print("⏳ Resolving empirical validation paths via non-parametric bootstrapping loops...")
np.random.seed(42)
n_boot = 100

def run_bootstrap_metrics(df_sub, prob_s, prob_r):
    ev = df_sub['status'].values
    b_auc_s, b_auc_r, b_c_s, b_c_r = [], [], [], []
    for _ in range(n_boot):
        idx = np.random.choice(len(ev), size=len(ev), replace=True)
        if ev[idx].sum() == 0 or (ev[idx] == 0).sum() == 0: 
            continue
        b_auc_s.append(auc(roc_curve(ev[idx], prob_s[idx])[0], roc_curve(ev[idx], prob_s[idx])[1]))
        b_auc_r.append(auc(roc_curve(ev[idx], prob_r[idx])[0], roc_curve(ev[idx], prob_r[idx])[1]))
        b_c_s.append(concordance_index(df_sub['time_t0'].values[idx], -prob_s[idx], ev[idx]))
        b_c_r.append(concordance_index(df_sub['time_t0'].values[idx], -prob_r[idx], ev[idx]))
    return np.percentile(b_auc_s, [2.5, 97.5]), np.percentile(b_auc_r, [2.5, 97.5]), np.percentile(b_c_s, [2.5, 97.5]), np.percentile(b_c_r, [2.5, 97.5])

ci_auc_s_d, ci_auc_r_d, ci_c_s_d, ci_c_r_d = run_bootstrap_metrics(df_deriv_input, prob_std_deriv, prob_rad_deriv)
ci_auc_s_v, ci_auc_r_v, ci_c_s_v, ci_c_r_v = run_bootstrap_metrics(df_valid_input, prob_std_valid, prob_rad_valid)

delong_p_deriv = compute_delong_p(df_deriv_input['status'].values, prob_std_deriv, prob_rad_deriv)
delong_p_valid = compute_delong_p(df_valid_input['status'].values, prob_std_valid, prob_rad_valid)

# ==============================================================================
# 7. EXPORTING 6 STANDALONE HIGH-RESOLUTION CLINICAL RISK CURVES
# ==============================================================================
print("🎨 Drafting publication-grade un-crossed prognostic single charts...")

def save_standalone_roc(ev, p_s, p_r, ci_s, ci_r, p_delong, filename, title_text):
    f_s, t_s, _ = roc_curve(ev, p_s)
    f_r, t_r, _ = roc_curve(ev, p_r)
    fig, ax = plt.subplots(figsize=(5.8, 5.0), dpi=300)
    ax.plot([0, 1], [0, 1], linestyle='--', color='gray', lw=1.2, label='Reference Baseline (AUC = 0.500)')
    ax.plot(f_s, t_s, color='#D32F2F', lw=1.8, linestyle=':', label='Baseline Clinical Model')
    ax.plot(f_r, t_r, color='#1976D2', lw=2.6, label='Integrated Model (Clin. + Radiomics)')
    ax.legend(frameon=False, loc='lower right', fontsize=9.5)
    
    card = (f"Integrated Model AUC: {auc(f_r, t_r):.3f} ($95\\%$ $\\text{{CI}}$: {ci_r[0]:.3f} to {ci_r[1]:.3f})\n"
            f"Baseline Clinical AUC: {auc(f_s, t_s):.3f} ($95\\%$ $\\text{{CI}}$: {ci_s[0]:.3f} to {ci_s[1]:.3f})\n"
            f"DeLong Test {fmt_p_latex(p_delong)}")
    ax.text(0.96, 0.22, card, transform=ax.transAxes, fontsize=9.5, fontweight='bold', ha='right', va='bottom', bbox=dict(facecolor='#F8F9FA', alpha=0.95, edgecolor='#E0E0E0', boxstyle='round,pad=0.3'))
    ax.set_xlabel("False Positive Rate (1 - Specificity)", fontsize=10.5, labelpad=6)
    ax.set_ylabel("True Positive Rate (Sensitivity)", fontsize=10.5, labelpad=6)
    ax.set_title(title_text, fontsize=11, fontweight='bold', pad=12)
    sns.despine(ax=ax, offset=6, trim=True)
    fig.savefig(os.path.join(OUTPUT_DIR, filename), bbox_inches='tight')
    plt.close(fig)

def save_standalone_cal(ev, p_s, p_r, c_s_val, c_r_val, ci_s, ci_r, filename, title_text):
    fig, ax = plt.subplots(figsize=(5.8, 5.0), dpi=300)
    ax.plot([0, 1], [0, 1], linestyle='--', color='gray', lw=1.2, label='Ideal Perfect Calibration')
    pr_s, ob_s = calculate_calibration(p_s, ev)
    pr_r, ob_r = calculate_calibration(p_r, ev)
    ax.plot(pr_s, ob_s, marker='X', color='#D32F2F', lw=1.8, linestyle=':', label='Baseline Clinical Model')
    ax.plot(pr_r, ob_r, marker='o', color='#1976D2', lw=2.6, label='Integrated Model (Clin. + Radiomics)')
    ax.legend(frameon=False, loc='upper left', fontsize=9.5)
    
    card = (f"Integrated Model C-index: {c_r_val:.3f} ($95\\%$ $\\text{{CI}}$: {ci_r[0]:.3f} to {ci_r[1]:.3f})\n"
            f"Baseline Clinical C-index: {c_s_val:.3f} ($95\\%$ $\\text{{CI}}$: {ci_s[0]:.3f} to {ci_s[1]:.3f})")
    ax.text(0.96, 0.04, card, transform=ax.transAxes, fontsize=9.5, fontweight='bold', ha='right', va='bottom', bbox=dict(facecolor='#F8F9FA', alpha=0.95, edgecolor='#E0E0E0', boxstyle='round,pad=0.3'))
    ax.set_xlabel("Predicted Risk Probability (8-Year Horizon)", fontsize=10.5, labelpad=6)
    ax.set_ylabel("Observed Actual Event Rate (Proportion)", fontsize=10.5, labelpad=6)
    ax.set_title(title_text, fontsize=11, fontweight='bold', pad=12)
    sns.despine(ax=ax, offset=6, trim=True)
    fig.savefig(os.path.join(OUTPUT_DIR, filename), bbox_inches='tight')
    plt.close(fig)

def save_standalone_dca(ev, p_s, p_r, filename, title_text):
    fig, ax = plt.subplots(figsize=(5.8, 5.0), dpi=300)
    nb_s = [get_net_benefit(t, p_s, ev) for t in thresholds]
    nb_r = [get_net_benefit(t, p_r, ev) for t in thresholds]
    nb_all = [get_net_benefit(t, np.ones(len(ev)), ev) for t in thresholds]
    ax.plot(thresholds, nb_r, color='#1976D2', lw=2.6, label='Integrated Model (Clin. + Radiomics)')
    ax.plot(thresholds, nb_s, color='#D32F2F', lw=1.8, linestyle='--', label='Baseline Clinical Model')
    ax.plot(thresholds, nb_all, color='gray', alpha=0.4, lw=1.2, label='Assume All Patients Fail')
    ax.axhline(0, color='black', lw=1.0, label='Assume None Fail')
    ax.legend(frameon=False, loc='upper right', fontsize=9.5)
    ax.set_ylim(-0.02, max(max(nb_r), max(nb_s)) + 0.03)
    ax.set_xlim(0.0, 0.52)
    ax.set_xlabel(r"Threshold Probability (Clinical Decision Preference $P_t$)", fontsize=10.5, labelpad=6)
    ax.set_ylabel("Net Benefit (Standardized Clinical Gain Metric)", fontsize=10.5, labelpad=6)
    ax.set_title(title_text, fontsize=11, fontweight='bold', pad=12)
    sns.despine(ax=ax, offset=6, trim=True)
    fig.savefig(os.path.join(OUTPUT_DIR, filename), bbox_inches='tight')
    plt.close(fig)

save_standalone_roc(df_deriv_input['status'].values, prob_std_deriv, prob_rad_deriv, ci_auc_s_d, ci_auc_r_d, delong_p_deriv, 'Figure_5A_ROC_Derivation_Cohort.png', 'ROC Curve: Derivation Cohort (Sites D,B,A,E)')
save_standalone_roc(df_valid_input['status'].values, prob_std_valid, prob_rad_valid, ci_auc_s_v, ci_auc_r_v, delong_p_valid, 'Figure_5B_ROC_Validation_Cohort.png', 'ROC Curve: Independent Blind Testing (Site C Locked)')

save_standalone_cal(df_deriv_input['status'].values, prob_std_deriv, prob_rad_deriv, cph_std.concordance_index_, cph_rad.concordance_index_, ci_c_s_d, ci_c_r_d, 'Figure_5C_Calibration_Derivation_Cohort.png', 'Calibration Curve: Derivation Cohort (Sites D,B,A,E)')
save_standalone_cal(df_valid_input['status'].values, prob_std_valid, prob_rad_valid, concordance_index(df_valid_input['time_t0'].values, -prob_std_valid, df_valid_input['status'].values), concordance_index(df_valid_input['time_t0'].values, -prob_rad_valid, df_valid_input['status'].values), ci_c_s_v, ci_c_r_v, 'Figure_5D_Calibration_Validation_Cohort.png', 'Calibration Curve: Independent Blind Testing (Site C Locked)')

save_standalone_dca(df_deriv_input['status'].values, prob_std_deriv, prob_rad_deriv, 'Figure_5E_DCA_Derivation_Cohort.png', 'Decision Curve Analysis: Derivation Cohort')
save_standalone_dca(df_valid_input['status'].values, prob_std_valid, prob_rad_valid, 'Figure_5F_DCA_Validation_Cohort.png', 'Decision Curve Analysis: Independent Blind Testing (Site C Locked)')

# ==============================================================================
# 8. CONSOLIDATED EXTERNAL SITE TESTING SHEET TABLE S11
# ==============================================================================
metrics_records = [
    {
        'Evaluation Cohort': 'Derivation Cohort (Sites D,B,A,E Combined)',
        'Baseline Model C-index': f"{cph_std.concordance_index_:.3f} ({ci_c_s_d[0]:.3f} to {ci_c_s_d[1]:.3f})",
        'Integrated Model C-index': f"{cph_rad.concordance_index_:.3f} ({ci_c_r_d[0]:.3f} to {ci_c_r_d[1]:.3f})",
        'Baseline 8-Year AUC': f"{auc(roc_curve(df_deriv_input['status'].values, prob_std_deriv)[0], roc_curve(df_deriv_input['status'].values, prob_std_deriv)[1]):.3f} ({ci_auc_s_d[0]:.3f} to {ci_auc_s_d[1]:.3f})",
        'Integrated 8-Year AUC': f"{auc(roc_curve(df_deriv_input['status'].values, prob_rad_deriv)[0], roc_curve(df_deriv_input['status'].values, prob_rad_deriv)[1]):.3f} ({ci_auc_r_d[0]:.3f} to {ci_auc_r_d[1]:.3f})",
        'DeLong Test P-value': fmt_p_dynamic(delong_p_deriv)
    },
    {
        'Evaluation Cohort': 'Independent Site Validation Cohort (Site C Locked)',
        'Baseline Model C-index': f"{concordance_index(df_valid_input['time_t0'].values, -prob_std_valid, df_valid_input['status'].values):.3f} ({ci_c_s_v[0]:.3f} to {ci_c_s_v[1]:.3f})",
        'Integrated Model C-index': f"{concordance_index(df_valid_input['time_t0'].values, -prob_rad_valid, df_valid_input['status'].values):.3f} ({ci_c_r_v[0]:.3f} to {ci_c_r_v[1]:.3f})",
        'Baseline 8-Year AUC': f"{auc(roc_curve(df_valid_input['status'].values, prob_std_valid)[0], roc_curve(df_valid_input['status'].values, prob_std_valid)[1]):.3f} ({ci_auc_s_v[0]:.3f} to {ci_auc_s_v[1]:.3f})",
        'Integrated 8-Year AUC': f"{auc(roc_curve(df_valid_input['status'].values, prob_rad_valid)[0], roc_curve(df_valid_input['status'].values, prob_rad_valid)[1]):.3f} ({ci_auc_r_v[0]:.3f} to {ci_auc_r_v[1]:.3f})",
        'DeLong Test P-value': fmt_p_dynamic(delong_p_valid)
    }
]
pd.DataFrame(metrics_records).to_csv(os.path.join(OUTPUT_DIR, 'Table_S11_Model_Performance_Metrics_Split.csv'), index=False)

print("\n🏁 [Success: Phase 5 Multi-Center Site-Split Prognostic Pipeline Concluded Perfectly]")
print(f"➡️ Total Asset Folder Dispatched: {OUTPUT_DIR}\n")