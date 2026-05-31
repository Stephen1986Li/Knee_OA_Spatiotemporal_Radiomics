import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os

# ==============================================================================
# 1. PATH DEFINITIONS & DIRECTORY BINDING
# ==============================================================================
BASE_DIR = r'C:\在研课题\radiomics_time_window\2026-05-12'
SHAP_DATA_PATH = os.path.join(BASE_DIR, 'Phase_2_Refinement_Retro', 'Table_S4_Sentinel_SHAP.csv')
RAW_DATA_PATH = os.path.join(BASE_DIR, 'final_merged_cleaned_v13.csv')
OUTPUT_DIR = os.path.join(BASE_DIR, 'Phase_2_Refinement_Retro')

if not os.path.exists(OUTPUT_DIR): 
    os.makedirs(OUTPUT_DIR)

# ==============================================================================
# 2. MATRIX LOADING & TOP CONTRIBUTING FEATURE FILTRATION
# ==============================================================================
print("⏳ Loading unpolluted countdown metrics and executing matrix isolation...")
df_shap = pd.read_csv(SHAP_DATA_PATH)
df_raw = pd.read_csv(RAW_DATA_PATH, low_memory=False)

# Isolate Top 20 features possessing maximum absolute mean SHAP parameters
top_features_list = df_shap.groupby('Feature')['SHAP'].apply(lambda x: x.abs().mean()).sort_values(ascending=False).head(20).index.tolist()
feature_mapping = df_shap[df_shap['Feature'].isin(top_features_list)][['Feature', 'Raw_ID']].drop_duplicates('Feature')
raw_ids = feature_mapping['Raw_ID'].tolist()

# Enforcing strict chronological hard-alignment boundaries at -24m
df_v00 = df_raw[df_raw['visit'] == 'V00'][['id1'] + raw_ids].copy().drop_duplicates('id1')
df_sentinel = df_raw[df_raw['years_to_event'] == -2][['id1'] + raw_ids].copy().drop_duplicates('id1')

df_master = pd.merge(df_v00, df_sentinel, on='id1', suffixes=('_v00', '_retro')).dropna()
print(f"🚀 Master unpolluted alignment completed. Paired sample tracking N = {len(df_master)}")

# Vectorized dictionary aggregation to prevent DataFrame fragmentation loops
delta_dict = {}
for _, row in feature_mapping.iterrows():
    raw_id = row['Raw_ID']
    display_name = row['Feature']
    delta_dict[display_name] = df_master[f'{raw_id}_retro'] - df_master[f'{raw_id}_v00']

delta_matrix = pd.DataFrame(delta_dict, index=df_master.index).dropna()
corr_matrix = delta_matrix.corr(method='pearson')

# ==============================================================================
# 3. RENDERING HIERARCHICAL CLUSTERMAP (Figure S5 Under Science Aesthetics)
# ==============================================================================
sns.set_style("white")

g = sns.clustermap(corr_matrix, 
                   method='complete', 
                   cmap='RdBu_r', 
                   annot=True, 
                   fmt=".2f", 
                   vmin=-1, vmax=1, 
                   figsize=(15, 13),
                   cbar_pos=(0.02, 0.82, 0.025, 0.12),
                   annot_kws={"size": 7.5, "weight": "bold"}) 

plt.setp(g.ax_heatmap.get_xticklabels(), rotation=45, ha='right', fontsize=9, fontweight='bold')
plt.setp(g.ax_heatmap.get_yticklabels(), rotation=0, fontsize=9, fontweight='bold')

# Academic caption formatting placed at the bottom region
note_text = (f"Figure S5: Hierarchical clustering of the top 20 radiomics features based on Pearson correlation.\n"
             f"All Delta values were calculated within the exact Sentinel Window (-24m vs Baseline) via rigorous Hard Alignment.\n"
             f"Clusters represent pure redundancy groups within the microstructural predictors of joint failure (N = {len(delta_matrix)} full paired cases).")
plt.figtext(0.5, 0.02, note_text, wrap=True, horizontalalignment='center', fontsize=10.5, style='italic', fontweight='bold', color='#222222')

# Saving unpolluted scientific assets
plt.savefig(os.path.join(OUTPUT_DIR, 'Figure_S5_Sentinel_Feature_Redundancy_Hard.png'), dpi=300, bbox_inches='tight')
corr_matrix.to_csv(os.path.join(OUTPUT_DIR, 'Table_S5_Sentinel_Correlation_Matrix_Hard.csv'))
plt.close()

print(f"🏁 [Success: Figure S5 Spatial Redundancy Matrix Asset Dispatched]")