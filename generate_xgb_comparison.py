import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from src import config

# 1. Load exact metrics from model comparison
metrics_path = os.path.join(config.METRICS_DIR, 'model_comparison_metrics.csv')
df = pd.read_csv(metrics_path)

# Extract rows for XGB Base and XGB Spatial
xgb_base_row = df[df['Modelo'].str.contains('XGBoost Base')].iloc[0]
xgb_spatial_row = df[df['Modelo'].str.contains('Covariables Geogr')].iloc[0]

metrics_data = [
    {
        'Métrica': 'Macro F1',
        'XGBoost Base': xgb_base_row['Macro F1 (Principal)'],
        'XGBoost Espacial': xgb_spatial_row['Macro F1 (Principal)'],
    },
    {
        'Métrica': 'Balanced Accuracy',
        'XGBoost Base': xgb_base_row['Balanced Accuracy'],
        'XGBoost Espacial': xgb_spatial_row['Balanced Accuracy'],
    },
    {
        'Métrica': 'MCC Multiclase',
        'XGBoost Base': xgb_base_row['MCC Multiclase'],
        'XGBoost Espacial': xgb_spatial_row['MCC Multiclase'],
    },
    {
        'Métrica': 'ROC-AUC (Macro)',
        'XGBoost Base': xgb_base_row['ROC-AUC (OvR)'],
        'XGBoost Espacial': xgb_spatial_row['ROC-AUC (OvR)'],
    },
    {
        'Métrica': 'PR-AUC (Macro)',
        'XGBoost Base': xgb_base_row['PR-AUC (OvR)'],
        'XGBoost Espacial': xgb_spatial_row['PR-AUC (OvR)'],
    }
]

df_comp = pd.DataFrame(metrics_data)

# Compute percentage improvement: ((Espacial - Base) / Base) * 100
df_comp['Mejora Absoluta'] = df_comp['XGBoost Espacial'] - df_comp['XGBoost Base']
df_comp['Mejora (%)'] = ((df_comp['XGBoost Espacial'] - df_comp['XGBoost Base']) / df_comp['XGBoost Base']) * 100

# Format for clear report
df_comp_formatted = df_comp.copy()
df_comp_formatted['XGBoost Base'] = df_comp_formatted['XGBoost Base'].map('{:.4f}'.format)
df_comp_formatted['XGBoost Espacial'] = df_comp_formatted['XGBoost Espacial'].map('{:.4f}'.format)
df_comp_formatted['Mejora (%)'] = df_comp_formatted['Mejora (%)'].map('+{:.2f} %'.format)
df_comp_formatted['Mejora Absoluta'] = df_comp_formatted['Mejora Absoluta'].map('+{:.4f}'.format)

# Save Table CSV & Excel
table_csv_path = os.path.join(config.TABLES_DIR, 'xgb_spatial_vs_base_comparison.csv')
table_article_path = os.path.join(config.ARTICLE_TABLES_DIR, 'Table_2_XGB_Spatial_vs_Base.csv')
df_comp_formatted.to_csv(table_csv_path, index=False)
df_comp_formatted.to_csv(table_article_path, index=False)
print(f"Saved comparison table to {table_csv_path}")

# 2. Generate Horizontal Grouped Bar Chart Figure (Q1 Publication Quality)
plt.rcParams.update({
    'font.size': 11,
    'font.family': 'sans-serif',
    'axes.labelsize': 11,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 11,
    'figure.titlesize': 14
})

fig, ax = plt.subplots(figsize=(10, 6))

y = np.arange(len(df_comp))
height = 0.35

# Color scheme: Muted orange for Base, Deep emerald for Spatial
c_base = '#ff7f0e'
c_spatial = '#2ca02c'

rects1 = ax.barh(y + height/2, df_comp['XGBoost Base'], height, label='XGBoost Base (Sin Espacio)', color=c_base, alpha=0.9, edgecolor='black', linewidth=0.5)
rects2 = ax.barh(y - height/2, df_comp['XGBoost Espacial'], height, label='XGBoost con Covariables Geográficas', color=c_spatial, alpha=0.95, edgecolor='black', linewidth=0.5)

ax.set_xlabel('Valor de la Métrica de Evaluación', fontweight='bold', labelpad=10)
ax.set_title('Performance Gain from Spatial Covariates in Homicide Mechanism Classification', fontweight='bold', pad=15)
ax.set_yticks(y)
ax.set_yticklabels(df_comp['Métrica'], fontweight='bold')
ax.invert_yaxis()  # top-down order
ax.set_xlim(0, 0.85)
ax.grid(True, linestyle='--', alpha=0.3, axis='x')
ax.legend(loc='lower right', frameon=True, framealpha=0.9, fontsize=10.5)

# Annotate bars with values and improvement percentages
for i in range(len(df_comp)):
    val_base = df_comp.loc[i, 'XGBoost Base']
    val_spatial = df_comp.loc[i, 'XGBoost Espacial']
    pct_gain = df_comp.loc[i, 'Mejora (%)']
    
    # Value label on Base bar
    ax.text(val_base + 0.01, i + height/2, f"{val_base:.4f}", va='center', ha='left', fontsize=9.5, color='#444444')
    # Value and % improvement label on Spatial bar
    ax.text(val_spatial + 0.01, i - height/2, f"{val_spatial:.4f} ({pct_gain:+.2f}%)", va='center', ha='left', fontsize=10, fontweight='bold', color='#1b5e20')

plt.tight_layout()

fig_path = os.path.join(config.FIGURES_DIR, 'xgboost_spatial_vs_base_improvement.png')
plt.savefig(fig_path, dpi=300, bbox_inches='tight')
plt.close()
print(f"Saved publication figure to {fig_path}")
