import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
from lightgbm import LGBMRegressor
import matplotlib.pyplot as plt
import shap
import joblib
import seaborn as sns

# Load dataset
df = pd.read_csv('final_clustered.csv')




# Extract skill columns
skill_cols = [col for col in df.columns if col not in 
              ['SOC2018_Code', 'SOC2018_Occupation_Title', 'Automation_Probability', 'Cluster']]

X = df[skill_cols].copy()
y = df['Automation_Probability'].copy()
clusters = df['Cluster'].copy()





# Check multicollinearity
corr_matrix = X.corr()
upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

# Find correlations > 0.8
high_corr = [(column, row, upper_tri.loc[row, column]) 
             for column in upper_tri.columns 
             for row in upper_tri.index 
             if upper_tri.loc[row, column] > 0.8]
high_corr_sorted = sorted(high_corr, key=lambda x: abs(x[2]), reverse=True)[:10]

if high_corr_sorted:
    print("High correlations:")
    for col1, col2, corr in high_corr_sorted:
        print(col1, "&", col2, ":", corr)
else:
    print("No high correlationsfound")
print()





# Stratified train/test split based on cluster
X_train, X_test, y_train, y_test, cluster_train, cluster_test = train_test_split(
    X, y, clusters, 
    test_size=0.2, 
    random_state=42,
    stratify=clusters
)

print("Train size:", X_train.shape[0], (len(X_train)/len(X)*100), "%")
print("Test size:", X_test.shape[0], (len(X_test)/len(X)*100), "%")





# Combine features, target, and clusters into DataFrames
train_data = X_train.copy()
train_data['Automation_Probability'] = y_train.values
train_data['Cluster'] = cluster_train.values

test_data = X_test.copy()
test_data['Automation_Probability'] = y_test.values
test_data['Cluster'] = cluster_test.values





# Save to CSV
train_data.to_csv('train_data.csv', index=False)
test_data.to_csv('test_data.csv', index=False)

print("Train data saved to train_data.csv")
print("Test data saved to test_data.csv")




# Display cluster distribution in train/test
print("\nCluster distribution in train set:")
for cluster in sorted(cluster_train.unique()):
    count = (cluster_train == cluster).sum()
    pct = count / len(cluster_train) * 100
    print("  Cluster", cluster, ":", count, "(", round(pct, 1), "%)")

print("\nCluster distribution in test set:")
for cluster in sorted(cluster_test.unique()):
    count = (cluster_test == cluster).sum()
    pct = count / len(cluster_test) * 100
    print("  Cluster", cluster, ":", count, "(", round(pct, 1), "%)")




# Initialize baseline model with randomized search and stratified 10-fold CV
print("\nStart LASSO model training...")
baseline = GridSearchCV(
    estimator=Lasso(random_state=42, max_iter=5000),
    param_grid={'alpha': [0.001, 0.01, 0.1, 0.5, 1.0]},
    cv=10,
    scoring='neg_mean_absolute_error',
    n_jobs=1
)

# Fit baseline model
baseline.fit(X_train, y_train)
baseline_best = baseline.best_estimator_
baseline_results = baseline_best.predict(X_test)

# Save best baseline model for later use
joblib.dump(baseline_best, 'best_lasso.pkl')





# Initialize Random Forest model with randomized search and 10-fold CV
print("\nStart Random Forest model training...")
rf = GridSearchCV(
    estimator=RandomForestRegressor(random_state=42, n_jobs=-1),
    param_grid={
        'n_estimators': [100, 200, 300],
        'max_depth': [5, 10, 15],
        'min_samples_split': [5, 10, 20],
        'min_samples_leaf': [5, 10, 20],
        'max_features': ['sqrt', 'log2', None]
    },
    cv=10,
    scoring='neg_mean_absolute_error',
    n_jobs=1
)

# Fit Random Forest model
rf.fit(X_train, y_train)
rf_best = rf.best_estimator_
rf_results = rf_best.predict(X_test)

# Save best RF model for later use
joblib.dump(rf_best, 'best_rf.pkl')




# Initialize LightGBM model with randomized search and 10-fold CV
print("\nStart LightGBM model training...")
lgbm = GridSearchCV(
    estimator=LGBMRegressor(random_state=42, verbose=-1, reg_alpha=0.5, reg_lambda=0.5),
    param_grid={
        'n_estimators': [100, 150, 200, 250, 300],
        'learning_rate': [0.005, 0.01, 0.05],
        'num_leaves': [5, 10, 20, 30],
        'min_child_samples': [5, 10, 15, 20],
        'subsample': [0.5, 0.7, 0.9, 1.0],
    },
    cv=10,
    scoring='neg_mean_absolute_error',
    n_jobs=1
)

# Fit LightGBM model
lgbm.fit(X_train, y_train)
lgbm_best = lgbm.best_estimator_ 
lgbm_results = lgbm_best.predict(X_test)

# Save best LGBM model for later use
joblib.dump(lgbm_best, 'best_lgbm.pkl')




# Initialize AdaBoost model
print("\nStart AdaBoost model training...")
ada = GridSearchCV(
    estimator=AdaBoostRegressor(random_state=42),
    param_grid={
        'n_estimators': [200, 300, 500],
        'learning_rate': [0.001, 0.005, 0.01],
        'loss': ['linear']
    },
    cv=10,
    scoring='neg_mean_absolute_error',
    n_jobs=1
)

# Train AdaBoost model
ada.fit(X_train, y_train)
ada_best = ada.best_estimator_
ada_results = ada_best.predict(X_test)

# Save best AdaBoost model for later use
joblib.dump(ada_best, 'best_ada.pkl')




# Evaluate models
baseline_mae = mean_absolute_error(y_test, baseline_results)
baseline_rmse = root_mean_squared_error(y_test, baseline_results)
baseline_r2 = r2_score(y_test, baseline_results)

rf_mae = mean_absolute_error(y_test, rf_results)
rf_rmse = root_mean_squared_error(y_test, rf_results)
rf_r2 = r2_score(y_test, rf_results)

lgbm_mae = mean_absolute_error(y_test, lgbm_results)
lgbm_rmse = root_mean_squared_error(y_test, lgbm_results)
lgbm_r2 = r2_score(y_test, lgbm_results)

ada_mae = mean_absolute_error(y_test, ada_results)
ada_rmse = root_mean_squared_error(y_test, ada_results)
ada_r2 = r2_score(y_test, ada_results)




# Collect CV scores for all models
cv_models = [
    ('Lasso', baseline_best),
    ('Random Forest', rf_best),
    ('LightGBM', lgbm_best),
    ('AdaBoost', ada_best),
]

fold_mae = {}
fold_rmse = {}
fold_r2 = {}

for name, model in cv_models:
    fold_mae[name] = -cross_val_score(model, X_train, y_train, cv=10, scoring='neg_mean_absolute_error')
    fold_rmse[name] = -cross_val_score(model, X_train, y_train, cv=10, scoring='neg_root_mean_squared_error')
    fold_r2[name] = cross_val_score(model, X_train, y_train, cv=10, scoring='r2')




# Print cross validation scores
print("LASSO CV")
print("MAE:", fold_mae['Lasso'])
print("RMSE:", fold_rmse['Lasso'])
print("R²:", fold_r2['Lasso'])

print("\nRandom Forest CV:")
print("MAE:", fold_mae['Random Forest'])
print("RMSE:", fold_rmse['Random Forest'])
print("R²:", fold_r2['Random Forest'])

print("\nLightGBM CV:")
print("MAE:", fold_mae['LightGBM'])
print("RMSE:", fold_rmse['LightGBM'])
print("R²:", fold_r2['LightGBM'])

print("\nAdaboost CV:")
print("MAE:", fold_mae['AdaBoost'])
print("RMSE:", fold_rmse['AdaBoost'])
print("R²:", fold_r2['AdaBoost'])




# Print evaluation results
print(
    "\nLasso Test Results:",
    "\nMAE:", baseline_mae,
    "\nRMSE:", baseline_rmse,
    "\nR²:", baseline_r2
)

print(
    "\nRandom Forest Test Results:",
    "\nMAE:", rf_mae,
    "\nRMSE:", rf_rmse,
    "\nR²:", rf_r2
)

print(
    "\nLightGBM Test Results:",
    "\nMAE:", lgbm_mae,
    "\nRMSE:", lgbm_rmse,
    "\nR²:", lgbm_r2
)

print(
    "\nAdaBoost Test Results:",
    "\nMAE:", ada_mae,
    "\nRMSE:", ada_rmse,
    "\nR²:", ada_r2
)




# Feature importance analysis for the best model
best_model = lgbm_best
fi_lgbm = pd.DataFrame({
    'Feature': skill_cols,
    'Importance': best_model.feature_importances_
}).sort_values('Importance', ascending=False).head(15)

print("\nMost important skills:")
print(fi_lgbm.head(10).to_string())




# Print average CV metrics per model
print("\nAverage CV metrics per model:")
print("\nModel                     MAE     RMSE       R2")
for name, _ in cv_models:
    mae_val = np.mean(fold_mae[name])
    rmse_val = np.mean(fold_rmse[name])
    r2_val = np.mean(fold_r2[name])
    print(name + " " * (20 - len(name)) + " {:>8.4f} {:>8.4f} {:>8.4f}".format(mae_val, rmse_val, r2_val))

# Create 3-panel CV figure
fig, axes = plt.subplots(3, 1, figsize=(15, 15), sharex=True)
fig.subplots_adjust(hspace=0.4)

panel_data = [
    (axes[0], fold_mae,  'MAE',  'Cross-Validation MAE per Fold'),
    (axes[1], fold_rmse, 'RMSE', 'Cross-Validation RMSE per Fold'),
    (axes[2], fold_r2,   'R²',   'Cross-Validation R² per Fold'),
]

colors = ['red', 'orange', 'green', 'blue']

# Plot lines and mean lines for each model in each panel
for i, (ax, fold_dict, ylabel, title) in enumerate(panel_data):
    for (name, _), color in zip(cv_models, colors):
        vals = fold_dict[name]
        avg = np.mean(vals)
        ax.plot(range(1, 11), vals, linewidth=2, markersize=4, marker='o',
                color=color, label=name)
        ax.axhline(avg, color=color, linestyle='--', linewidth=1.2, alpha=0.5)
    
    ax.set_ylabel(ylabel, fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=13, fontweight='bold')
    ax.set_xlabel('Cross-Validation Fold', fontsize=12, fontweight='bold')
    ax.tick_params(axis='x', labelbottom=True)
    if i == 0:
        ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(range(1, 11))

axes[2].set_xlabel('Cross-Validation Fold', fontsize=12, fontweight='bold')

plt.savefig('cv_fold_performance.png', dpi=300, bbox_inches='tight')
plt.close()




# Subgroup error analysis by occupational cluster
clusters_list = sorted(cluster_test.unique())
maes = []
y_pred = lgbm_results

for c in clusters_list:
    bool_mask = cluster_test == c
    mae = mean_absolute_error(y_test[bool_mask], y_pred[bool_mask])
    r2 = r2_score(y_test[bool_mask], y_pred[bool_mask])
    n = bool_mask.sum()
    
    maes.append(mae)
    print("Cluster", c, "(", n, "samples): MAE=", round(mae, 4), ", R²=", round(r2, 4))





# Additional Visualizations
## General settings for figure consistency
df = pd.read_csv('final_clustered.csv')
skill_cols = [col for col in df.columns if col not in 
              ['SOC2018_Code', 'SOC2018_Occupation_Title', 'Automation_Probability', 'Cluster']]

X_full = df[skill_cols]
clusters_full = df['Cluster']

CLUSTER_COLORS = ['#2196F3', '#FF5722', '#4CAF50', '#9C27B0']
CLUSTER_LABELS = ['Cluster 0', 'Cluster 1', 'Cluster 2', 'Cluster 3']
N_CLUSTERS = 4

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 15,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

## Skill correlation matrices
corr_matrix = X_full.corr()
mask_upper = np.triu(np.ones_like(corr_matrix, dtype=bool))
mask_low = corr_matrix <= 0.8
mask = mask_upper | mask_low

fig, ax = plt.subplots(figsize=(20, 17))
sns.heatmap(corr_matrix,
    mask=mask,
    ax=ax,
    cmap='RdBu_r',
    vmin=-1, vmax=1,
    annot=True,
    fmt='.2f',
    linewidths=0.3,
    linecolor='white',
    square=True,
    cbar_kws={'label': 'Pearson Correlation (> 0.8)', 'shrink': 0.7},
    annot_kws={'size': 6},
)
ax.set_title('Skill Correlation Matrix (> 0.8)', fontsize=15, fontweight='bold')
ax.tick_params(axis='x', labelsize=15, rotation=90)
ax.tick_params(axis='y', labelsize=15, rotation=0)
plt.tight_layout()
plt.savefig('skill_correlation_matrix.png', dpi=300, bbox_inches='tight')
plt.close()

mask_upper_only = np.triu(np.ones_like(corr_matrix, dtype=bool))
fig, ax = plt.subplots(figsize=(20, 17))
sns.heatmap(corr_matrix,
    mask=mask_upper_only, 
    ax=ax,
    cmap='RdBu_r',
    vmin=-1, vmax=1,
    annot=True,
    fmt='.2f', 
    linewidths=0.3,
    linecolor='white',
    square=True,
    cbar_kws={'label': 'Pearson Correlation', 'shrink': 0.7},
    annot_kws={'size': 10},
)
ax.set_title('Skill Correlation Matrix', fontsize=15, fontweight='bold')
ax.tick_params(axis='x', labelsize=15, rotation=90)
ax.tick_params(axis='y', labelsize=15, rotation=0)
plt.tight_layout()
plt.savefig('skill_correlation_matrix_full.png', dpi=300, bbox_inches='tight')
plt.close()

## Full feature importance for LightGBM
fi_df = pd.DataFrame({
    'Skill': skill_cols,
    'Importance': best_model.feature_importances_
}).sort_values('Importance', ascending=True)

top10_threshold = fi_df['Importance'].nlargest(10).min()
bar_colors = ['#2196F3' if v >= top10_threshold else '#B0BEC5' for v in fi_df['Importance']]

fig, ax = plt.subplots(figsize=(10, 14))
bars = ax.barh(fi_df['Skill'], fi_df['Importance'], color=bar_colors, edgecolor='white', linewidth=0.4, height=0.7)
for bar, val in zip(bars, fi_df['Importance']):
    ax.text(val + 1, bar.get_y() + bar.get_height() / 2,
            str(int(val)), va='center', fontsize=15)
ax.set_xlabel('Feature Importance Score (LightGBM)', fontsize=12, fontweight='bold')
ax.set_title('Feature Importance', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3, axis='x')
plt.tight_layout()
plt.savefig('feature_importance_plot.png', dpi=300, bbox_inches='tight')
plt.close()

## SHAP Values for LightGBM
xai_explainer = shap.TreeExplainer(best_model)
shap_values = xai_explainer.shap_values(X_test)

shap.summary_plot(shap_values, X_test, max_display=35, show=False, plot_size=(12, 14))
plt.title('SHAP Values', fontsize=15, fontweight='bold')
plt.tight_layout()
plt.savefig('shap_beeswarm.png', dpi=300, bbox_inches='tight')
plt.close()

## Feature importance bar chart (top 15)
fi_top15 = fi_df.nlargest(15, 'Importance').sort_values('Importance', ascending=True)

n_bars = len(fi_top15)
palette = [plt.cm.Blues(0.35 + 0.65 * i / (n_bars - 1)) for i in range(n_bars)]

fig, ax = plt.subplots(figsize=(10, 8))
bars = ax.barh(fi_top15['Skill'], fi_top15['Importance'],
               color=palette, edgecolor='white', linewidth=0.4, height=0.65)

x_max = fi_top15['Importance'].max()
for bar, val in zip(bars, fi_top15['Importance']):
    ax.text(val + x_max * 0.012, bar.get_y() + bar.get_height() / 2,
            str(int(val)), va='center', ha='left', fontsize=9, color='#333333')

ax.set_xlabel('Feature Importance Score', fontsize=11, fontweight='bold')
ax.set_xlim(0, x_max * 1.13)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_visible(False)
ax.tick_params(axis='y', length=0, labelsize=10)
ax.tick_params(axis='x', labelsize=9)
ax.grid(True, alpha=0.25, axis='x', linestyle='--')
ax.set_axisbelow(True)
fig.suptitle('Top 15 Most Important Skills (LightGBM)', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('feature_importance_top15.png', dpi=300, bbox_inches='tight')
plt.close()

print("All plots created successfully.")