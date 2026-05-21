import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
import matplotlib.pyplot as plt
import shap
import joblib

# Load merged, clustered data
df = pd.read_csv('final_clustered.csv')

# Load train and test data
train_data = pd.read_csv('train_data.csv')
test_data = pd.read_csv('test_data.csv')

# Separate features and target
X_train = train_data.drop(['Automation_Probability', 'Cluster'], axis=1)
y_train = train_data['Automation_Probability']
cluster_train = train_data['Cluster']

X_test = test_data.drop(['Automation_Probability', 'Cluster'], axis=1)
y_test = test_data['Automation_Probability']
cluster_test = test_data['Cluster']

# Load best models from interim pipeline
baseline_best = joblib.load('best_lasso.pkl')
rf_best = joblib.load('best_rf.pkl')
lgbm_best = joblib.load('best_lgbm.pkl')
ada_best = joblib.load('best_ada.pkl')

# Predict with best models
baseline_results = baseline_best.predict(X_test)
rf_results = rf_best.predict(X_test)
lgbm_results = lgbm_best.predict(X_test)
ada_results = ada_best.predict(X_test)  

# Evaluate models
## LASSO baseline
baseline_mae = mean_absolute_error(y_test, baseline_results)
baseline_rmse = root_mean_squared_error(y_test, baseline_results)
baseline_r2 = r2_score(y_test, baseline_results)
## Random Forest Regressor
rf_mae = mean_absolute_error(y_test, rf_results)
rf_rmse = root_mean_squared_error(y_test, rf_results)
rf_r2 = r2_score(y_test, rf_results)
## LightGBM Regressor
lgbm_mae = mean_absolute_error(y_test, lgbm_results)
lgbm_rmse = root_mean_squared_error(y_test, lgbm_results)
lgbm_r2 = r2_score(y_test, lgbm_results)
## AdaBoost Regressor
ada_mae = mean_absolute_error(y_test, ada_results)
ada_rmse = root_mean_squared_error(y_test, ada_results)
ada_r2 = r2_score(y_test, ada_results)

# Print cross validation scores
print("LASSO CV")
print("MAE:", -cross_val_score(baseline_best, X_train, y_train, cv=10, scoring='neg_mean_absolute_error'))
print("RMSE:", -cross_val_score(baseline_best, X_train, y_train, cv=10, scoring='neg_root_mean_squared_error'))
print("R²:", cross_val_score(baseline_best, X_train, y_train, cv=10, scoring='r2'))

print("\nRandom Forest CV:")
print("MAE:", -cross_val_score(rf_best, X_train, y_train, cv=10, scoring='neg_mean_absolute_error'))
print("RMSE:", -cross_val_score(rf_best, X_train, y_train, cv=10, scoring='neg_root_mean_squared_error'))
print("R²:", cross_val_score(rf_best, X_train, y_train, cv=10, scoring='r2'))

print("\nLightGBM CV:")
print("MAE:", -cross_val_score(lgbm_best, X_train, y_train, cv=10, scoring='neg_mean_absolute_error'))
print("RMSE:", -cross_val_score(lgbm_best, X_train, y_train, cv=10, scoring='neg_root_mean_squared_error'))
print("R²:", cross_val_score(lgbm_best, X_train, y_train, cv=10, scoring='r2'))

print("\nAdaboost CV:")
print("MAE:", -cross_val_score(ada_best, X_train, y_train, cv=10, scoring='neg_mean_absolute_error'))
print("RMSE:", -cross_val_score(ada_best, X_train, y_train, cv=10, scoring='neg_root_mean_squared_error'))
print("R²:", cross_val_score(ada_best, X_train, y_train, cv=10, scoring='r2'))

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
## Define skill columns
skill_cols = [col for col in df.columns if col not in 
              ['SOC2018_Code', 'SOC2018_Occupation_Title', 'Automation_Probability', 'Cluster']]
## Feature importance for LightGBM model
best_model = lgbm_best
fi_lgbm = pd.DataFrame({
    'Feature': skill_cols,
    'Importance': best_model.feature_importances_
}).sort_values('Importance', ascending=False).head(15)

print("\nMost important skills:")
print(fi_lgbm.head(10).to_string())

# Plot feature importance - ALL 35 skills
fi_all = pd.DataFrame({
    'Skill': skill_cols,
    'Importance': best_model.feature_importances_
}).sort_values('Importance', ascending=True)

fig, ax = plt.subplots(figsize=(12, 10))
ax.barh(fi_all['Skill'], fi_all['Importance'], color='steelblue', edgecolor='black', linewidth=0.5)
ax.set_xlabel('Importance Score', fontsize=12, fontweight='bold')
ax.set_ylabel('Skill', fontsize=12, fontweight='bold')
ax.set_title('Feature Importance', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
plt.close()

# SHAP values for LightGBM model
xai_explainer = shap.TreeExplainer(best_model)
shap_values = xai_explainer.shap_values(X_test)

# Plot SHAP results
shap.summary_plot(shap_values, X_test, plot_type="bar", show=False)
plt.savefig('shap_importance.png', dpi=300, bbox_inches='tight')
plt.close()

# Cross-validation plot
cv_results = {}
baseline_cv = cross_val_score(baseline_best, X_train, y_train, cv=10, scoring='neg_mean_absolute_error')
rf_cv = cross_val_score(rf_best, X_train, y_train, cv=10, scoring='neg_mean_absolute_error')
lgbm_cv = cross_val_score(lgbm_best, X_train, y_train, cv=10, scoring='neg_mean_absolute_error')
ada_cv = cross_val_score(ada_best, X_train, y_train, cv=10, scoring='neg_mean_absolute_error')

cv_results['Lasso'] = -baseline_cv
cv_results['Random Forest'] = -rf_cv
cv_results['LightGBM'] = -lgbm_cv
cv_results['AdaBoost'] = -ada_cv

plt.figure(figsize=(12, 6))
colors = ['red', 'orange', 'green', 'blue']
for (model_name, scores), color in zip(cv_results.items(), colors):
    plt.plot(range(1, 11), scores, marker='o', label=model_name, linewidth=2, markersize=6, color=color)
    avg_score = np.mean(scores)
    plt.axhline(y=avg_score, color=color, linestyle='--', linewidth=1.5, alpha=0.6)

plt.xlabel('Cross-Validation Fold', fontsize=12, fontweight='bold')
plt.ylabel('Mean Absolute Error (MAE)', fontsize=12, fontweight='bold')
plt.title('Model Performance Across Folds', fontsize=14, fontweight='bold')
plt.legend(fontsize=11, loc='upper left')
plt.grid(True, alpha=0.3)
plt.xticks(range(1, 11))
plt.tight_layout()
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