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

plt.figure(figsize=(15, 5))
colors = ['red', 'orange', 'green', 'blue']
for (model_name, scores), color in zip(cv_results.items(), colors):
    plt.plot(range(1, 11), scores, marker='o', label=model_name, linewidth=2, markersize=6, color=color)
    avg_score = np.mean(scores)
    plt.axhline(y=avg_score, color=color, linestyle='--', linewidth=1.5, alpha=0.6)

plt.xlabel('Cross-Validation Fold', fontsize=12, fontweight='bold')
plt.ylabel('MAE', fontsize=12, fontweight='bold')
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