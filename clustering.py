import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Load data
final = pd.read_csv('merged_dataset_SOC2018.csv')

# Extract only skill columns
skill_cols = [col for col in final.columns if col not in ['SOC2018_Code', 'SOC2018_Occupation_Title', 'Automation_Probability']]
skill_df = final[skill_cols]

# PCA
scaler = StandardScaler()
skill_df_scaled = scaler.fit_transform(skill_df)

pca = PCA(n_components=3, svd_solver='auto', random_state=42)
skill_df_pca = pca.fit_transform(skill_df_scaled)

print("Original features:", {skill_df_scaled.shape[1]})
print("PCA components:", {skill_df_pca.shape[1]})
print("Variance explained:", {pca.explained_variance_ratio_.sum()}) 
print()

# Lise & Poste-Vinvay exclusion restrictions
# Get the PCA loadings (components)
loadings = pca.components_.T 

# Find indices of the three indicator skills
cognitive_idx = skill_cols.index('Mathematics')
manual_idx = skill_cols.index('Equipment Maintenance')
interpersonal_idx = skill_cols.index('Negotiation')

# For each skill, find which PC has the highest absolute loading
cognitive_pc = np.argmax(np.abs(loadings[cognitive_idx, :]))
manual_pc = np.argmax(np.abs(loadings[manual_idx, :]))
interpersonal_pc = np.argmax(np.abs(loadings[interpersonal_idx, :]))

# Reorder the components
component_order = [cognitive_pc, manual_pc, interpersonal_pc]
skill_df_pca_rotated = skill_df_pca[:, component_order]

# Rename for clarity
skill_df_pca_final = skill_df_pca_rotated.copy()

# Create dataframe with named skill dimensions
skill_dimensions = pd.DataFrame(
    skill_df_pca_final,
    columns=['Cognitive', 'Manual', 'Interpersonal']
)

# Print skills per component
print("\nTop skills per component:")

component_names = ['Cognitive', 'Manual', 'Interpersonal']
pca_loadings = pca.components_.T

for i, comp_name in enumerate(component_names):
    pc_idx = component_order[i]
    loadings_comp = pca_loadings[:, pc_idx]
    top_indices = np.argsort(np.abs(loadings_comp))[-10:][::-1]
    print("\n", comp_name, "(PC", pc_idx, "):")
    print()
    for rank, idx in enumerate(top_indices, 1):
        loading_value = loadings_comp[idx]
        print(rank, skill_cols[idx], "Loading: ", loading_value)

print("\n", skill_dimensions.head(10))

# k-means clustering
k_list = range(2, 11)
inertias = []
sli_scores = []

def fit_k_means(data, k):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init="auto", init="k-means++", max_iter=1000)
    labels = kmeans.fit_predict(data)
    inertia = kmeans.inertia_
    sil_score = silhouette_score(data, labels)
    return inertia, sil_score

for k in k_list:
    inertia, sil_score = fit_k_means(skill_df_pca_final, k)

    inertias.append(inertia)
    sli_scores.append(sil_score)

print("Inertias per k:", dict(zip(k_list, inertias)))
print("Silhouette per k:", dict(zip(k_list, sli_scores)))

# Elbow plot (Inertia)
plt.figure(figsize=(12,4))
plt.subplot(1,2,1)
plt.plot(list(k_list), inertias)
plt.title("Elbow method")
plt.xlabel("N clusters (k)")
plt.ylabel("Inertia")
plt.grid(True, alpha=0.3)
 
# Silhouette plot
best_k_sil = k_list[int(np.argmax(sli_scores))]
plt.subplot(1,2,2)
plt.plot(list(k_list), sli_scores, color='purple')
plt.title("Silhouette scores")
plt.xlabel("N clusters (k)")
plt.ylabel("Silhouette score")
plt.grid(True, alpha=0.3)
 
plt.tight_layout()
plt.savefig('cluster_evaluation_plots.png', dpi=300, bbox_inches='tight')
plt.show()

# Kmeans clustering with k=4
K = 4
kmeans_final = KMeans(n_clusters=K, random_state=42, n_init="auto", init="k-means++", max_iter=1000)
final_clusters = kmeans_final.fit_predict(skill_df_pca_final)

# Add clusters to original data
final['Cluster'] = final_clusters

# Cluster statistics
print( "\nCluster distribution:")
for k in range(K):
    count = np.sum(final_clusters == k)
    pct = (count / len(final_clusters)) * 100
    print("Cluster", k, ":", count, "occupations", "(", round(pct, 1), "%)")

print("\nTotal:", len(final_clusters), "occupations")

# Save to final_data.csv
final.to_csv('final_clustered.csv', index=False)

# Cluster Visualizations
## Clusters in PCA space
CLUSTER_COLORS = ['#2196F3', '#FF5722', '#4CAF50', '#9C27B0']
CLUSTER_LABELS = ['Cluster 0', 'Cluster 1', 'Cluster 2', 'Cluster 3']
N_CLUSTERS = 4

dim_names = ['Cognitive', 'Manual', 'Interpersonal']
variance_explained = pca.explained_variance_ratio_[component_order]

pairs = [(0, 1), (0, 2), (1, 2)]
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for ax, (i, j) in zip(axes, pairs):
    for k in range(N_CLUSTERS):
        mask = final_clusters == k
        ax.scatter(skill_df_pca_final[mask, i], skill_df_pca_final[mask, j],
                   c=CLUSTER_COLORS[k], label=CLUSTER_LABELS[k],
                   alpha=0.60, s=35, edgecolors='none')
    for k in range(N_CLUSTERS):
        mask = final_clusters == k
        ax.scatter(skill_df_pca_final[mask, i].mean(), skill_df_pca_final[mask, j].mean(),
                   c=CLUSTER_COLORS[k], s=220, marker='o',
                   edgecolors='black', linewidths=0.8, zorder=5)
    ax.set_xlabel(dim_names[i] + " PC (" + str(round(variance_explained[i]*100, 0)) + "% var)", fontsize=10, fontweight='bold')
    ax.set_ylabel(dim_names[j] + " PC (" + str(round(variance_explained[j]*100, 0)) + "% var)", fontsize=10, fontweight='bold')
    ax.set_title(dim_names[i] + " vs " + dim_names[j], fontsize=12, fontweight='bold')
    ax.legend(fontsize=9, markerscale=1.4)
    ax.grid(True, alpha=0.25)

plt.suptitle('Occupational Clusters in PCA Skill Space', fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('cluster_scatter.png', dpi=300, bbox_inches='tight')
plt.show()

## Average skill importance scores per cluster
skill_order = final[skill_cols].mean().sort_values(ascending=False).index
x_pos = list(range(len(skill_cols)))

fig, ax = plt.subplots(figsize=(18, 10))
for k in range(N_CLUSTERS):
    cluster_data = final[final['Cluster'] == k][skill_cols]
    mean_scores = cluster_data.mean()[skill_order]
    ax.plot(x_pos, mean_scores.values, color=CLUSTER_COLORS[k], linewidth=2,
         alpha=0.9, label=CLUSTER_LABELS[k])
ax.set_xticks(x_pos)
ax.set_xticklabels(skill_order, rotation=90, fontsize=15)
ax.set_xlabel('Skill', fontsize=15, fontweight='bold')
ax.set_ylabel('Average Score', fontsize=15, fontweight='bold')
ax.set_title('Average Skill Scores per Cluster', fontsize=15, fontweight='bold')
ax.legend(fontsize=15)
ax.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('skill_avg_all_clusters.png', dpi=300, bbox_inches='tight')
plt.show()

## Automation Probability distribution per cluster
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 15,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

fig, axes = plt.subplots(2, 2, figsize=(12, 9))
axes = axes.flatten()

for k in range(N_CLUSTERS):
    ax = axes[k]
    subset = final[final['Cluster'] == k]['Automation_Probability']
    ax.hist(subset, bins=20, color=CLUSTER_COLORS[k], edgecolor='white', alpha=0.9)
    ax.axvline(subset.mean(), color='#E53935', linestyle='--',
               linewidth=1.8, label="Mean: " + str(round(subset.mean(), 2)))
    ax.axvline(subset.median(), color='#FB8C00', linestyle='--',
               linewidth=1.8, label="Median: " + str(round(subset.median(), 2)))
    ax.set_xlabel('Automation Probability', fontsize=15, fontweight='bold')
    ax.set_ylabel('Frequency', fontsize=15, fontweight='bold')
    ax.set_title(CLUSTER_LABELS[k], fontsize=15, fontweight='bold')
    ax.legend(fontsize=15)
    ax.grid(True, alpha=0.3, axis='y')

plt.suptitle('Distribution of Automation Probability per Cluster', fontsize=14, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('cluster_target_distribution.png', dpi=300, bbox_inches='tight')
plt.show()

