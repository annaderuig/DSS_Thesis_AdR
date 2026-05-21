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

# Plot cluster distribution
cluster_counts = [np.sum(final_clusters == k) for k in range(K)]
plt.figure(figsize=(8, 5))
plt.bar(range(K), cluster_counts)
plt.xlabel('Cluster')
plt.ylabel('Number of Occupations')
plt.title('Cluster Distribution')
plt.xticks(range(K))
plt.savefig('cluster_distribution.png', dpi=300, bbox_inches='tight')
plt.show()

# Save to final_data.csv
final.to_csv('final_clustered.csv', index=False)

#### ALLEEN VOOR EXCEL AI CREATIE => Create Excel file with occupations by cluster
max_cluster_size = final['Cluster'].value_counts().max()
excel_data = {}

for cluster in sorted(final['Cluster'].unique()):
    cluster_occupations = final[final['Cluster'] == cluster].sort_values('SOC2018_Occupation_Title')
    occupation_list = cluster_occupations['SOC2018_Occupation_Title'].tolist()
    # Pad with None to make all columns same length
    occupation_list.extend([None] * (max_cluster_size - len(occupation_list)))
    excel_data[f'Cluster {cluster}'] = occupation_list

occupations_df = pd.DataFrame(excel_data)

output_excel = 'skill_clusters.xlsx'
occupations_df.to_excel(output_excel, sheet_name='Clusters', index=False)

