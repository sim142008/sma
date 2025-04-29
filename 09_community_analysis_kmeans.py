import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Load the dataset and take the first 100 rows
df = pd.read_csv('twitter_dataset.csv').head(100)

# Create an undirected graph
G = nx.Graph()

# Add edges between users who mention the same brand
for brand in df['Brand'].dropna().unique():
    users = df[df['Brand'] == brand]['Username'].dropna().unique()
    for i in range(len(users)):
        for j in range(i + 1, len(users)):
            G.add_edge(users[i], users[j], brand=brand)

# Convert the graph to an adjacency matrix
adj_matrix = nx.to_pandas_adjacency(G)

# Apply KMeans clustering
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(adj_matrix)

# Get cluster labels for each user
labels = kmeans.labels_
for user, label in zip(adj_matrix.index, labels):
    print(f"User: {user}, Cluster: {label}")

# Generate the graph layout with extra spacing
pos = nx.spring_layout(G, k=1.2, iterations=100, seed=42)  # Increased 'k' for more space

# Set figure size
plt.figure(figsize=(16, 12))

# Draw nodes, grouped by clusters
for cluster in set(labels):
    nodes_in_cluster = [node for node, lbl in zip(adj_matrix.index, labels) if lbl == cluster]
    nx.draw_networkx_nodes(G, pos, nodelist=nodes_in_cluster, label=f'Cluster {cluster}', node_size=300)

# Draw edges and labels
nx.draw_networkx_edges(G, pos, alpha=0.5)
nx.draw_networkx_labels(G, pos, font_size=10)

# Display the plot with title and legend
plt.title("User Graph Clustered by KMeans", fontsize=14)
plt.legend()
plt.axis('off')
plt.tight_layout()
plt.show(
