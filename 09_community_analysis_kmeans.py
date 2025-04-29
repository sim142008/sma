import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

df = pd.read_csv('twitter_dataset.csv').head(100)
G = nx.Graph()

for brand in df['Brand'].dropna().unique():
    users = df[df['Brand'] == brand]['Username'].dropna().unique()
    for i in range(len(users)):
        for j in range(i + 1, len(users)):
            G.add_edge(users[i], users[j], brand=brand)


adj_matrix = nx.to_pandas_adjacency(G)
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(adj_matrix)

labels = kmeans.labels_
for user, label in zip(adj_matrix.index, labels):
    print(f"User: {user}, Cluster: {label}")

plt.figure(figsize=(10, 8))
pos = nx.spring_layout(G, seed=42)
for cluster in set(labels):
    nodes_in_cluster = [node for node, lbl in zip(adj_matrix.index, labels) if lbl == cluster]
    nx.draw_networkx_nodes(G, pos, nodelist=nodes_in_cluster, label=f'Cluster {cluster}', node_size=50)
nx.draw_networkx_edges(G, pos, alpha=0.5)
nx.draw_networkx_labels(G, pos, font_size=8)
plt.title("User Graph Clustered by KMeans")
plt.legend()
plt.show()