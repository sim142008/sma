import pandas as pd
import networkx as nx
from networkx.algorithms.community import girvan_newman
import matplotlib.pyplot as plt

# Load the dataset and take the first 50 rows
df = pd.read_csv('twitter_dataset.csv').head(50)

# Create an undirected graph
G = nx.Graph()

# Add edges between users who mention the same brand
for brand in df['Brand'].dropna().unique():
    users = df[df['Brand'] == brand]['Username'].dropna().unique()
    for i in range(len(users)):
        for j in range(i + 1, len(users)):
            G.add_edge(users[i], users[j], brand=brand)

# Detect communities using the Girvan-Newman algorithm
communities = next(girvan_newman(G))

# Print each community
for i, community in enumerate(communities, start=1):
    print(f"Community {i}: {community}")

# Generate graph layout with better spacing
pos = nx.spring_layout(G, k=1.2, iterations=100)

# Draw the graph with improved spacing
plt.figure(figsize=(12, 10))
nx.draw(G, pos, with_labels=True, node_size=300, font_size=8, edge_color='gray')
plt.title("User Graph Based on Common Brand Mentions")
plt.show()
