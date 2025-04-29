import pandas as pd
import networkx as nx
from networkx.algorithms.community import girvan_newman
import matplotlib.pyplot as plt

df = pd.read_csv('twitter_dataset.csv').head(50)
G = nx.Graph()

for brand in df['Brand'].dropna().unique():
    users = df[df['Brand'] == brand]['Username'].dropna().unique()
    for i in range(len(users)):
        for j in range(i + 1, len(users)):
            G.add_edge(users[i], users[j], brand=brand)

comp = next(girvan_newman(G))
i = 1
for community in comp:
    print(f"Community {i}: {community}")
    i += 1

plt.figure(figsize=(10, 8))
nx.draw(G, with_labels=True, node_size=50, font_size=8)
plt.title("User Graph Based on Common Brand Mentions")
plt.show()