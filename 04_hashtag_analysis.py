import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt

df = pd.read_csv('twitter_dataset.csv')
hashtags = df['Hashtags'].dropna().str.split(',').sum()
counter = Counter(hashtags)
common = counter.most_common(10)
labels, counts = zip(*common)
plt.barh(labels, counts)
plt.title('Top 10 Hashtags')
plt.show()