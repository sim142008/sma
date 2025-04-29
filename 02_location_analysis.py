import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('twitter_dataset.csv')
loc_counts = df['Location'].value_counts().head(10)
sns.barplot(x=loc_counts.values, y=loc_counts.index)
plt.title('Top 10 Locations')
plt.show()