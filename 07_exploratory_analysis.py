import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

df = pd.read_csv('twitter_dataset.csv')
print(df.describe(include='all'))

sns.pairplot(df[['Retweet_count', 'Like_count']])
plt.show()
