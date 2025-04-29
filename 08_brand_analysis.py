import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('twitter_dataset.csv')
brand_counts = df['Brand'].value_counts().head(10)
brand_likes_retweets = df.groupby('Brand')[['Like_count', 'Retweet_count']].mean().loc[brand_counts.index]

# Plot 1: Bar plot for Brand Mentions
plt.figure(figsize=(10, 6))
sns.barplot(x=brand_counts.values, y=brand_counts.index, color='lightblue')
plt.title('Top 10 Brands Mentioned')
plt.xlabel('Brand Mentions')
plt.ylabel('Brand')
plt.show()

# Plot 2: Line plot for Average Likes per Brand
plt.figure(figsize=(10, 6))
sns.lineplot(x=brand_likes_retweets['Like_count'], y=brand_likes_retweets.index, color='green', marker='o')
plt.title('Average Likes per Brand')
plt.xlabel('Average Likes')
plt.ylabel('Brand')
plt.show()

# Plot 3: Line plot for Average Retweets per Brand
plt.figure(figsize=(10, 6))
sns.lineplot(x=brand_likes_retweets['Retweet_count'], y=brand_likes_retweets.index, color='orange', marker='o')
plt.title('Average Retweets per Brand')
plt.xlabel('Average Retweets')
plt.ylabel('Brand')
plt.show()
