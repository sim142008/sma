import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import ast

df = pd.read_csv('twitter_dataset.csv')
df['Sentiment'] = df['Sentiment'].str.lower()
df['Engagement'] = df['Like_count'] + df['Retweet_count']
df['text_length'] = df['Tweet'].str.len()
engagement_by_length = df.groupby(pd.cut(df['text_length'], bins=5), observed=False)['Engagement'].mean()

# Visualize engagement by text length
plt.figure(figsize=(10, 6))
sns.barplot(x=engagement_by_length.index, y=engagement_by_length.values)
plt.title('Average Engagement by Tweet Length')
plt.xlabel('Tweet Length Bins')
plt.ylabel('Average Engagement')
plt.show()

# Top 10 engaging users
top_users = df.groupby('Username')['Engagement'].sum().nlargest(10).reset_index()
plt.figure(figsize=(10,6))
sns.barplot(x='Engagement', y='Username', data=top_users, palette='coolwarm')
plt.title('Top 10 Engaging Users')
plt.xlabel('Total Engagement (Likes + Retweets)')
plt.ylabel('Username')
plt.show()

# Top 10 Tweets by Engagement
top_tweets = df.nlargest(10, 'Engagement')[['Username', 'Tweet', 'Engagement']]
print("\nTop 10 Tweets by Engagement:")
print(top_tweets.to_string(index=False))
