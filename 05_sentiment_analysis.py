import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud

# Load dataset
df = pd.read_csv('twitter_dataset.csv')
df = df.dropna(subset=['Tweet', 'Sentiment'])

# Sentiment Distribution Plot
sns.countplot(data=df, x='Sentiment')
plt.title('Sentiment Count')
plt.show()

# Show example tweets
print("\nPositive Tweets Examples:\n", df[df['Sentiment']=='positive']['Tweet'].head())
print("\nNegative Tweets Examples:\n", df[df['Sentiment']=='negative']['Tweet'].head())
print("\nNeutral Tweets Examples:\n", df[df['Sentiment']=='neutral']['Tweet'].head())

# ➡ Content analysis per Sentiment (WordCloud)
sentiments = ['positive', 'negative', 'neutral']

for sentiment in sentiments:
    text = ' '.join(df[df['Sentiment'] == sentiment]['Tweet'])
    if text:  # only if there is text
        wc = WordCloud(width=800, height=400, background_color='white', stopwords='english').generate(text)
        plt.figure(figsize=(8, 4))
        plt.imshow(wc, interpolation='bilinear')
        plt.axis('off')
        plt.title(f'WordCloud for {sentiment} Sentiment')
        plt.show()