import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# Load dataset
df = pd.read_csv('twitter_dataset.csv')
df['Timestamp'] = pd.to_datetime(df['Timestamp'])
df = df.dropna(subset=['Tweet'])
df['Month'] = df['Timestamp'].dt.to_period('M')

# Text Vectorization
vectorizer = CountVectorizer(stop_words='english', max_features=1000)
X = vectorizer.fit_transform(df['Tweet'])

# Topic Modeling
lda = LatentDirichletAllocation(n_components=3, random_state=42)
topics = lda.fit_transform(X)

# Assign most probable topic
df['Topic'] = topics.argmax(axis=1)

# ➡ Assign topic names based on top keywords
words = vectorizer.get_feature_names_out()
topic_keywords = []
for idx, topic in enumerate(lda.components_):
    top_words = [words[i] for i in topic.argsort()[:-6:-1]]
    topic_keywords.append(' / '.join(top_words))

# Group by Month and Topic
monthly_topic_trend = df.groupby(['Month', 'Topic']).size().unstack(fill_value=0)

# Rename columns using topic keywords
monthly_topic_trend.columns = topic_keywords

# Plot
monthly_topic_trend.plot(marker='o', figsize=(12, 6))
plt.title('Trending Topics Over Time')
plt.xlabel('Month')
plt.ylabel('Number of Tweets')
plt.legend(title='Topics', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()
