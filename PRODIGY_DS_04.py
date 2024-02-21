import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.sentiment import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
import seaborn as sns

df  = pd.read_csv(r"D:\OneDrive\Prodigy InfoTech\DS_04\archive\twitter_training.csv", header=None, names=['ID', 'Brand', 'Sentiment', 'Text'])

print(df.head())

print(df.isnull().sum())

df = df.dropna()

print(df.isnull().sum())

nltk.download('stopwords')
nltk.download('punkt')
nltk.download('vader_lexicon')

def preprocess_text(text):
    text = re.sub(r'http\S+', '', text)  
    text = re.sub(r'@[A-Za-z0-9]+', '', text)  
    text = re.sub(r'#', '', text)  
    text = re.sub(r'RT[\s]+', '', text)  
    text = re.sub(r'\n', '', text)  
    text = re.sub(r'\s+', ' ', text).strip()  

    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(text)

    filtered_words = []
    for word in word_tokens:
        if word.isalpha() and word.lower() not in stop_words:
            filtered_words.append(word.lower())
    filtered_text = ' '.join(filtered_words)        
    
    return filtered_text

df['cleaned_text'] = df['Text'].apply(preprocess_text)

sia = SentimentIntensityAnalyzer()

def get_sentiment_score(text):
    return sia.polarity_scores(text)['compound']

df['sentiment_score'] = df['cleaned_text'].apply(get_sentiment_score)

plt.figure(figsize=(10, 6))
sns.histplot(df['sentiment_score'], bins=30, kde=True)
plt.title('Sentiment Distribution')
plt.xlabel('Sentiment Score')
plt.ylabel('Frequency')
plt.show()