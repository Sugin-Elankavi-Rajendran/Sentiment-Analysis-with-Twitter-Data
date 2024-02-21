# Sentiment-Analysis-with-Twitter-Data

This Python script performs sentiment analysis on Twitter data using the Natural Language Toolkit (nltk) library and visualizes the sentiment distribution.

# Prerequisites

Before running the script, ensure you have the required libraries installed. You can install them using the following commands:
```bash
pip install pandas nltk matplotlib seaborn
```
# Usage

1. Clone the repository or download the script.
2. Modify the file_path variable in the script to point to your Twitter data CSV file.

```bash
file_path = r"D:\OneDrive\Prodigy InfoTech\DS_04\archive\twitter_training.csv"
```

3. Run the script.
```bash
python sentiment_analysis.py
```

# Description

The script performs the following tasks:

1. **Loading Data:** Reads the Twitter data from the specified CSV file into a Pandas DataFrame.
```bash
df = pd.read_csv(file_path, header=None, names=['ID', 'Brand', 'Sentiment', 'Text'])
```

2. **Data Inspection:** Displays the first few rows of the DataFrame and checks for null values.
```bash
print(df.head())
print(df.isnull().sum())
```

3. **Data Cleaning:** Drops rows with null values in the 'Text' column.
```bash
df = df.dropna()
print(df.isnull().sum())
```

4. **Text Preprocessing:** Defines a function preprocess_text to clean the text by removing URLs, mentions, hashtags, retweet tags, and non-alphabetic characters. It also tokenizes the text and removes stop words.
```bash
def preprocess_text(text):
    # ... (code snippet)
    return filtered_text

df['cleaned_text'] = df['Text'].apply(preprocess_text)
```

5. **Sentiment Analysis:** Utilizes the VADER sentiment intensity analyzer from nltk to calculate sentiment scores for each cleaned text.
```bash
sia = SentimentIntensityAnalyzer()

def get_sentiment_score(text):
    return sia.polarity_scores(text)['compound']

df['sentiment_score'] = df['cleaned_text'].apply(get_sentiment_score)
```

6. **Sentiment Distribution Visualization:** Plots a histogram of the sentiment scores using seaborn and matplotlib.
```bash
plt.figure(figsize=(10, 6))
sns.histplot(df['sentiment_score'], bins=30, kde=True)
plt.title('Sentiment Distribution')
plt.xlabel('Sentiment Score')
plt.ylabel('Frequency')
plt.show()
```
![Example Image](https://github.com/sugin22/Sentiment-Analysis-with-Twitter-Data/blob/main/Figure_1.png)
