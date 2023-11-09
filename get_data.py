import re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk
import wikipediaapi
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

nltk.download("punkt")
nltk.download("stopwords")

wiki_wiki = wikipediaapi.Wikipedia(language="en", user_agent="DataCollector/1.0")

page_name = "Artificial Intelligence"
page = wiki_wiki.page(page_name)

if page.exists():
    text = page.text

    cleaned_text = re.sub(r"http\S+", "", text)  # Remove URLs
    cleaned_text = re.sub(r"[^A-Za-z0-9 ]+", "", cleaned_text)  # Remove special chars

    # Tokenization
    tokens = word_tokenize(cleaned_text)

    # Further cleaning: removing stopwords and lowercasing
    stop_words = set(stopwords.words("english"))
    filtered_tokens = [w.lower() for w in tokens if w.lower() not in stop_words]

    vectorizer = CountVectorizer()

    df = pd.DataFrame({"words": filtered_tokens})
    df.to_csv("cleaned_wikipedia_data.csv", index=False)

    X = vectorizer.fit_transform(df["words"])
    word_features = pd.DataFrame(
        X.toarray(), columns=vectorizer.get_feature_names_out()
    )

    final_dataset = word_features.copy()
    final_dataset.to_csv("my_dataset.csv", index=False)  # Save as CSV

    print(final_dataset)
else:
    print("Page not found")
