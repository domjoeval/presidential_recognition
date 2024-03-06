import polars as pl
from transformers import pipeline
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk
nltk.download('vader_lexicon')

# Function to get sentiment using VADER
def get_vader_sentiment(text):
    sia = SentimentIntensityAnalyzer()
    sentiment_scores = sia.polarity_scores(text)
    if sentiment_scores['compound'] >= 0.05:
        return 'positive'
    elif sentiment_scores['compound'] <= -0.05:
        return 'negative'
    else:
        return 'neutral'

# Function to get emotions using RoBERTa model with probabilities for each emotion
def get_roberta_emotions(text):
    roberta_emotion = pipeline("text-classification", model="SamLowe/roberta-base-go_emotions")
    emotions = roberta_emotion(text)
    emotion_dict = {emotion['label']: emotion['score'] for emotion in emotions}
    return emotion_dict

if __name__ == "__main__":
    df_tr_id_is = pl.read_csv("data/df_tr_id_is_testing.csv")

    # Add columns for sentiment and emotions
    df_tr_id_is = df_tr_id_is.with_columns(pl.lit(999).alias("vader_sentiment")).with_columns(
        pl.col("transcript").map_elements(lambda text: get_vader_sentiment(text), return_dtype=pl.String).alias("vader_sentiment")
    )

    # Apply the get_roberta_emotions function to each row and expand the result into multiple columns
    roberta_emotions_columns = {f"roberta_{emotion}": df_tr_id_is["transcript"].map_elements(lambda text: get_roberta_emotions(text).get(emotion, None), return_dtype=pl.Float32) for emotion in ['admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring', 'confusion', 'curiosity', 'desire', 'disappointment', 'disapproval', 'disgust', 'embarrassment', 'excitement', 'fear', 'gratitude', 'grief', 'joy', 'love', 'nervousness', 'optimism', 'pride', 'realization', 'relief', 'remorse', 'sadness', 'surprise', 'neutral']}

    df_tr_id_is = df_tr_id_is.hstack(
        pl.DataFrame(roberta_emotions_columns)
    )

    print(df_tr_id_is)