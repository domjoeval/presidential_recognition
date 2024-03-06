import os
import whisper
import polars as pl
import pickle
from babel.dates import format_time

# Function to transcribe audio using OpenAI's transcription service
def transcribe_audio(model, audio_path, transcript_path, df):
    n = df.shape[0]
    temp_column = []
    for i in range(0, n):
        print("Processing segment " + str(i + 1) + " of " + str(n))
        file_path = audio_path + "/" + df[i, 'file_id'] + "_" + str(round(df[i, 'start'], 3)) + ".wav"
        result = model.transcribe(file_path)
        temp_column.append(result['text']) # add transcript text to column to add to df
        with open(transcript_path + "/" + df[i, 'file_id'] + "_" + str(round(df[i, 'start'], 3)) + ".pkl", "wb") as fp:   # pickling full transcript data
            pickle.dump(result, fp)
        print("Segment " + str(i + 1) + " of " + str(n) + " complete")
    df = df.with_columns(pl.Series(name="transcript", values=temp_column))
    return(df)

if __name__ == "__main__":
    df = pl.read_csv("data/ad_validation_initial.csv")
    df_test = df[6:8]
    model = whisper.load_model("large")
    df_tr = transcribe_audio(model, "data/ad_audio_split_testing", "data/ad_transcripts_testing", df_test)
    df_tr.write_csv("data/df_tr_testing.csv")