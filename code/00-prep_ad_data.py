from pyannote.audio import Pipeline, Inference, Model
import os
from extract_audio import *
from sp_diarization import *
from rttm_to_csv import *
from split_audio import *
from transcribe_audio_files import * 
from sp_id import *
from count_issues import *
from label_emosent import *
from convert_wmv_mp4 import convert_wmv_to_mp4
from clear_directory import delete_files_in_directories

# declare permanent directories
ad_directory = "data/ad_testing"
pres_embeddings_directory = "data/speech_embedding_models"

# declare intermediate directories
audio_directory = "data/ad_audio_testing"
diar_directory = "data/ad_audio_diarized_testing"
split_audio_directory = "data/ad_audio_split_testing"
transcript_directory = "data/ad_transcripts_testing"

# CAUTION: delete files in intermediate directories (everything but ad_directory and pres_embeddings_directory) for a clean slate
delete_files_in_directories([audio_directory, diar_directory, split_audio_directory, transcript_directory])

# conver .wmv files to .mp4
convert_wmv_to_mp4(ad_directory)

# extract audio from input directory videos --------------------------------------------------------------------------------
batch_extract(ad_directory, audio_directory)

# speaker diarization ------------------------------------------------------------------------------------------------------
hf_auth = os.environ['HF_AUTH']

pipeline = Pipeline.from_pretrained(
"pyannote/speaker-diarization-3.1",
use_auth_token=hf_auth)

batch_diarize(audio_directory, diar_directory, hf_auth)

# turn RTTM into CSV -------------------------------------------------------------------------------------------------------
# convert rttm files to single csv
result_df = combine_rttm_files(diar_directory)

# select, rename, create columns
df = (
result_df
.select(pl.col("FileID", "SpeakerID"), pl.col("StartTime", "Duration").cast(pl.Float32))
.rename({"FileID": "file_id", "StartTime": "start", "Duration": "duration", "SpeakerID": "speaker_est"})
.with_columns((pl.col("start") + pl.col("duration")).alias("end"))
)

# # save result
# df.write_csv("data/ad_validation_testing.csv")

# split audio --------------------------------------------------------------------------------------------------------------
split_audio(audio_directory, split_audio_directory, df)

# compare audio with presidential voices-----------------------------------------------------------------------------------
df_id = speaker_id(pres_embeddings_directory, split_audio_directory, df, hf_auth)

# combine audio according to 2 procedures: speakers detected by comparing each segment, versus comparing to a sample of segments from each speaker
# for now will attempt to get transcripts without combining audio so that I can put off identifying speakers - need to tune threshold

# get transcripts --------------------------------------------------------------------------------------------------------
model = whisper.load_model("large")
df_tr_id = transcribe_audio(model, split_audio_directory, "../data/ad_transcripts_testing", df_id)

# issue detection on transcripts -----------------------------------------------------------------------------------------
keywords_data = pl.read_csv("data/important_terms.csv") # get issue keywords from Tarr et al., 2023

df_tr_id_is = df_tr_id # make copy of data for adding issue counts
del df_tr_id # delete old

n = df_tr_id_is.shape[0] # get number of rows
for topic, keywords in zip(keywords_data["yt"], keywords_data["word"]):
    temp_column = []
    for i in range(0, n):
        transcript = df_tr_id_is[i, 'transcript']
        issue_count = count_issues(transcript, keywords)
        temp_column.append(issue_count)
    df_tr_id_is = df_tr_id_is.with_columns(pl.Series(name = str(topic) + "_count", values = temp_column))

# sentiment and emotion classification on transcripts ----------------------------------------------------------------------------------
df_tr_id_is_es = df_tr_id_is
del df_tr_id_is
df_tr_id_is_es = df_tr_id_is_es.with_columns(pl.lit(999).alias("vader_sentiment")).with_columns(
    pl.col("transcript").map_elements(lambda text: get_vader_sentiment(text), return_dtype=pl.String).alias("vader_sentiment")
)

# Apply the get_roberta_emotions function to each row and expand the result into multiple columns -------------------------------------
roberta_emotions_columns = {f"roberta_{emotion}": df_tr_id_is_es["transcript"].map_elements(lambda text: get_roberta_emotions(text).get(emotion, None), return_dtype=pl.Float32) for emotion in ['admiration', 'amusement', 'anger', 'annoyance', 'approval', 'caring', 'confusion', 'curiosity', 'desire', 'disappointment', 'disapproval', 'disgust', 'embarrassment', 'excitement', 'fear', 'gratitude', 'grief', 'joy', 'love', 'nervousness', 'optimism', 'pride', 'realization', 'relief', 'remorse', 'sadness', 'surprise', 'neutral']}

df_tr_id_is_es = df_tr_id_is_es.hstack(
    pl.DataFrame(roberta_emotions_columns)
)

df_tr_id_is_es.write_csv("data/ad_data_complete.csv")
