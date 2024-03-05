from pyannote.audio import Pipeline, Inference, Model
import os
from extract_audio import *
from sp_diarization import *
from rttm_to_csv import *
from split_audio import *
from transcribe_audio_files import * 
from sp_id import *

# declare project directories
ad_directory = "data/ad_testing"
audio_directory = "data/ad_audio_testing"
diar_directory = "data/ad_audio_diarized_testing"
split_audio_directory = "data/ad_audio_split_testing/"
pres_embeddings_directory = "../data/speech_embedding_models"

# extract audio from input directory videos --------------------------------------------------------------------------------
batch_extract(ad_directory, audio_directory)

# speaker diarization ------------------------------------------------------------------------------------------------------
hf_auth = os.environ['HF_AUTH']

pipeline = Pipeline.from_pretrained(
"pyannote/speaker-diarization-3.1",
use_auth_token=hf_auth)

batch_diarize(input_directory, diar_directory)

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

# save result
df.write_csv("data/ad_validation_testing.csv")

# split audio --------------------------------------------------------------------------------------------------------------
split_audio(audio_directory, split_audio_directory, df)

# compare audio with presidential voices-----------------------------------------------------------------------------------
df_id = speaker_id(pres_embeddings_directory, split_audio_directory, df, hf_auth)

# combine audio according to 2 procedures: speakers detected by comparing each segment, versus comparing to a sample of segments from each speaker
# for now will attempt to get transcripts without combining audio so that I can put off identifying speakers - need to tune threshold

# get transcripts
model = whisper.load_model("large")
df_tr_id = transcribe_audio(model, "../data/ad_audio_split_testing", "../data/ad_transcripts_testing", df_id)

# text analysis on transcript (sentiment analysis, issue detection)