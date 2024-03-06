import os
import math
import numpy as np
from pyannote.audio import Pipeline, Inference, Model
from scipy.spatial.distance import cdist
from extract_audio import *
from sp_diarization import *
from rttm_to_csv import *
from split_audio import * 

def speaker_id(embeddings_directory, split_audio_directory, df, hf_auth):
    embeddings_dict = {}
    for filename in os.listdir(embeddings_directory):
        if filename.endswith(".npy"):
            file_path = os.path.join(embeddings_directory, filename)
            try:
                # Read the numpy array from the file
                embeddings = np.load(file_path)
                # Extract the speaker ID from the filename (assuming the filename is in the format "speaker_id.npy")
                speaker_id = os.path.splitext(filename)[0]
                # Add the embeddings to the dictionary with speaker ID as the key
                embeddings_dict[speaker_id] = embeddings
            except Exception as e:
                print(f"Error reading embeddings from '{filename}': {str(e)}")

    model = Model.from_pretrained("pyannote/embedding", use_auth_token=hf_auth)
    inference = Inference(model, window="whole")

    n = df.shape[0]
    for pres in embeddings_dict:
        temp_column = []
        for i in range(0, n):
            file_path = split_audio_directory + "/" + df[i, 'file_id'] + "_" + str(round(df[i, 'start'], 3)) + ".wav"
            temp_embedding = inference(file_path)
            distance = 1 - (np.matmul(temp_embedding, embeddings_dict[pres]) / (math.sqrt(np.matmul(temp_embedding, temp_embedding)) * math.sqrt(np.matmul(embeddings_dict[pres], embeddings_dict[pres])))) # cosine distance
            temp_column.append(distance)
        df = df.with_columns(pl.Series(name=pres, values=temp_column))
        return(df)

if __name__ == "__main__":

    pres_embeddings_directory = "../data/speech_embedding_models"
    split_audio_directory = "../data/ad_audio_split_testing"
    hf_auth = os.environ['HF_AUTH']

    df = pl.read_csv("../data/ad_validation_initial.csv")

    # Perform speaker identification
    speaker_id(pres_embeddings_directory, split_audio_directory, hf_auth)