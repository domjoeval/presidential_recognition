from pyannote.audio import Pipeline
import os

def batch_diarize(input_directory, output_directory):
    i = 1
    for filename in os.listdir(input_directory):
        if filename.endswith(".mp3"): 
            print(i)
            input_path = os.path.join(input_directory, filename)
            base_name = os.path.splitext(filename)[0]
            output_path = os.path.join(output_directory, base_name + ".rttm")
             
            # apply pretrained pipeline
            diarization = pipeline(input_path)

            # dump the diarization output to disk using RTTM format
            with open(output_path, "w") as rttm:
                diarization.write_rttm(rttm)
            i += 1

if __name__ == "__main__":

    hf_auth = os.environ['HF_AUTH']

    pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization-3.1",
    use_auth_token=hf_auth)
    # Replace these paths with your input and output directories
    input_directory = "data/ad_audio"
    output_directory = "data/ad_audio_diarized"

    # Perform batch conversion
    batch_diarize(input_directory, output_directory)