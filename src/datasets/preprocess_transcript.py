import json
import os
import pandas as pd
from tqdm import tqdm

# from src import constants

DATA_DIR = "/pasteur/u/esui/data/podclip/"
# DATA_DIR = "/Users/elainesui/CS224W/cs224w_project/src/data"
# DATA_DIR = '/Users/caravanuden/git-repos/cs224w_project/data'
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
INTERMEDIATE_DATA_DIR = os.path.join(DATA_DIR, "intermediate")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")

df = pd.read_table(os.path.join(INTERMEDIATE_DATA_DIR, "episode_subset.tsv"))

for i, row in tqdm(df.iterrows(), total=df.shape[0]):
    episode = row["episode_uri"]
    transcript_path = os.path.join(DATA_DIR, row["path"])
    processed_transcript_path = os.path.join(
        INTERMEDIATE_DATA_DIR, f"processed_transcripts/{episode}.json"
    )

    with open(transcript_path, "r") as infile:
        transcript = json.load(infile)
        transcript_words = []
        for slice in transcript["results"]:
            if "words" in slice["alternatives"][0].keys():
                slice_words = slice["alternatives"][0]["words"]
                slice_words = [
                    {
                        "word": word["word"],
                        "start_timestamp": float(word["startTime"].replace("s", "")),
                        "end_timestamp": float(word["endTime"].replace("s", "")),
                    }
                    for word in slice_words
                ]
                transcript_words += slice_words

    json_object = json.dumps({"words": transcript_words})
    with open(processed_transcript_path, "w") as outfile:
        outfile.write(json_object)
