import nltk
import string
import os

from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from collections import Counter

import pandas as pd
from tqdm import tqdm

DATA_DIR = "/pasteur/u/esui/data/podclip/"
# DATA_DIR = "/Users/caravanuden/git-repos/cs224w_project/data"
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
INTERMEDIATE_DATA_DIR = os.path.join(DATA_DIR, "intermediate")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
TOP_N_WORDS = 1000
TOP_N_COUNTS = 100

nltk.download("stopwords")
nltk_stop_words = stopwords.words("english")


def tokenize(text):
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    tokens = nltk.word_tokenize(text)
    stems = []
    for item in tokens:
        if not item in nltk_stop_words:
            stems.append(PorterStemmer().stem(item))
    return stems


token_dict = {}
df = pd.read_table(
    os.path.join(
        RAW_DATA_DIR, "podcasts-no-audio-13GB/spotify-podcasts-2020/metadata.tsv"
    )
)

for i, row in tqdm(df.iterrows(), total=df.shape[0]):
    # token_dict[row.show_uri] = tokenize(str(row.show_description))
    token_dict[row.episode_uri] = tokenize(str(row.show_description)) + tokenize(
        str(row.episode_description)
    )

keyword = "interview"
show_uris = []
show_descs = []
df["include"] = df.apply(lambda row: keyword in token_dict[row["episode_uri"]], axis=1)
df = df[df.include == True]
df = df.drop("include", axis=1)
transcripts_dir = os.path.join(
    RAW_DATA_DIR, "podcasts-no-audio-13GB/spotify-podcasts-2020/podcasts-transcripts"
)
df["path"] = df.apply(
    lambda row: os.path.join(
        transcripts_dir,
        row["show_filename_prefix"].split("_")[-1][0],
        row["show_filename_prefix"].split("_")[-1][1],
        row["show_filename_prefix"],
        f"{row['episode_filename_prefix']}.json",
    ),
    axis=1,
)

subset_token_dict = {
    key: val for key, val in token_dict.items() if key in list(df["episode_uri"])
}
corpus = list(subset_token_dict.values())
corpus = [item for sublist in corpus for item in list(set(sublist))]

counter = Counter(corpus)

# import pdb; pdb.set_trace()

print(counter.most_common(100))
words = [word for (word, count) in counter.items() if count >= TOP_N_COUNTS]
counts = [count for (_, count) in counter.items()]

print(len([c for c in counts if c > 1000]))
print(len([c for c in counts if c > 100]))
print(len([c for c in counts if c > 10]))
print(len([c for c in counts if c > 1]))

counter_df = pd.DataFrame(words, columns=["words"])
counter_df.to_csv(os.path.join(INTERMEDIATE_DATA_DIR, "words.csv"), index=False)

subset_token_dict = {key: " ".join(val) for key, val in subset_token_dict.items()}
tokenized_desc_df = pd.DataFrame.from_dict(subset_token_dict, orient="index")
tokenized_desc_df.columns = ["tokenized_description"]
df = df.merge(tokenized_desc_df, left_on="episode_uri", right_index=True)

df.to_csv(
    os.path.join(INTERMEDIATE_DATA_DIR, "episode_subset.tsv"), sep="\t", index=False
)
