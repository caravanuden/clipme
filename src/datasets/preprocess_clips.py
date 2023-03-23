from operator import index
import os
from tracemalloc import start

import collections
import pandas as pd
import random
import json
import string
import tqdm
import numpy as np

from keybert import KeyBERT
from bertopic import BERTopic
from bertopic.representation import MaximalMarginalRelevance
from bertopic.vectorizers import ClassTfidfTransformer
from hdbscan import HDBSCAN
import nltk
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np

import sys
import argparse

sys.path.insert(0, "../")
import constants
from constants import INTERMEDIATE_DATA_DIR

nltk.download("stopwords")
nltk.download("punkt")
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


def extract_vocab(
    df, index_col, args, keyphrase_ngram_range=constants.KEYBERT_VOCAB_NGRAM_RANGE
):
    transcripts = df[index_col].tolist()

    # Extract keywords
    kw_model = KeyBERT()
    keywords = kw_model.extract_keywords(
        transcripts, keyphrase_ngram_range=keyphrase_ngram_range, stop_words="english"
    )

    # Create our vocabulary
    vocabulary = [k[0] for keyword in keywords for k in keyword]
    vocabulary = list(set(vocabulary))

    print(f"KeyBERT found {len(vocabulary)} vocabulary words!")
    print(vocabulary[:10])

    with open(args.keybert_vocab_path, "wb") as f:
        for line in vocabulary:
            f.write((line + "\n").encode("utf-8"))
    return vocabulary


def extract_clip_transcript(episode_uri, start_timestamp, end_timestamp):
    transcript_fn = os.path.join(
        INTERMEDIATE_DATA_DIR, f"processed_transcripts/{episode_uri}.json"
    )

    processed_transcript = []
    try:
        with open(transcript_fn, "r") as f:
            transcript_json = json.load(f)
            for word in transcript_json["words"]:
                if (
                    word["start_timestamp"] >= start_timestamp
                    and word["end_timestamp"] < end_timestamp
                ):
                    processed_transcript.append(word["word"])
    except FileNotFoundError:
        return processed_transcript

    return processed_transcript


def extract_clip_topics(
    df,
    index_col,
    args,
    vocabulary=[],
    vectorizer_max_df=constants.BERTOPIC_MAX_DF,
    mmr_diversity=constants.BERTOPIC_MMR_DIVERSITY,
    min_topic_size=constants.BERTOPIC_MIN_TOPIC_SIZE,
    min_samples=constants.BERTOPIC_MIN_SAMPLES,
    top_n_words=constants.BERTOPIC_TOP_N_WORDS,
    reduce_outliers=constants.BERTOPIC_REDUCE_OUTLIERS,
    reduce_outliers_threshold=constants.BERTOPIC_REDUCE_OUTLIERS_THRESHOLD,
):
    transcripts = df[index_col].astype("str").tolist()

    if os.path.exists(args.bertopic_model_path):
        topic_model = BERTopic.load(args.bertopic_model_path)
    else:
        # vectorizer_model options
        if len(vocabulary) == 0:
            vectorizer_model = CountVectorizer(
                ngram_range=constants.KEYBERT_VOCAB_NGRAM_RANGE,
                stop_words="english",
                max_df=vectorizer_max_df,
            )
        else:
            vectorizer_model = CountVectorizer(
                vocabulary=vocabulary, max_df=vectorizer_max_df
            )

        # ctfidf_model options
        ctfidf_model = ClassTfidfTransformer(
            bm25_weighting=True, reduce_frequent_words=True
        )

        # representation model options
        ### MMR
        representation_model = MaximalMarginalRelevance(diversity=mmr_diversity)

        hdbscan_model = HDBSCAN(
            min_cluster_size=min_topic_size,
            metric="euclidean",
            cluster_selection_method="eom",
            prediction_data=True,
            min_samples=min_samples,
        )

        # Now create BERTopic model
        topic_model = BERTopic(
            top_n_words=top_n_words,
            vectorizer_model=vectorizer_model,
            ctfidf_model=ctfidf_model,
            representation_model=representation_model,
            hdbscan_model=hdbscan_model,
            language="english",
            # calculate_probabilities=True,
            verbose=True,
        )
        topics, probs = topic_model.fit_transform(transcripts)
        if reduce_outliers:
            new_topics = topic_model.reduce_outliers(
                transcripts, topics, threshold=reduce_outliers_threshold
            )
            topic_model.update_topics(transcripts, topics=new_topics)

    topic_info = topic_model.get_topic_info()
    print(f"BERTopic found {len(topic_info)} topics!")
    print(topic_info.head())

    topic_model.save(args.bertopic_model_path)

    topic_df = topic_model.get_document_info(transcripts)
    topic_df.columns = [
        index_col,
        "topic_id",
        "topic_name",
        "topic_top_n_words",
        "topic_prob",
        "topic_representative_doc",
    ]
    topic_df["topic_id"] = (
        topic_df["topic_id"] + 1
    )  # including the -1 noise topic, for now
    topic_df["topic_top_n_words"] = topic_df["topic_top_n_words"].str.split(" - ")

    df = df.merge(topic_df, left_index=True, right_index=True)
    df = df.drop(columns=[f"{index_col}_y"])
    df = df.rename(columns={f"{index_col}_x": index_col})

    return df


def create_clips(
    episode_df, args, n_clips=constants.N_CLIPS, clip_length=constants.CLIP_LENGTH
):
    ##################
    # Debugging on subset
    # episode_df = episode_df[:100]
    ##################

    print(f"Processing {len(episode_df)} episodes")
    print(
        f"episode duration: mean={np.mean(episode_df.duration)}, std={np.std(episode_df.duration)}"
    )

    if not os.path.exists(args.clip_path):
        clip_df = episode_df.loc[episode_df.index.repeat(n_clips)].reset_index(
            drop=False
        )

        new_columns = []
        repeat_counter = collections.defaultdict(int)
        for i, row in tqdm.tqdm(clip_df.iterrows(), total=len(clip_df)):
            episode_uri = row["episode_uri"]
            episode_duration = row["duration"]
            clip_num = repeat_counter[episode_uri]

            random.seed(clip_num)
            start_timestamp = random.uniform(0, episode_duration * 60.0)
            end_timestamp = start_timestamp + clip_length

            processed_transcript = extract_clip_transcript(
                episode_uri, start_timestamp, end_timestamp
            )

            new_columns.append(
                {
                    "episode_uri": episode_uri,
                    "episode_clip_uri": f"{episode_uri}:{clip_num}",
                    "clip_num": clip_num,
                    "transcript": " ".join(processed_transcript),
                    "start_timestamp": start_timestamp,
                    "end_timestamp": end_timestamp,
                }
            )
            repeat_counter[episode_uri] += 1

        clip_df = clip_df.drop_duplicates()
        clip_df = clip_df.merge(
            pd.DataFrame(new_columns), on="episode_uri", how="right"
        )

        print(f"==> Generated {len(clip_df)} clips! Dropping empty transcripts...")
        clip_df = clip_df[clip_df["transcript"] != ""]

        clip_df.to_csv(args.clip_path, index=False)
    else:
        clip_df = pd.read_csv(args.clip_path)

    print(f"==> Now have {len(clip_df)} clips!")
    # print("==> Tokenizing transcript")
    # clip_df["tokenized_transcript"] = clip_df["transcript"].apply(tokenize)

    print("==> Extracting clip vocab")
    if not os.path.exists(args.keybert_vocab_path):
        vocabulary = extract_vocab(df=clip_df, index_col="transcript", args=args)
    else:
        with open(args.keybert_vocab_path, "rb") as f:
            vocabulary = [line.decode("utf-8").rstrip() for line in f]

    print("==> Extracting clip topics")
    clip_df = extract_clip_topics(
        df=clip_df, index_col="transcript", args=args, vocabulary=vocabulary
    )

    counts = np.unique(clip_df["episode_uri"].values, return_counts=True)[1]
    print(counts)
    print(n_clips)
    # assert (counts == n_clips).all()

    return clip_df


def main(args):
    print(f"Loading raw data from {args.raw_path}")
    df = pd.read_csv(args.raw_path, sep="\t", on_bad_lines="skip")

    print("=> Creating clips")
    df = create_clips(df, args)

    print("Finished processing clip dataframe!")
    print(f"Saving processed data to {args.clip_topics_path}")
    df.to_csv(args.clip_topics_path, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--raw_path",
        help="raw data path",
        required=True,
    )
    parser.add_argument(
        "--clip_path",
        help="processed clip data path",
        required=True,
    )
    parser.add_argument(
        "--keybert_vocab_path",
        help="path to save keybert vocab to",
        required=True,
    )
    parser.add_argument(
        "--bertopic_model_path",
        help="path to save bertopic model to",
        required=True,
    )
    parser.add_argument(
        "--clip_topics_path",
        help="processed clip data path",
        required=True,
    )

    args = parser.parse_args()

    main(args)
