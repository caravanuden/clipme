import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
# import stanza
import torch
# nlp = stanza.Pipeline(lang='en', processors='tokenize', tokenize_pretokenized=True)


class EmbedSimilarityBaseline:
    def __init__(self, embedding_dim=0):
        self.index = faiss.IndexFlatL2(embedding_dim)

    def add_to_index(self, embedding):
        self.index.add(embedding)

    def retrieve_from_index(self, new_embeds, num_k):
        D, I = self.index.search(new_embeds, num_k + 1) # since the closest neighbor is always itself
        corresponding_indices = I[:, 1:]
        corresponding_scores = D[:, 1:]
        return corresponding_scores, corresponding_indices


class SentenceBERTSimilarityBaseline(EmbedSimilarityBaseline):
    def __init__(self, embedding_dim=384, model_name="all-MiniLM-L6-v2"):
        super(SentenceBERTSimilarityBaseline, self).__init__(embedding_dim)
        self.model = SentenceTransformer(model_name)
        if torch.cuda.is_available():
            self.model.to("cuda:0")
        self.index_list = []

    def process_clip(self, new_clip):
        """
        Transform a clip worth of transcripts into its corresponding sentence representations.
        :param new_clip: Transcript sentences.
        """
        # separate into individual sentences
        # doc = nlp(new_clip)
        # all_sentences = [s.text for s in doc.sentences]
        # sentence_embeds = self.model(all_sentences)
        # sentence_embeds = np.mean(sentence_embeds, axis=0)
        sentence_embeds = self.model.encode([new_clip])
        return sentence_embeds

    def add_clip_to_index(self, new_clip):
        """
        Add the new representation to the Faiss index.
        :param new_clip: Transcript sentences.
        """
        new_sentence_embeds = self.process_clip(new_clip)
        self.add_to_index(new_sentence_embeds)
        self.index_list.append(new_sentence_embeds)






