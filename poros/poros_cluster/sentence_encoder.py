# -*- coding: utf-8 -*-
# author: Feynman
# email: diqiuzhuanzhuan@gmail.com

from typing import AnyStr, List
from sentence_transformers import SentenceTransformer
import numpy as np
from poros.poros_common import registrable
from poros.poros_common import Params
from sklearn.feature_extraction.text import TfidfVectorizer


class SentenceEmbeddingModel(registrable.Registrable):
    def encode(self, utterances: List[str]) -> np.ndarray:
        """
        Encode a list of utterances as an array of real-valued vectors.
        :param utterances: original utterances
        :return: output encoding
        """
        raise NotImplementedError


@SentenceEmbeddingModel.register('sentence_transformers_model')
class SentenceTransformersModel(SentenceEmbeddingModel):

    def __init__(self, model_name_or_path: str) -> None:
        """
        Initialize SentenceTransformers model for a given path or model name.
        :param model_name_or_path: model name or path for SentenceTransformers sentence encoder
        """
        super().__init__()
        self._sentence_transformer = model_name_or_path

    def encode(self, utterances: List[str]) -> np.ndarray:
        encoder = SentenceTransformer(self._sentence_transformer)
        return encoder.encode(utterances) 


@SentenceEmbeddingModel.register('tf_idf_model')
class TfIdfModel(SentenceEmbeddingModel):

    def __init__(self, model_name_or_path: AnyStr=None) -> None:
        """_summary_

        Args:
            model_name_or_path (AnyStr): model path from which a tfidf model can initialize
        """
        super().__init__()
        

    def encode(self, utterances: List[str]) -> np.ndarray:
        encoder = TfidfVectorizer()
        encoder.fit_transform(utterances)
        return encoder.transform(utterances)
    


if __name__ == "__main__":
    params = Params({
        'type': 'sentence_transformers_model', 
        'model_name_or_path': 'albert-base-v1'
    })
    sentence_model = SentenceEmbeddingModel.from_params(params=params)
    utterances = ['how to add this song into playlist?']
    print(sentence_model.encode(utterances))
    utterances = ['to add this song into playlist?']
    print(sentence_model.encode(utterances))
    params = Params({
        'type': 'tf_idf_model',
        'model_name_or_path': None
    })
    tf_idf_model = SentenceEmbeddingModel.from_params(params=params)
    utterances = ['how to add this song into playlist?', 'where is the menu？']
    print(tf_idf_model.encode(utterances))