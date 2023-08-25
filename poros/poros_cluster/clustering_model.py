# -*- coding: utf-8 -*-
# author: Feynman
# email: diqiuzhuanzhuan@gmail.com

from asyncio.log import logger
import functools
from importlib.resources import path
from xml.dom import minicompat
from poros.poros_common.registrable import Registrable
from poros.poros_common.params import Params
from typing import Dict, List, Set, Any, Optional, AnyStr
from dataclasses import dataclass, field
from pathlib import Path
from numpy import ndarray
import numpy as np
from sklearn.cluster import KMeans, OPTICS, DBSCAN, BisectingKMeans
from sklearn.metrics import pairwise_distances
import networkx as nx
from cdlib import algorithms
from poros.poros_cluster import SentenceEmbeddingModel
from sklearn.cluster import MiniBatchKMeans, KMeans, SpectralClustering
from poros.poros_cluster.similarity_matrix import ClusterSimilarityMatrix, EnsembleCustering
from scipy.sparse.csgraph import connected_components
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


@dataclass
class ClusteringContext:
    """
    Wrapper for ndarray containing clustering inputs.
    """
    features: ndarray
    # output intermediate clustering results/metadata here
    output_dir: Path = None
    # dynamically inject parameters to clustering algorithm here
    parameters: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ClusterData:
    """
    Wrapper class for cluster labels.
    """
    clusters: List[int]


class ClusteringAlgorithm(Registrable):

    def cluster(self, context: ClusteringContext) -> ClusterData:
        """
        Predict cluster labels given a clustering context consisting of raw features and any parameters
        to dynamically pass to the clustering algorithm.
        :param context: clustering context
        :return: cluster labels
        """
        raise NotImplementedError

@ClusteringAlgorithm.register('density_based_clustering')
class DensityBasedClustering(ClusteringAlgorithm):

    CLUSTERING_ALGORITHM_BY_NAME = {
        'kmeans': KMeans,
        'dbscan': DBSCAN,
        'optics': OPTICS,
        'bisectingkmeans': BisectingKMeans,
    }

    def __init__(
        self,
        clustering_algorithm_name: str,
        clustering_algorithm_params: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Initialize a clustering algorithm with scikit-learn `ClusterMixin` interface.
        :param clustering_algorithm_name: key for algorithm, currently supports 'kmeans', 'dbscan', 'optics' and 'bisectingkmeans'
        :param clustering_algorithm_params: optional constructor parameters used to initialize clustering algorithm
        """
        super().__init__()
        # look up clustering algorithm by key
        clustering_algorithm_name = clustering_algorithm_name.lower()
        if clustering_algorithm_name not in DensityBasedClustering.CLUSTERING_ALGORITHM_BY_NAME:
            raise ValueError(f'Clustering algorithm "{clustering_algorithm_name}" not supported')
        self._constructor = DensityBasedClustering.CLUSTERING_ALGORITHM_BY_NAME[clustering_algorithm_name]
        if not clustering_algorithm_params:
            clustering_algorithm_params = {}
        self._clustering_algorithm_params = clustering_algorithm_params

    def cluster(self, context: ClusteringContext) -> ClusterData:
        # combine base parameters with any clustering parameters from the clustering context
        params = {**self._clustering_algorithm_params.copy(), **context.parameters}
        # initialize the clustering algorithm
        algorithm = self._constructor(**params)
        # predict and return cluster labels
        labels = algorithm.fit_predict(context.features).tolist()
        return ClusterData(labels)


@ClusteringAlgorithm.register('graph_based_clustering')
class GraphBasedClusetring(ClusteringAlgorithm):
    SIMILARITY_ALGORITHM_BY_NAME = {
        'cosine': functools.partial(pairwise_distances, metric='cosine'),
        'euclidean': functools.partial(pairwise_distances, metric='euclidean'),
        'l1': functools.partial(pairwise_distances, metric='l1'),
        'l2': functools.partial(pairwise_distances, metric='l2'),
        'manhattan': functools.partial(pairwise_distances, metric='manhattan'),
    }

    COMMUNITY_DETECTION_BY_NAME = {
        'louvain':algorithms.louvain,
        'leiden': algorithms.leiden 
    }

    def __init__(
        self,
        similarity_algorithm_name: str = 'cosine',
        similarity_algorithm_params: Optional[Dict[str, Any]] = None,
        community_detection_name: str = 'leiden',
        community_detection_params: Optional[Dict[str, Any]] = None,
    ) -> None:
        """_summary_

        Args:
            similarity_algorithm_name (str, optional): similarity function of two vectors, currently support 'cosine', 'euclidean', 'l1', 'l2', 'manhattan', Defaults to 'cosine'.
            similarity_algorithm_params (Optional[Dict[str, Any]], optional):  Defaults to None.
            community_detection_name (str, optional): algorithm to optimize graph, currntly support 'louvain', 'leiden',  Defaults to 'leiden'.
            community_detection_params (Optional[Dict[str, Any]], optional): _description_. Defaults to None.

        Raises:
            ValueError: _description_
        """
        super().__init__()
        # look up clustering algorithm by key
        similarity_algorithm_name = similarity_algorithm_name.lower()
        if similarity_algorithm_name not in GraphBasedClusetring.SIMILARITY_ALGORITHM_BY_NAME:
            raise ValueError(f'Similarity algorithm "{similarity_algorithm_name}" not supported')
        self._similarity = GraphBasedClusetring.SIMILARITY_ALGORITHM_BY_NAME[similarity_algorithm_name]
        if not similarity_algorithm_params:
            similarity_algorithm_params = {}
        self._similarity_algorithm_params = similarity_algorithm_params

        community_detection_name = community_detection_name.lower()
        if community_detection_name not in GraphBasedClusetring.COMMUNITY_DETECTION_BY_NAME:
            raise ValueError(f'Community detection "{community_detection_name}" not supported')
        self._community_detection = GraphBasedClusetring.COMMUNITY_DETECTION_BY_NAME[community_detection_name]
        if not community_detection_params:
            community_detection_params = {}
        self._community_detection_params = community_detection_params
        
    def _build_graph(self, matrix):
        g = nx.Graph()
        m = len(matrix)
        g.add_nodes_from(range(m))
        for i in range(m):
            for j in range(i):
                g.add_edge(i, j, weight=np.exp(-matrix[i][j]+3)) ## scale up
                #g.add_edge(i, j, weight=matrix[i][j])
        return g
    
    def cluster(self, context: ClusteringContext) -> ClusterData:
        params = {**self._similarity_algorithm_params.copy()}
        similarity_matrix = self._similarity(X=context.features, **params)
        g = self._build_graph(similarity_matrix)
        params = {**self._community_detection_params.copy()}
        coms = self._community_detection(g, **params)
        cluster_labels = [0] * len(context.features)
        for i, l in enumerate(coms.communities):
            for j in l:
                cluster_labels[j] = i
        return ClusterData(cluster_labels)


@ClusteringAlgorithm.register('ensemble_clustering')
class EnsembleClustering(ClusteringAlgorithm):
    def __init__(self, num_kemans=100, min_probability=0.5, agg='spectral_cluster') -> None:
        super().__init__()
        self.agg = agg
        self.num_kemans = num_kemans
        self.min_probability = min_probability
    
    def cluster(self, context: ClusteringContext)  -> ClusterData:
    
        # Generating a "Cluster Forest"
        clustering_models = self.num_kemans*[
            # Note: Do not set a random_state, as the variability is crucial
            # This is a extreme simple K-Means
            MiniBatchKMeans(n_clusters=32, batch_size=128, n_init=1, max_iter=20)
        ]
    
        clt_sim_matrix = ClusterSimilarityMatrix()
        for model in clustering_models:
            x = model.fit_predict(X=context.features)
            clt_sim_matrix.fit(x)
    
        sim_matrix = clt_sim_matrix.similarity
        norm_sim_matrix = sim_matrix/sim_matrix.diagonal()
    
        # Transforming the probabilities into graph edges
        # This is very similar to DBSCAN
        graph = (norm_sim_matrix > self.min_probability).astype(int)
    
        # Extractin the connected components
        n_clusters, y_ensemble = connected_components(graph, directed=False, return_labels=True)
    
        if self.agg == 'spectral_cluster':
            aggregator_clt = SpectralClustering(n_clusters=n_clusters, affinity="precomputed")
            ens_clt = EnsembleCustering(clustering_models, aggregator_clt)
            y_ensemble = ens_clt.fit_predict(context.features)
    
        # Default K-Means
        y_kmeans = KMeans(n_clusters=3).fit_predict(context.features)
        return ClusterData(y_ensemble)


@ClusteringAlgorithm.register('ensemble_clustering_from_result')
class EnsembleClusteringFromResult(ClusteringAlgorithm):
    def __init__(self, result_path: Path, min_probability=0.9, agg='spectral_cluster', grep='finance') -> None:
        super().__init__()
        self.result_path = result_path
        self.min_probability = min_probability
        self.agg = agg
        self.grep = grep


    def _traverse(self, path: Path, ans: List[Path]):
        if not isinstance(path, Path):
            path = Path(path)
        if path.is_dir():
            for p in path.iterdir():
                if not self.grep in p.as_posix():
                    continue
                if p.is_dir():
                    self._traverse(p, ans)
                elif p.as_posix().endswith("y_pred.npy"):
                    ans.append(p)
                else:
                    # file we don't care about
                    pass

    def cluster(self, context: ClusteringContext) -> ClusterData:
    
        clt_sim_matrix = ClusterSimilarityMatrix()
        files = []
        self._traverse(self.result_path, files)
        logger.info('reading {} files'.format(len(files)))
        for file in files:
            x = np.load(file)
            clt_sim_matrix.fit(x)
    
        sim_matrix = clt_sim_matrix.similarity
        norm_sim_matrix = sim_matrix/sim_matrix.diagonal()
    
        # Transforming the probabilities into graph edges
        # This is very similar to DBSCAN
        graph = (norm_sim_matrix > self.min_probability).astype(int)
    
        # Extractin the connected components
        n_clusters, y_ensemble = connected_components(graph, directed=False, return_labels=True)
        logger.info('predict K is {}'.format(n_clusters))
    
        if self.agg == 'spectral_cluster':
            aggregator_clt = SpectralClustering(n_clusters=n_clusters, affinity="precomputed")
            cluster_matrix = sim_matrix / sim_matrix.diagonal()
            #cluster_matrix = np.abs(np.log(cluster_matrix + 1e-8)) # Avoid log(0)
            y_ensemble = aggregator_clt.fit_predict(cluster_matrix)
        
        return  ClusterData(y_ensemble)


@dataclass
class IntentClusteringContext:
    """
    Dialogue clustering context consisting of a list of dialogues and set of target turn IDs to be labeled
    with clusters.
    """
    dataset: List[AnyStr]
    labels: List[AnyStr]
    # output intermediate clustering results/metadata here
    output_dir: Path = None

class IntentClusteringModel(Registrable):

    def cluster_intents(self, context: IntentClusteringContext) -> Dict[str, str]:
        """
        Assign cluster IDs to intent turns within a collection of dialogues.

        :param context: dialogue clustering context

        :return: assignment of turn IDs to cluster labels
        """
        raise NotImplementedError


@IntentClusteringModel.register('baseline_intent_clustering_model')
class BaselineIntentClusteringModel(IntentClusteringModel):

    def __init__(
        self,
        clustering_algorithm: ClusteringAlgorithm,
        embedding_model: SentenceEmbeddingModel,
    ) -> None:
        """
        Initialize intent clustering model based on clustering utterance embeddings.
        :param clustering_algorithm: clustering algorithm applied to sentence embeddings
        :param embedding_model: sentence embedding model
        """
        super().__init__()
        self._clustering_algorithm = clustering_algorithm
        self._embedding_model = embedding_model

    def cluster_intents(self, context: IntentClusteringContext) -> Dict[str, str]:
        # collect utterances corresponding to intents
        utterances = []
        labels = set()
        for utterance, label in zip(context.dataset, context.labels):
            utterances.append(utterance)
            labels.update({label})

        # compute sentence embeddings
        features = self._embedding_model.encode(utterances)
        # cluster sentence embeddings
        context = ClusteringContext(
            features,
            output_dir=context.output_dir,
        )
        result = self._clustering_algorithm.cluster(context)
        # map turn IDs to cluster labels
        return [str(label) for label in result.clusters]
        
        
if __name__ == "__main__":
    sentence_embedding_params = Params({
        'type': 'sentence_transformers_model', 
        'model_name_or_path': 'albert-base-v1'
    })
    # it indicates that: sentence_embedding_model = SentenceEmbeddingModel.from_params(params=sentence_embedding_params)

    clustering_algorithm_params = Params({
        'type': 'graph_based_clustering',
        'similarity_algorithm_name': 'cosine',
        'similarity_algorithm_params': None,
        'community_detection_name': 'louvain',
        'community_detection_params': {
            'weight': 'weight', 
            'resolution': 0.95, 
            'randomize': False
        } 
    })
    '''
    clustering_algorithm_params = Params({
        'type': 'density_based_clustering',
        'clustering_algorithm_name': 'kmeans',
        'clustering_algorithm_params': {
            'n_init': 2
        }
    })
    clustering_algorithm_params = Params({
        'type': 'ensemble_clustering',
        'num_kemans': 500
        }
    )

    clustering_algorithm_params = Params({
        'type': 'ensemble_clustering_from_result',
        'result_path': '/Users/malong/github/dstc11/TEXTOIR/open_intent_discovery/results'
    })
    '''

    # it indicates that: clustering_algorithm = ClusteringAlgorithm.from_params(params=clustering_algorithm_params)

    intent_clustering_params = Params({
        'type': 'baseline_intent_clustering_model',
        'clustering_algorithm': clustering_algorithm_params,
        'embedding_model': sentence_embedding_params
    })

    intent_clustering_model = IntentClusteringModel.from_params(params=intent_clustering_params)
    cluster_context = IntentClusteringContext(
        dataset=['你好吗'] * 150 + ['我很不好'] * 150,
        labels=[1] * 150 + [2] * 150,
        output_dir=Path(".")
    )
    labels = intent_clustering_model.cluster_intents(cluster_context)
    print(labels)
    