
This is a project for later lazy work!
Only support for python3, ☹️, but maybe you can try in python2

# Install
命令行直接安装
```bash
pip install poros
```
从代码库安装
```bash
git clone https://github.com/diqiuzhuanzhuan/poros.git
cd poros
python setup install
```

Some code is created by myself, and some code is inspired by others,  such as allennlp etc.

# poros_chars
Provide a set of small functions

usage:
- convert Chinese words into Arabic numbers:
```python
from poros.poros_chars import chinese_to_arabic
>>> print(chinese_to_arabic.NumberAdapter.convert("四千三百万"))
43000000

```
# poros_loss
Provide some loss functions, such as gravity loss, and dice loss usage:
- 
```python
from poros.poros_loss import GravityLoss
>>> gl = GravityLoss()
        # [1, 2]
>>> input_a = torch.tensor([[1.0, 1]], requires_grad=True)
>>> input_b = torch.tensor([[1.0, 1]], requires_grad=True)
>>> target = torch.tensor([[4.0]])
>>> output = gl(input_a, input_b, target)
>>> torch.testing.assert_close(output, target)
```

# clustering 
```python
from poros.poros_common.params import Params

# some cluster algorithm, graph cluster
>>> sentence_embedding_params = Params({
        'type': 'sentence_transformers_model', 
        'model_name_or_path': 'albert-base-v1'
        })
    # it indicates that: sentence_embedding_model = SentenceEmbeddingModel.from_params(params=sentence_embedding_params)
>>> clustering_algorithm_params = Params({
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

>>> intent_clustering_params = Params({
        'type': 'baseline_intent_clustering_model',
        'clustering_algorithm': clustering_algorithm_params,
        'embedding_model': sentence_embedding_params
    })

>>> intent_clustering_model = IntentClusteringModel.from_params(params=intent_clustering_params)
```

# Thanks
PyCharm, Mircosoft Visual Studio Code