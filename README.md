
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


Some code is from other people, and some is from me.

# unilmv2
- create pretraining data
```python
from poros.unilmv2.dataman import PreTrainingDataMan
# vocab_file: make sure [SOS],[EOS] and [Pseudo] are in vocab_file
vocab_file = "vocab_file"# your vocab file
ptdm = PreTrainingDataMan(vocab_file=vocab_file, max_seq_length=128, max_predictions_per_seq=20, random_seed=2334)
input_file = "my_input_file" #file format is like bert
output_file = "my_output_file" #the output file is a tfrecord file
ptdm.create_pretraining_data(input_file, output_file)
dataset = ptdm.read_data_from_tfrecord(output_file, is_training=True, batch_size=8)
```
- create unilmv2 model and train it
```python
from poros.unilmv2.config import Unilmv2Config
from poros.unilmv2 import Unilmv2Model
from poros_train import optimization

"""
the configuration is like this:
{
  "attention_probs_dropout_prob": 0.1,
  "directionality": "bidi", 
  "hidden_act": "gelu",
  "hidden_dropout_prob": 0.1,
  "hidden_size": 768, 
  "initializer_range": 0.08,
  "intermediate_size": 3072, 
  "max_position_embeddings": 512, 
  "num_attention_heads": 12, 
  "num_hidden_layers": 12,
  "pooler_fc_size": 768,
  "pooler_num_attention_heads": 12,
  "pooler_num_fc_layers": 3, 
  "pooler_size_per_head": 128, 
  "pooler_type": "first_token_transform",
  "type_vocab_size": 2, 
  "vocab_size": 21131
}
A json file recording these configuration is recommended.
"""
json_file = "my_config_file"
unilmv2_config = Unilmv2Config.from_json_file(json_file)
unilmv2_model = Unilmv2Model(config=unilmv2_config, is_training=True)
epoches=2000
steps_per_epoch=15
optimizer = optimization.create_optimizer(init_lr=6e-4, num_train_steps=epoches * steps_per_epoch, num_warmup_steps=1500)
unilmv2_model.compile(optimizer=optimizer)
unilmv2_model.fit(dataset, epochs=epoches, steps_per_epoch=15)
```

# bert
usage:
- create pretrain data

```python
from poros.bert import create_pretraining_data

>>> create_pretraining_data.main(input_file="./test_data/sample_text.txt",
output_file="./test_data/output", vocab_file="./test_data/vocab.txt")
```

- pretrain bert model
```python
from poros.bert_model import pretrain
>>> pretrain.run(input_file="./test_data/output",  bert_config_file="./test_data/bert_config.json", 
output_dir="./output")

```
- prepare a trained model, tell classifier model
- prepare train.csv and test.csv, its format is like this: "id, text1, label", but remember no header!
- init the model, the code is like below
````python
from poros.bert_model.run_classifier import SimpleClassifierModel
>>> model = SimpleClassifierModel(
    bert_config_file="./data/chinese_L-12_H-768_A-12/bert_config.json",      
     vocab_file="./data/chinese_L-12_H-768_A-12/vocab.txt",                   
     output_dir="./output",                                                   
     max_seq_length=512,                                                      
     train_file="./data/train.csv",                                           
     dev_file="./data/dev.csv",                                               
     init_checkpoint="./data/chinese_L-12_H-768_A-12/bert_model.ckpt",        
     label_list=["0", "1", "2", "3"]                                                  
    )
````

# poros_dataset
some operations about tensor
```python
from poros.poros_dataset import about_tensor
import tensorflow as tf
>>> A = tf.constant(value=[0])
>>> print(about_tensor.get_shape(A))
[1]

```
    
  
# poros_chars
Provide a list of small functions

usage:
- convert chinese words into arabic number:
```python
from poros.poros_chars import chinese_to_arabic
>>> print(chinese_to_arabic.NumberAdapter.convert("四千三百万"))
43000000

```

