
This is a project for later lazy work!


#Install
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

# bert_model
usage:
- prepare a trained model, tell classifier model
- prepare train.csv and test.csv, its format is like this: "id, text1, label"
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
     label_list=[0, 1, 2, 3]                                                  
    )
````
    
  
# poros_chars
Provide a list of small functions

usage:
- convert chinese words into arabic number:
```python
from poros.poros_chars import chinese_to_arabic
>>> print(chinese_to_arabic.NumberAdapter.convert("四千三百万"))
43000000

```


