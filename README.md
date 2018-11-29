
This is a project for later lazy work!


Some code is from other people, and some is from me.

# bert_model
usage:
- prepare a trained model, tell classifier model
- prepare train.csv and test.csv, its format is like this: "id, text1, label"
- init the model, the code is like below
    model = SimpleClassifierModel(
    bert_config_file="./data/chinese_L-12_H-768_A-12/bert_config.json",      
     vocab_file="./data/chinese_L-12_H-768_A-12/vocab.txt",                   
     output_dir="./output",                                                   
     max_seq_length=512,                                                      
     train_file="./data/train.csv",                                           
     dev_file="./data/dev.csv",                                               
     init_checkpoint="./data/chinese_L-12_H-768_A-12/bert_model.ckpt",        
     label_list=[0, 1, 2, 3]                                                  
    )
    
  
