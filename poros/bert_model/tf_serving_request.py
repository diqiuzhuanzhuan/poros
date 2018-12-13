# -*- coding: utf-8 -*-
"""
comments
author: diqiuzhuanzhuan
email: diqiuzhuanzhuan@gmail.com

"""

from poros.bert_model.run_classifier import JustClassifierDataProcessor, convert_single_example, convert_feature_to_tf_example
from poros.bert_model.tokenization import FullTokenizer


def main():
    lines = ["你好", "this is her file"]
    jc = JustClassifierDataProcessor()
    jc.set_labels(["0", "1", "2", "3"])
    examples = jc.get_test_examples_not_from_file(lines)
    tokenizer = FullTokenizer(vocab_file="./test_data/vocab.txt")
    import time
    t1 = time.time()
    for i, j in enumerate(examples):
        feature = convert_single_example(i, j, ["0", "1"], 512, tokenizer=tokenizer)
        tf_example = convert_feature_to_tf_example(feature=feature)
        _string = tf_example.SerializeToString()
        import base64
        _string = base64.b64encode(_string)

        data = {"instances": [{"b64": _string.decode("utf-8")}]}
        print(data)
        url = "http://localhost:8501/v1/models/export:predict"
        import requests
        import json
        response = requests.post(url=url, data=json.dumps(data))
        print(response.text)
    costs = time.time() - t1
    print("costs is {} sec".format(costs))


if __name__ == "__main__":
    main()