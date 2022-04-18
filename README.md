
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

# poros_chars
Provide a list of small functions

usage:
- convert chinese words into arabic number:
```python
from poros.poros_chars import chinese_to_arabic
>>> print(chinese_to_arabic.NumberAdapter.convert("四千三百万"))
43000000

```
# poros_loss
Provide some loss functions, such as gravity loss, dice loss
usage:
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

# Thanks
PyCharm, Mircosoft Visual Studio Code