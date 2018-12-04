# -*- coding: utf-8 -*-
"""
comments
author: diqiuzhuanzhuan
email: diqiuzhuanzhuan@gmail.com

"""

import collections


class NumberAdapter(object):

    _data = collections.OrderedDict({
        "零": 0,
        "0": 0,
        "一": 1,
        "1": 1,
        "壹": 1,
        "二": 2,
        "2": 2,
        "贰": 2,
        "三": 3,
        "3": 3,
        "叁": 3,
        "四": 4,
        "4": 4,
        "肆": 4,
        "五": 5,
        "5": 5,
        "伍": 5,
        "六": 6,
        "6": 6,
        "陆": 6,
        "七": 7,
        "7": 7,
        "柒": 7,
        "八": 8,
        "8": 8,
        "捌": 8,
        "九": 9,
        "9": 9,
        "玖": 9,
        "十": 10,
        "拾": 10,
        "百": 100,
        "佰": 100,
        "千": 1000,
        "仟": 1000,
        "万": 10000,
        "萬": 10000,
        "亿": 100000000,
    })

    def __init__(self):
        pass

    @classmethod
    def convert(cls, text):
        if not isinstance(text, str):
            raise ValueError("{} is not str".format(type(text)))
        t = text.replace(' ', '')
        return cls._convert(t)

    @classmethod
    def _convert(cls, text):
        value = 0
        last_actor = 0
        guess_next_unit = 1
        for i, s in enumerate(text):
            t = cls._data[s]
            if t < 10:
                last_actor = 10 * last_actor + t
            else:
                if last_actor == 0:
                    value *= t
                else:
                    value += last_actor * t
                last_actor = 0
            if i == len(text) - 2 and t > 10:
                guess_next_unit = cls._data[s] // 10 or 1
        if last_actor:
            value += last_actor * guess_next_unit
        return value


if __name__ == "__main__":
    test_data = [
        ("一万五", 15000),
        ("一万五百", 10500),
        ("一万零五", 10005),
        ("一百万", 1000000),
        ("一百万五千", 1005000),
        ("一百万零五千", 1005000),
        ("一八三零", 1830),
        ("3000", 3000),
        ("五百", 500)
    ]
    for ele in test_data:
        print("expect is {}, actually is {}".format(ele[1], NumberAdapter.convert(ele[0])))
        assert(NumberAdapter.convert(ele[0]) == ele[1])
