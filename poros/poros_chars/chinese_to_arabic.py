# -*- coding: utf-8 -*-
"""
comments
author: diqiuzhuanzhuan
email: diqiuzhuanzhuan@gmail.com

"""

import collections


class NumberAdapter(object):

    _data = collections.OrderedDict({
        "个": -1,
        "零": 0,
        "又": 0,
        "0": 0,
        "一": 1,
        "1": 1,
        "壹": 1,
        "二": 2,
        "两": 2,
        "俩": 2,
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
        "億": 100000000,
        "兆": 1000000000000,
        "点": -2,
        ".": -2
    })

    def __init__(self):
        pass

    @classmethod
    def convert(cls, text):
        if not isinstance(text, str):
            raise ValueError("{} is not str".format(type(text)))
        t = text.replace(' ', '')
        new_t = []
        for i, _ in enumerate(t):
            if _ in cls._data and 1 < i < (len(t) - 1):
                if cls._data[_] == 10 and cls._data[t[i-1]] > 0 and cls._data[t[i+1]] < 10:
                    continue
            new_t.append(_)
        return cls._convert("".join(new_t))

    @classmethod
    def _convert(cls, text):
        value = 0
        last_actor = 0
        guess_next_unit = 1
        denominator_actor = 0
        for i, s in enumerate(text):
            t = cls._data.get(s, -1)
            if t == -1:
                continue
            if t == -2:
                denominator_actor = 1
            elif t < 10:
                denominator_actor = 10 * denominator_actor
                last_actor = 10 * last_actor + t
            else:
                if last_actor == 0:
                    value = (value or 1) * t
                elif (value > 0) and (value < t):
                    value = (value + last_actor) * t
                else:
                    value += last_actor * t / (denominator_actor or 1)
                last_actor = 0
                denominator_actor = 0
            if i == len(text) - 2 and t > 10:
                guess_next_unit = cls._data[s] // 10 or 1
        if last_actor:
            value += last_actor / (denominator_actor or 1) * guess_next_unit
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
        ("五百", 500),
        ("五个百", 500),
        ("五个亿", 500000000),
        ("四千万又五十三", 40000053),
        ("二点五万", 25000),
        ("9.5万", 95000),
        ("九.5万", 95000),
        ("三十点四八", 30.48),
        ("两点九", 2.9),
        ("十万零三千六百零九", 103609),
        ("一七六七四点四四", 17674.44),
        ("两百三十千点五", 230000.5),
        ("12.60", 12.60),
        ("零点四一四一四", 0.41414),
        ("两五万五千点九", 255000.9),
        ("一千零一万", 10010000),
        ("两十五万五千", 255000),
        ("两拾", 20),
        ("两拾万", 200000),
        ("afafdsaf一百万", 1000000),
        ("五十一亿零一", 5100000001),
        ("五十万一千", 501000),
        ("六千三百五十一万六十七", 63510067),
        ("六千三百五十一万零六十七", 63510067),
        ("五十一亿零一万零五百", 5100010500),
        ("三万二", 32000),
        ("五十一亿五百", 5100000500),
        ("五亿五十五万五千", 500555000)
    ]
    for ele in test_data:
        print("input is {}, expect is {}, actually is {}".format(ele[0], ele[1], NumberAdapter.convert(ele[0])))
        assert(NumberAdapter.convert(ele[0]) == ele[1])
    print("everything is ok? so good!\n~~I love you~~")
