# -*- coding: utf-8 -*-
"""
comments
author: diqiuzhuanzhuan
email: diqiuzhuanzhuan@gmail.com

"""
import os
import google

PS_OPS = [
    'Variable', 'VariableV2', 'AutoReloadVariable', 'MutableHashTable',
    'MutableHashTableOfTensors', 'MutableDenseHashTable'
]

def mount_google_drive():
    try:
        from google.colab import drive
        drive.mount('/content/drive')
    except Exception as e:
        print("not in googole colab environment!")

def google_colab_authenticate_user():
    try:
        from google.colab import auth
        auth.authenticate_user()
    except Exception as e:
        print("not in googole colab environment!")

