# -*- coding: utf-8 -*-
import sys
import uuid
import requests
import hashlib
import time
from imp import reload

import time

reload(sys)

YOUDAO_URL = 'https://openapi.youdao.com/api'
APP_KEY = '58056f9b9a194809'
APP_SECRET = 'wKAkqnlEgfaMtzBnSkAGbF4nUorBrAdk'


def encrypt(signStr):
    hash_algorithm = hashlib.sha256()
    hash_algorithm.update(signStr.encode('utf-8'))
    return hash_algorithm.hexdigest()


def truncate(q):
    if q is None:
        return None
    size = len(q)
    return q if size <= 20 else q[0:10] + str(size) + q[size - 10:size]


def do_request(data):
    headers = {'Content-Type': 'application/x-www-form-urlencoded'}
    return requests.post(YOUDAO_URL, data=data, headers=headers)


def connect(q):
    # q = """Luo Feng is the true God of the king, there are hundreds of millions of life genes for themselves.\n\nOne of the incarnations of the true God of the Uhai is the Nine U"""

    data = {}

    data['signType'] = 'v3'
    curtime = str(int(time.time()))
    data['curtime'] = curtime
    salt = str(uuid.uuid1())
    signStr = APP_KEY + truncate(q) + salt + curtime + APP_SECRET
    sign = encrypt(signStr)
    data['appKey'] = APP_KEY
    data['q'] = q
    data['salt'] = salt
    data['sign'] = sign
    data['vocabId'] = "您的用户词表ID"

    response = do_request(data)
    contentType = response.headers['Content-Type']
    # print(response.json()['translation'])
    # print(response['translation'])
    return response.json()['translation'][0]

if __name__ == '__main__':
    r = connect("翻译一下 这是什么意思")
    print(r)