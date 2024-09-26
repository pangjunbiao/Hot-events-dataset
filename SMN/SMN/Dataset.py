#!/usr/bin/env python
# coding: utf-8

import os
import time
import gensim
import pickle
import random
import torch
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

from gensim.corpora import Dictionary
from gensim.models import TfidfModel
from collections import Counter
from itertools import chain
import numpy as np
import sys
from tokenization import *

sys.stdout = open(sys.stdout.fileno(), mode='w', encoding='utf8', buffering=1)


# 传入一个taskname
class DocDataset(Dataset):
    def __init__(self, taskname, txtPath=None, lang="zh", tokenizer=None, stopwords=None, no_below=5, no_above=0.1,
                 hasLable=False, rebuild=False, use_tfidf=False):
        cwd = os.getcwd()  # 返回当前路径
        txtPath = os.path.join(cwd, 'data', f'{taskname}_lines.txt') if txtPath == None else txtPath
        tmpDir = os.path.join(cwd, 'data', taskname)  # 创建文件夹
        self.txtLines = [line.strip('\n') for line in open(txtPath, 'r', encoding='utf-8')]  # 逐行读title
        self.dictionary = None
        self.bows, self.docs = None, None
        self.use_tfidf = use_tfidf
        self.tfidf, self.tfidf_model = None, None
        if not os.path.exists(tmpDir):
            os.mkdir(tmpDir)
        # 选停用词
        if not rebuild and os.path.exists(os.path.join(tmpDir, 'corpus.mm')):
            self.bows = gensim.corpora.MmCorpus(os.path.join(tmpDir, 'corpus.mm'))
            if self.use_tfidf:
                self.tfidf = gensim.corpora.MmCorpus(os.path.join(tmpDir, 'tfidf.mm'))
            self.dictionary = Dictionary.load_from_text(os.path.join(tmpDir, 'dict.txt'))
            self.docs = pickle.load(open(os.path.join(tmpDir, 'docs.pkl'), 'rb'))
            self.dictionary.id2token = {v: k for k, v in
                                        self.dictionary.token2id.items()}  # because id2token is empty be default, it is a bug.
        # 这里
        else:
            if stopwords is None:
                stopwords = set([l.strip('\n').strip() for l in
                                 open(os.path.join(cwd, 'data', 'stopwords.txt'), 'r', encoding='utf-8')])
            # self.txtLines is the list of string, without any preprocessing.
            # self.texts is the list of list of tokens.
            print('Tokenizing ...')
            if tokenizer is None:
                tokenizer = globals()[LANG_CLS[lang]](stopwords=stopwords)  # 根据lang选择调用tokenization中的哪个类
            self.docs = tokenizer.tokenize(self.txtLines)
            self.docs = [line for line in self.docs if line != []]
            # build dictionary
            self.dictionary = Dictionary(self.docs)
            # self.dictionary.filter_n_most_frequent(remove_n=20) 筛选
            # use Dictionary to remove un-relevant tokens 筛选
            # self.dictionary.filter_extremes(no_below=no_below, no_above=no_above,keep_n=None)
            # 重新分配id
            self.dictionary.compactify()

            self.dictionary.id2token = {v: k for k, v in
                                        self.dictionary.token2id.items()}  # because id2token is empty by default, it is a bug.
            # convert to BOW representation
            self.bows, _docs = [], []
            for doc in self.docs:
                _bow = self.dictionary.doc2bow(doc)  # doc2bow 函数作用
                if _bow:
                    _docs.append(list(doc))
                    self.bows.append(_bow)
            self.docs = _docs
            if self.use_tfidf:
                self.tfidf_model = TfidfModel(self.bows)
                self.tfidf = [self.tfidf_model[bow] for bow in self.bows]
            # serialize the dictionary 序列化字典
            gensim.corpora.MmCorpus.serialize(os.path.join(tmpDir, 'corpus.mm'), self.bows)
            self.dictionary.save_as_text(os.path.join(tmpDir, 'dict.txt'))
            pickle.dump(self.docs, open(os.path.join(tmpDir, 'docs.pkl'), 'wb'))
            if self.use_tfidf:
                gensim.corpora.MmCorpus.serialize(os.path.join(tmpDir, 'tfidf.mm'), self.tfidf)
        self.vocabsize = len(self.dictionary)
        self.numDocs = len(self.bows)
        print(f'Processed {len(self.bows)} documents.')
        # 求词频矩阵
        # 所有字母转换位小写
        docs = ["".join(d).lower() for d in self.docs]
        # 获取所有词(bows)
        words = list(set(chain(*self.docs)))
        # print(words)
        # 词到ID的映射, 使得每个词有一个ID
        dictionary = dict(zip(words, range(len(words))))
        # 创建一个空的矩阵, 行数等于词数, 列数等于文档数
        self.matrix = np.zeros((len(docs), self.vocabsize))
        # 逐个文档统计词频
        for col, d in enumerate(self.docs):  # col 表示矩阵第几列，d表示第几个文档。
            # 统计词频
            count = Counter(d)  # 其实是个词典，词典元素为：{单词：次数}。
            for word in count:
                # 用word的id表示word在矩阵中的行数，该文档表示列数。
                word_id = dictionary[word]
                # 把词频赋值给矩阵
                self.matrix[col, word_id] = count[word]

    def __getitem__(self, idx):
        bow = torch.zeros(self.vocabsize)
        if self.use_tfidf:
            item = list(zip(*self.tfidf[idx]))
        else:
            item = list(zip(*self.bows[idx]))  # bow = [[token_id1,token_id2,...],[freq1,freq2,...]]
        bow[list(item[0])] = torch.tensor(list(item[1])).float()
        txt = self.docs[idx]
        return txt, bow

    def __len__(self):
        return self.numDocs

    def collate_fn(self, batch_data):
        texts, bows = list(zip(*batch_data))
        return texts, torch.stack(bows, dim=0)

    def __iter__(self):
        for doc in self.docs:
            yield doc

    def show_dfs_topk(self, topk=20):
        ndoc = len(self.docs)
        dfs_topk = sorted([(self.dictionary.id2token[k], fq) for k, fq in self.dictionary.dfs.items()],
                          key=lambda x: x[1], reverse=True)[:topk]
        for i, (word, freq) in enumerate(dfs_topk):
            print(f'{i + 1}:{word} --> {freq}/{ndoc} = {(1.0 * freq / ndoc):>.13f}')
        return dfs_topk

    def show_cfs_topk(self, topk=20):
        ntokens = sum([v for k, v in self.dictionary.cfs.items()])
        cfs_topk = sorted([(self.dictionary.id2token[k], fq) for k, fq in self.dictionary.cfs.items()],
                          key=lambda x: x[1], reverse=True)[:topk]
        for i, (word, freq) in enumerate(cfs_topk):
            print(f'{i + 1}:{word} --> {freq}/{ntokens} = {(1.0 * freq / ntokens):>.13f}')

    #
    def topk_dfs(self, topk=20):
        ndoc = len(self.docs)
        dfs_topk = self.show_dfs_topk(topk=topk)
        return 1.0 * dfs_topk[-1][-1] / ndoc


if __name__ == '__main__':
    docSet = DocDataset('mydatatest', rebuild=True)
    dataloader = DataLoader(docSet, batch_size=64, shuffle=True, num_workers=16, collate_fn=docSet.collate_fn,
                            pin_memory=True)
    print('docSet.docs[10]:', docSet.docs[10])
    print(next(iter(dataloader)))
    print('The top 20 tokens in document frequency:')
    docSet.show_dfs_topk()
    print('The top 20 tokens in collections frequency:')
    input("Press any key ...")
    docSet.show_cfs_topk()
    input("Press any key ...")
    for doc in docSet:
        print(doc)
        break
    print(docSet.topk_dfs(20))
