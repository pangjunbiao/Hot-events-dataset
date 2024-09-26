from __future__ import division
import numpy as np
import itertools
from collections import Counter
import numpy as np
from itertools import chain
from Dataset import DocDataset
import torch
from gensim.models import Word2Vec
import math
import scipy.sparse as sp
import lda
import lda.datasets
import torch
import torch.nn.functional as F
import numpy as np

def inverse_gumbel_cdf(y, mu, beta):
    """
    获取提前采样好的标准Gumbel分布序列
    分布是CDF分布
    """
    return mu - beta * np.log(-np.log(y))

def gumbel_softmax_sampling(h, mu=0, beta=1, tau=0.1):
    """
    h : (N x K) tensor. Assume we need to sample a NxK tensor, each row is an independent r.v.
    """
    shape_h = h.shape
    p = F.softmax(h, dim=1)
    y = torch.rand(shape_h) + 1e-25  # ensure all y is positive.
    g = inverse_gumbel_cdf(y, mu, beta)
    x = torch.log(p) + g  # samples follow Gumbel distribution.
    # using softmax to generate one_hot vector:
    # 生成one_hot是因为原来的arg max不可导，生成one_hot后可以用softmax函数近似
    x = x/tau
    x = F.softmax(x, dim=1)  # now, the x approximates a one_hot vector.
    return x

def Word_frequency(docs, words):
    dictionary = dict(zip(words, range(len(words))))
    matrix = np.zeros((len(docs), len(words)))

    for d, col in enumerate(docs):  # col 表示矩阵第几列，d表示第几个文档。
        # 统计词频
        count = Counter(col)  # 其实是个词典，词典元素为：{单词：次数}。
        for word in count:
            # 用word的id表示word在矩阵中的行数，该文档表示列数。
            id = dictionary[word]
            # 把词频赋值给矩阵
            matrix[d, id] = count[word]
    return matrix

def event(bows, index):
    event_lst = []
    for i in index:
        event_lst.append(bows[i])
    return event_lst


class PD:
    def __init__(self, words, test_num):
        self.words = words
        self.documentNum = len(words)
        self.test_num = test_num

    def getHeat(self, path):
        heat = []

        f = open(path, "a+", encoding='utf-8')
        f.seek(0)
        lines = f.readlines()
        leng = len(lines)
        for i in range(leng):
            heat.append(int(lines[i]))
        return heat

    def remove_duplicate(self):
        """
        去重
        @return:
        """
        word_lst = []
        for i in range(self.documentNum):
            data_key = dict(Counter([k for k in self.words[i]]))
            data_keys_new = [k for k in data_key.keys()]
            word_lst.append(data_keys_new)
        return word_lst

    def getbows(self, words):
        """
        去重，构建words_matrix矩阵，维度为words*words
        @return:
        """
        temp_lst = []
        # 将所有词存放在一个list中
        for i in range(self.documentNum - self.test_num):
            # if words[i]
            for j in range(len(words[i])):
                temp_lst.append(words[i][j])
        # 对给list去重，用于构建words*words的矩阵
        data_key = dict(Counter([k for k in temp_lst]))
        data_keys_new = [k for k in data_key.keys()]
        return data_keys_new

    def getWord_frequency(self, docs):
        # 获取所有词
        words = list(set(chain(*docs)))
        dictionary = dict(zip(words, range(len(words))))
        matrix = np.zeros((len(docs), len(words)))

        for d, col in enumerate(docs):  # col 表示矩阵第几列，d表示第几个文档。
            # 统计词频
            count = Counter(col)  # 其实是个词典，词典元素为：{单词：次数}。
            for word in count:
                # 用word的id表示word在矩阵中的行数，该文档表示列数。
                id = dictionary[word]
                # 把词频赋值给矩阵
                matrix[d, id] = count[word]
        return matrix

    def norm_doc_length(self, docs, num_words):
        temp_lst = []
        for i in docs:
            temp_lst.append(len(i))
        Max = max(temp_lst)
        # Min = min(temp_lst)
        Min = num_words
        for i in range(len(docs)):
            docs[i] = docs[i][:Min]
        return docs


# def getbows_vec(words_indix, bows_vec):  # 求词频矩阵中词索引对应的当前事件词向量
#     cor_bows_vec = []
#     for i in range(len(words_indix)):
#         for j in range(words_indix[0].size):
#             cor_bows_vec.append(bows_vec[words_indix[i][0][j]])
#
#     allnews_bows_vec = np.empty((len(words_indix), words_indix[0].size, len(bows_vec[0])), dtype=float)
#     c = 0
#     for i in range(0, len(words_indix), 4):
#         for j in range(words_indix[i].size):
#             for k in range(len(bows_vec[0])):
#                 allnews_bows_vec[i][j][k] = cor_bows_vec[c][k]
#             c = c + 1
#     return allnews_bows_vec
def get_index(index, device):  # 求每个事件在邻接矩阵中的索引并固定长度
    temp_lst = []
    for i in index:
        temp_lst.append(len(i))
    Max = max(temp_lst)
    t_index = torch.zeros(Max)
    for i in range(len(index)):
        padding = torch.zeros(Max).to(device)
        for j in range(len(index[i])):
            padding[j] = index[i][j]
        t_index = torch.vstack((t_index, padding))
    t_index = t_index[1:]
    return t_index

def getbows_vec(words_indix, bows_vec):  # 求词频矩阵中词索引对应的当前事件词向量
    cor_bows_vec = []
    for i in range(len(words_indix)):
        for j in range(words_indix[i].shape[1]):
            cor_bows_vec.append(bows_vec[words_indix[i][0][j]])

    allnews_bows_vec = np.zeros((len(words_indix), 20, len(bows_vec[0])), dtype=float)  # (1800, 12, 100)先将数组设置为最大长度12

    for i in range(0, len(words_indix)):
        c = 0
        for j in range(words_indix[i].shape[1]):
            for k in range(len(bows_vec[0])):
                allnews_bows_vec[i][j][k] = cor_bows_vec[i + c][k]

        c += words_indix[i].shape[1]  # 跳过一个事件中的所有词数
    return allnews_bows_vec

def adj_norm(adjacency):
    """计算 L=D^-0.5 * (A+I) * D^-0.5"""
    adjacency = sp.coo_matrix(adjacency)    # 稀疏矩阵
    degree = np.array(adjacency.sum(1))
    d_hat = sp.diags(np.power(degree, -0.5).flatten())
    return d_hat.dot(adjacency).dot(d_hat).tocoo()

def AdjacencyMatrix(words_frequency):  # PMI的方式建立边
    adjacency = np.eye(words_frequency.shape[1], dtype=float)
    # index = []
    dij = []  # 存放ij词对出现频次最高的词对的索引，用于观测这些词对是否有意义
    sum_ij = 0  # 所有ij词对的频次之和
    sum_node = 0  # 所有单个节点出现的频次之和
    all_index = []
    all_fre = []
    for i in range(len(words_frequency)):
        # cor_index = torch.zeros(words_frequency.shape[1])
        # D = len(np.where(words_frequency[i] != 0)[0])
        index = []
        fre = []

        for j in range(len(words_frequency[i])):
            if words_frequency[i][j] != 0:
                index.append(j)
                fre.append(words_frequency[i][j])
        sum_node = sum_node + sum(fre)
        cor_index = list(itertools.combinations(index, 2))  # 找出单词对应的索引并进行排列组合
        cor_fre = list(itertools.combinations(fre, 2))  # 找出单词出现的频率并进行排列组合
        all_index.append(cor_index)
        all_fre.append(cor_fre)
    for i in range(len(all_index)):
        sum_ij = sum_ij + len(all_index[i])
    for i in range(len(all_index)):
        for k in range(len(all_index[i])):
            d_ij = 0
            # 下面的for循环是为了找出d_ij在所有事件中的次数
            for m in range(len(all_index)):
                 for n in range(len(all_index[m])):
                    if all_index[i][k] == all_index[m][n] or tuple(reversed(all_index[0][0])) == all_index[m][n]:
                        d_ij = d_ij + min(all_fre[i][k])
            if d_ij >= 50:  # 记录d_ij的一些最大值，用于观测最大值是否有真实含义
                dij.append(all_index[i][k])
            d_i = all_fre[i][k][0]  # 当前事件中i出现的频次
            d_j = all_fre[i][k][1]
            # Di = sum(words_frequency[:, all_index[i][k][0]])  # 词i出现的频次之和
            # Dj = sum(words_frequency[:, all_index[i][k][1]])  # 词j出现的频次之和
            weight = math.log((d_ij/sum_ij) / ((d_i/sum_node) * (d_j/sum_node)))
            if weight > 0:
                adjacency[all_index[i][k][0]][all_index[i][k][1]] = math.sqrt(weight)
                adjacency[all_index[i][k][1]][all_index[i][k][0]] = math.sqrt(weight)
            if weight <= 0:
                adjacency[all_index[i][k][0]][all_index[i][k][1]] = 0
                adjacency[all_index[i][k][1]][all_index[i][k][0]] = 0
        # index.append(torch.tensor(indix))
    return adjacency, dij


def getword_frequencyIndix(word_frequency, num_words):
    all_indix = np.zeros([len(word_frequency), num_words])
    for i in range(len(word_frequency)):
        index = np.array(np.where(word_frequency[i] != 0))
        for j in range(index.shape[1]):
            all_indix[i][j] = index[0][j]
    return all_indix


def Heat_normalization(lambda_org):
    Max = max(lambda_org)
    Min = min(lambda_org)
    lambda_label = []
    for i in range(len(lambda_org)):
        lambda_temp = (lambda_org[i] - Min) / (Max - Min)
        lambda_label.append(format(lambda_temp, '.4f'))  # 取小数点后四位

    lambda_label = np.array(lambda_label)
    return lambda_label

def ZeroAvg_Normalize(data):
    text = (data - data.mean())/data.std()
    return text


def normalization(org):
    Max = max(org)
    Min = min(org)
    last = []
    for i in range(len(org)):
        temp = (org[i] - Min) / (Max - Min)
        last.append(float(format(temp[0], '.4f')))  # 取小数点后四位

    last = np.array(last)
    return last


def docs_intensity(words_indix, all_bows_Transvec):
    docs_intensity_vec = torch.empty((1, 66))
    for i in range(len(words_indix)):
        iter = list(itertools.combinations(words_indix[i].tolist()[0], 2))  # 求列表两两之间的所有不重复的排列组合
        words_intensity_vec = all_bows_Transvec[np.asarray(iter[0])[0]][np.asarray(iter[0])[1]].data
        for indix in range(len(iter) - 1):
            intensity_vec = all_bows_Transvec[np.asarray(iter[indix + 1])[0]][np.asarray(iter[indix + 1])[1]].data
            words_intensity_vec = torch.hstack([words_intensity_vec, intensity_vec])  # 一个事件中的所有激发值
        docs_intensity_vec = torch.vstack((docs_intensity_vec, words_intensity_vec))  # 所有事件中的所有激发值
    return docs_intensity_vec[1:]


def sum_docs_intensity(docs_intensity_vec):
    lambda_intensity = torch.sum(docs_intensity_vec[0])
    for i in range(len(docs_intensity_vec) - 1):
        sum_value = torch.sum(docs_intensity_vec[i + 1])
        lambda_intensity = torch.hstack([lambda_intensity, sum_value])
    return lambda_intensity


def getData(train_num, path, Heatpath, img_features, num_words):
    # 获取文本的词向量
    dataset = DocDataset(taskname=path, no_below=3, no_above=0.0134, rebuild=False, use_tfidf=False)
    data = dataset.docs
    data_train = data[0:train_num]
    data_test = data[train_num:]

    PD_model = PD(data, test_num=0)
    PD_model_train = PD(data_train, test_num=0)  # test_num：文档数-test_num=测试文档数
    PD_model_test = PD(data_test, test_num=0)

    # 获取热度值
    lambda_org = PD_model.getHeat(Heatpath)

    # 真实热度值处理
    lambda_label = Heat_normalization(lambda_org)
    # lambda_label = ZeroAvg_Normalize(lambda_org)
    lambda_label = lambda_label.astype(float)  # 将真值中的字符串转化为数字

    lambda_label_train = lambda_label[0:train_num]
    lambda_label_test = lambda_label[train_num:]

    docs_train = PD_model_train.norm_doc_length(data_train, num_words)
    docs_test = PD_model_test.norm_doc_length(data_test, num_words)

    # 获取文本的词袋
    bows_train = PD_model_train.getbows(docs_train)
    bows_test = PD_model_test.getbows(docs_test)

    # 基于词共现构建词频矩阵
    # word_frequency_train = PD_model_train.getWord_frequency(docs_train).astype(int)
    # word_frequency_test = PD_model_test.getWord_frequency(docs_test).astype(int)
    # 词频矩阵中词的顺序对应词袋中词的顺序
    word_frequency_train = Word_frequency(docs_train, bows_train).astype(int)
    word_frequency_test = Word_frequency(docs_test, bows_test).astype(int)


    # 求词频矩阵中每个文档中词的索引
    words_indix_train = getword_frequencyIndix(word_frequency_train, num_words)
    words_indix_test = getword_frequencyIndix(word_frequency_test, num_words)

    image_features_train = img_features[0:train_num]
    image_features_test = img_features[train_num:]

    # 获取文本的词向量
    word2vec_model = Word2Vec.load("word2vec_model.model")

    bows_vec_train = word2vec_model.wv[bows_train]
    bows_vec_test = word2vec_model.wv[bows_test]

    return torch.tensor(bows_vec_train), lambda_label_train, image_features_train, words_indix_train, word_frequency_train, bows_train,\
           torch.tensor(bows_vec_test), lambda_label_test, image_features_test, words_indix_test, word_frequency_test, bows_test


def txt_img(files):
    files_order = [0 for i in range(len(files))]
    for filename in files:
        index = int(filename.split('-', 2)[0])
        files_order[index] = filename
    return files_order


def percentile(scores, sparsity):
    """"
    numel()：计算tensor中一共包含多少个元素
    kthvalue(k)：
    """
    k = 1 + round(.01 * float(sparsity) * (scores.numel() - 1))
    return scores.view(-1).kthvalue(k).values.item()

class GetSubnetFaster(torch.autograd.Function):
    @staticmethod
    def forward(ctx, scores, zeros, ones, sparsity):
        ctx.save_for_backward(scores)  # 保存带有正负数的scores以用于反向传播时判断梯度的正负

        k_val = percentile(scores, sparsity * 100)

        return torch.where(scores.abs() < k_val, zeros.to(scores.device), ones.to(scores.device))

    @staticmethod
    def backward(ctx, g):
        scores, = ctx.saved_tensors
        return torch.where(scores < 0, -g, g), None, None, None


def L2Loss(x, y):
    """
      x，预测数据
      y，真实数据

    """
    assert len(x) == len(y)
    loss = np.sqrt(np.sum(np.square(x - y)) / len(x))
    return torch.tensor(loss)


def MAELoss(x, y):
    """
      x，预测数据
      y，真实数据

    """
    assert len(x) == len(y)
    m = len(x)
    loss = 1 / m * np.sum(np.abs(y - x))
    return torch.tensor(loss)


def R_Accuracy(output, label, delta):
    assert len(output) == len(label)
    R_acc = []
    for i in range(len(output)):
        acc = abs(math.pow(abs(((label[i] - output[i]) / label[i]) <= delta), 2))
        # acc = abs(abs((label[i] - output[i]) / label[i]) <= delta)
        R_acc.append(acc)
    R_acc = np.array(R_acc)
    R_acc = sum(R_acc) / len(output)
    return R_acc


def LMSE(output, label):
    assert len(output) == len(label)
    mse_acc = []
    for i in range(len(output)):
        acc = math.pow((math.log(label[i]) - math.log(output[i])), 2)
        mse_acc.append(acc)
    mse_acc = np.array(mse_acc)
    mse_acc = sum(mse_acc) / len(output)
    return mse_acc

def order_loss(org_lst, pred_lst, K):
    """
    预测出来的排序和真实排序之间的差值（比如真实为第一，预测出来为第三，差值就为2）
    @param org_lst: 真实的排序
    @ pred_lst:  预测出来的排序
    @return: loss值
    @param k: 排序的数量
    """
    loss_lst = []
    lst_len = len(org_lst)
    for i in range(lst_len):
        order_loss_temp = np.abs(org_lst[i] - pred_lst[i])
        loss_lst.append(order_loss_temp)
    order_loss = sum(loss_lst) / K
    return loss_lst, order_loss