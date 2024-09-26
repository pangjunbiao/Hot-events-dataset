#!/user/bin/env python
# coding=utf-8
import numpy as np
import torch.nn as nn
# from models import SelfAttention
from tools import *
from util import get_args
# from Transformer_Encoder_Layer import Encoder
from torch import sigmoid
from models import GcnNet, MLP, GAT
import torch
import clip
from PIL import Image
import torch.nn.functional as F
import os
import heapq
import scipy.sparse as sp
import matplotlib.pyplot as plt
from itertools import combinations, permutations
import seaborn as sns
from sklearn.metrics import roc_auc_score, ndcg_score
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


def normal_distribution(x, mean, sigma):
    return np.exp(-1 * ((x - mean) ** 2) / (2 * (sigma ** 2))) / (math.sqrt(2 * np.pi) * sigma)


def auc_calculate(label, preds):
    y_argsort = label.argsort()
    pred_argsort = preds.argsort()
    y_lat = np.zeros(shape=(len(label), len(label)))
    pred_lat = np.zeros(shape=(len(label), len(label)))
    for i in range(len(label)):
        y_lat[i][y_argsort[i]] = 1
        # x_len = max(y_argsort[i], 200 - y_argsort[i])
        mean = pred_argsort[i]
        sigma = 8
        x = np.arange(0, 200)
        y = normal_distribution(x, mean, sigma)
        pred_lat[i] = y
    forest_auc = roc_auc_score(y_lat, pred_lat, multi_class='ovo')
    return forest_auc


class NTS_model(nn.Module):
    def __init__(self, config):
        super(NTS_model, self).__init__()
        self.fc_docs = nn.Linear(config['topic_num'], 1, bias=False)
        self.GCN_model = GcnNet()
        self.GAT_model = GAT(nfeat=100, nhid=args.hidden, ofeat=100, dropout=0.6, nheads=args.nb_heads,
                             alpha=args.alpha)
        self.fc_image = nn.Linear(512, 1, bias=False)
        self.mlp = MLP(512, 128, 1)
        self.W_mu1 = nn.Linear(100, 100)
        self.W_mu2 = nn.Linear(100, 1)

        self.W_eta1 = nn.Linear(100, 100)
        self.W_eta2 = nn.Linear(100, 1)

        self.W_gamma1 = nn.Linear(100, 100)
        self.W_gamma2 = nn.Linear(100, 1)

        self.W_beta1 = nn.Linear(100, 100)
        self.W_beta2 = nn.Linear(100, 1)

        self.W_h1 = nn.Linear(100, 100)
        self.W_h2 = nn.Linear(100, 100)
        self.relu = nn.ReLU()

        # 在beta参数上稀疏
        # self.sparsity = 0.3
        self.W_beta = nn.Parameter(torch.empty(100, 1))
        self.w_m = nn.Parameter(torch.empty(100, 1))
        self.zeros_weight, self.ones_weight = torch.zeros(self.w_m.shape), torch.ones(self.w_m.shape)
        self.weight_mask = None

        # 在mu参数上稀疏
        self.W_mu = nn.Parameter(torch.empty(100, 1))
        self.w_mu = nn.Parameter(torch.empty(100, 1))
        self.zeros_weight_mu, self.ones_weight_mu = torch.zeros(self.w_m.shape), torch.ones(self.w_m.shape)
        self.weight_mask_mu = None

    def forward(self, epoch, epochs, train_adj, masks, bows_vec, image_features):
        H, H_p = self.GCN_model(train_adj.float(), bows_vec.float(), masks)
        # H, H_p = self.GAT_model(bows_vec.float(), train_adj.float())

        # H = gumbel_softmax_sampling(H, 0, 1, 0.1)

        H_eli = F.relu(self.W_h2(F.relu(self.W_h1(H))))  # 将H变化到自激空间
        # H_eli = F.relu(self.W_h1(H))

        # 归一化
        # p=2表示二范式, dim=1表示按行归一化
        H_eli_norm = F.normalize(H_eli, p=2, dim=1)
        H_p_norm = F.normalize(H_p, p=2, dim=1)
        # mu = F.gelu(self.W_mu2(F.gelu(self.W_mu1(H_p))))
        mu = F.relu(self.W_mu2(H_p))
        H_p = F.dropout(H_p, p=0.3, training=self.training)
        # eta = F.gelu(self.W_eta2(F.relu(self.W_eta1(H_p))))
        eta = F.relu(self.W_eta2(H_p))
        # H_p = F.dropout(H_p, p=0.3, training=self.training)
        # gamma = F.softplus(self.W_gamma2(F.softplus(self.W_gamma1(H_p))))
        gamma = F.relu(self.W_gamma2(H_p))

        # beta_ = F.relu(self.W_beta3(F.relu(self.W_beta2(F.relu(self.W_beta1(H))))))  # 每个词的beta值
        # beta_ = F.relu(self.W_beta2(H))  # 每个词的beta值

        self.sparsity = (epoch / epochs) * 0.3  # sparsity随着迭代次数增加，降低STE震荡
        # self.sparsity_mu = (epoch/epochs) * 0.3

        # 利用STE的方法稀疏beta的参数
        # 对self.w_m初始化
        nn.init.kaiming_uniform_(self.w_m, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.W_beta, a=math.sqrt(5))

        self.weight_mask = GetSubnetFaster.apply(self.w_m,
                                                 self.zeros_weight,
                                                 self.ones_weight,
                                                 self.sparsity)
        w_pruned = self.weight_mask * self.W_beta
        beta_ = F.normalize(H.matmul(w_pruned).squeeze(), dim=0)

        beta = torch.mm(masks, beta_.unsqueeze(1))  # 每个事件的beta值

        Z_ = F.relu(torch.mm(H_eli_norm, H_eli_norm.t()))

        # 求Z的另一个思路
        Z = torch.zeros(1)
        Z_event = []  # 画图用
        for i in range(len(masks)):
            index = torch.tensor(np.where(masks[i] != 0))[0]
            _, indices = torch.sort(index)
            cor_index = list(combinations(index, 2))
            cor_indices = list(combinations(indices, 2))
            z = torch.zeros(1)
            Z_event_temp = torch.zeros(args.num_words, args.num_words)

            for j in range(len(cor_index)):
                # if(j != i):
                hi = H_eli_norm[cor_index[j][0].item()]
                hj = H_eli_norm[cor_index[j][1].item()]
                if (cor_index[j][0].item() != cor_index[j][1].item()):
                    z_temp = Z_[cor_indices[j][0].item()][cor_indices[j][1].item()]
                    z_ins_temp = torch.exp(-torch.square(2 - 2 * z_temp))  # 求两个词之间的激发分数
                    # z_ins_temp = torch.exp(-z_temp)  # 求两个词之间的激发分数
                    z = z + torch.mm(hj.unsqueeze(1).t(), hi.unsqueeze(1))  # 一个事件内的所有词的激发项加起来作为一个事件的激发值
                    Z_event_temp[cor_indices[j][0].item()][cor_indices[j][1].item()] = z_ins_temp
                    Z_event_temp[cor_indices[j][1].item()][cor_indices[j][0].item()] = z_ins_temp
            Z_event.append(Z_event_temp)
            # Z = torch.vstack((Z, torch.square(2-2*z)))

            Z = torch.vstack((Z, z))
        Z = F.relu(Z[1:])

        lambda_texts = sigmoid(mu + beta + torch.mul(eta, torch.exp(-torch.mul(gamma, Z))))
        # lambda_texts = mu + beta
        # lambda_texts = mu + torch.mul(eta, torch.exp(-torch.mul(gamma, Z)))

        # lambda_texts = mu

        # lambda_imgs = sigmoid(self.mlp(image_features.to(device)))  # 获取图片热度值。
        lambda_imgs = self.mlp(image_features.to(device))  # 获取图片热度值。

        lambda_total = sigmoid(lambda_texts.to(device) + lambda_imgs.to(device))
        # 消融实验
        # lambda_total = sigmoid(lambda_texts.to(device))
        # lambda_total = sigmoid(lambda_imgs.to(device))

        return lambda_total, Z_, beta_, gamma, eta, Z_event, H


def auc_calculat(label, preds):
    y_argsort = label.argsort()
    pred_argsort = preds.argsort()
    y_lat = np.zeros(shape=(len(label), len(label)))
    pred_lat = np.zeros(shape=(len(label), len(label)))
    for i in range(len(label)):
        y_lat[i][y_argsort[i]] = 1
        # x_len = max(y_argsort[i], 200 - y_argsort[i])
        mean = pred_argsort[i]
        sigma = 8
        x = np.arange(0, 200)
        y = normal_distribution(x, mean, sigma)
        pred_lat[i] = y
    forest_auc = roc_auc_score(y_lat, pred_lat, multi_class='ovo')
    return forest_auc


def get_ap(y_argsort, pred_argsort, k, m):
    ap_sum = 0
    for i in range(1, k):
        index = np.where(y_argsort == i)
        pred_sort = pred_argsort[index]
        if abs(pred_sort - i) > m:
            ap_sum = ap_sum + 0
        else:
            ap_sum = ap_sum + i / pred_sort
            # ap_sum = ap_sum + 1
    ap = ap_sum / k
    return ap


def get_NDCG(y_argsort, pred_argsort, k):
    # 预测的相关度定义
    # 选k个，rank(real)-rank(pred)
    # 相关度定义为  k-abs(rank(real)-rank(pred))
    DCG = 0
    IDCG = 0
    # 排序从1开始
    for i in range(1, k):
        index = np.where(y_argsort == i)
        pred_sort = pred_argsort[index]
        rel = k - abs(i - pred_sort)
        log2i = np.log2(i + 1)
        DCG = DCG + log2i
        IDCG = IDCG + rel / log2i
    if IDCG == 0:
        IDCG = 1
        print("k="+str(k)+"时IDCG为0")
    NDCG = DCG / IDCG
    return NDCG


if __name__ == '__main__':
    # parse the arguments
    args = get_args()
    torch.manual_seed(args.seed)
    criterion2 = nn.MSELoss().cuda()

    if os.access("./img_features.pt", os.F_OK):
        device = "cpu"
        image_features = torch.load('img_features.pt')

    else:
        # 加载CLIP的image encoder
        device = "cuda" if torch.cuda.is_available() else "cpu"
        clip_model, clip_preprocess = clip.load('ViT-B/32', device, jit=False)

        # 利用clip获取图片特征
        path = args.img_path
        files = os.listdir(path)
        files_order = txt_img(files)  # 图文按顺序对应对应
        image_features = torch.empty((1, 512)).to(device)
        for file in files_order:
            img_fn = os.path.join(path, file)

            image = clip_preprocess(Image.open(img_fn)).unsqueeze(0).to(device)
            with torch.no_grad():
                temp_image_features = clip_model.encode_image(image)
                image_features = torch.vstack((image_features, temp_image_features))
        image_features = image_features[1:]
        # 归一化
        image_features = image_features / image_features.norm(dim=1, keepdim=True)
        # 计算余弦相似度 logit_scale是尺度系数
        logit_scale = clip_model.logit_scale.exp()
        image_features = logit_scale * image_features.float()
        torch.save(image_features, 'img_features.pt')

    if args.model == 'NTS':

        # 获取训练数据和测试数据
        train_bows_vec, train_lambda_label, train_image_features, train_words_indix, train_word_frequency, train_bows, \
        test_bows_vec, test_lambda_label, test_image_features, test_words_indix, test_word_frequency, test_bows = getData(
            args.num_train_docs, args.task, args.Heatpath, image_features, args.num_words)
        model = NTS_model({
            'topic_num': args.topic_num,
            'num_atten_heads': args.num_atten_heads
        })
    else:

        # 加载时序数据和模型
        pass

    if args.cuda:
        model = model.cuda()

    # optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=5e-4)
    optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr)
    # optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, momentum=0.8, nesterov=True)
    # optimizer = torch.optim.Adagrad(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=5e-4)
    # optimizer = torch.optim.RMSprop(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, alpha=0.9)

    # 学习率下降算法
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 5)
    # scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.0001, max_lr=0.0005)  # 无法迭代收敛
    # scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)

    loss_lst = []

    # train_adj, _ = AdjacencyMatrix(train_word_frequency)
    # train_tensor_adj = torch.tensor(train_adj)
    # torch.save(train_tensor_adj, 'train_adj_nocoo.pt')

    # 构建训练邻接矩阵
    if os.access("./train_adj_nocoo.pt", os.F_OK):
        train_tensor_adj = torch.load('train_adj_nocoo.pt')

    else:
        # # 找出单个词出现频次最高的一些词观察是否有意义
        # temp = []
        # for i in range(train_word_frequency.shape[1]):
        #     temp.append(sum(train_word_frequency[:, i]))
        #     if sum(train_word_frequency[:, i]) >= 50:
        #         print(train_bows[i])
        #         if sum(train_word_frequency[:, i]) >= 108:
        #             print("最大频次的词", train_bows[i])
        # te = max(temp, key=temp.count)  # 统计出现最多的频次

        train_adj, dij = AdjacencyMatrix(train_word_frequency)  # PMI
        # torch.save(train_adj, 'train_adj_org.pt')
        # 画出train_adj中边的权重值
        # plt.hist(train_adj.flatten()[np.where(train_adj.flatten() != 0)][np.isfinite(train_adj.flatten()[np.where(train_adj.flatten() != 0)])], bins=100)
        # plt.show()

        train_norm_adj = adj_norm(train_adj)
        # train_edge_index = sp.coo_matrix(train_normalize_adj)
        train_num_nodes, train_input_dim = train_bows_vec.shape
        train_indices = torch.from_numpy(np.asarray([train_norm_adj.row, train_norm_adj.col]).astype('int32')).long()
        train_values = torch.from_numpy(train_norm_adj.data.astype(np.float32))
        train_tensor_adj = torch.sparse.FloatTensor(train_indices, train_values, (train_num_nodes, train_num_nodes)).to(
            device)
        torch.save(train_tensor_adj, 'train_adj.pt')

    if os.access("./train_masks.pt", os.F_OK):
        train_masks = torch.load('train_masks.pt')
    else:
        train_masks = torch.zeros(train_word_frequency.shape[1])
        for i in range(len(train_words_indix)):
            mask = torch.zeros(train_word_frequency.shape[1])
            for j in train_words_indix[i]:
                if j != 0:  # 因为一些重复的词去除的位置值为0，
                    mask[int(j)] = 1
            train_masks = torch.vstack((train_masks, mask))
        train_masks = train_masks[1:]
        train_masks[0][0] = 1  # 但第一个词的真正索引就是1
        torch.save(train_masks, 'train_masks.pt')

    # 构建测试邻接矩阵
    # test_adj, _ = AdjacencyMatrix(test_word_frequency)
    # test_tensor_adj = torch.tensor(test_adj)
    # torch.save(test_tensor_adj, 'test_adj_nocoo.pt')

    if os.access("./test_adj_nocoo.pt", os.F_OK):
        test_tensor_adj = torch.load('test_adj_nocoo.pt')
    else:
        test_adj, _ = AdjacencyMatrix(test_word_frequency)  # PMI
        test_norm_adj = adj_norm(test_adj)
        # train_edge_index = sp.coo_matrix(train_normalize_adj)
        test_num_nodes, test_input_dim = test_bows_vec.shape
        test_indices = torch.from_numpy(np.asarray([test_norm_adj.row, test_norm_adj.col]).astype('int32')).long()
        test_values = torch.from_numpy(test_norm_adj.data.astype(np.float32))
        test_tensor_adj = torch.sparse.FloatTensor(test_indices, test_values, (test_num_nodes, test_num_nodes)).to(
            device)
        torch.save(test_tensor_adj, 'test_adj.pt')
    if os.access("./test_masks.pt", os.F_OK):
        test_masks = torch.load('test_masks.pt')
    else:
        test_masks = torch.zeros(test_word_frequency.shape[1])
        for i in range(len(test_words_indix)):
            mask_ = torch.zeros(test_word_frequency.shape[1])
            for j in test_words_indix[i]:
                if j != 0:
                    mask_[int(j)] = 1
            test_masks = torch.vstack((test_masks, mask_))
        test_masks = test_masks[1:]
        torch.save(test_masks, 'test_masks.pt')

    for epoch in range(args.epochs):
        print("training...")

        train_lambda_total, Z_, beta_, gamma, eta, Z_event, H = model(epoch, args.epochs,
                                                                      torch.tensor(train_tensor_adj), train_masks,
                                                                      train_bows_vec, train_image_features)
        train_target = torch.from_numpy(train_lambda_label).reshape(-1, 1)
        # 损失函数
        # mse_loss = nn.MSELoss()
        huber_loss = nn.SmoothL1Loss()
        loss = huber_loss(train_lambda_total.float().to(device), train_target.float().to(device))
        alpha1 = 0.001
        alpha2 = 0.001
        loss = loss + alpha1 / len(Z_) * (torch.abs(Z_).sum()) + alpha2 / len(beta_) * (torch.abs(beta_).sum())
        # loss = loss + alpha2/len(beta_)*(torch.abs(beta_).sum())

        loss_lst.append(loss)
        optimizer.zero_grad()
        loss.backward()

        optimizer.step()
        # scheduler.step()

        print('| epoch {:3d} | loss {:5f} '.format(epoch, loss))

        #####################################################

        print("testing...")
        model.eval()  # turn on the eval() switch to disable dropout

        # test_lambda_total, _, _, _, _, _, _ = model(torch.tensor(test_tensor_adj), test_masks, test_bows_vec, test_image_features)
        test_lambda_total, test_Z, test_beta, _, _, _, _ = model(epoch, args.epochs, torch.tensor(test_tensor_adj),
                                                                 test_masks, test_bows_vec, test_image_features)

        test_lambda_total_np = test_lambda_total.to(device).detach().numpy()
        test_lambda_total_np = np.around(test_lambda_total_np, decimals=4)
        target_test = test_lambda_label

        K = len(target_test)

        top10_index = heapq.nlargest(10, range(len(target_test)), target_test.__getitem__)
        top20_index = heapq.nlargest(20, range(len(target_test)), target_test.__getitem__)
        top30_index = heapq.nlargest(30, range(len(target_test)), target_test.__getitem__)

        loss_10lst, top10_loss = order_loss(np.arange(10),
                                            (np.ones(K) * (K - 1) - np.argsort(test_lambda_total_np.reshape(K)))[
                                                top10_index], K)  #
        loss_20lst, top20_loss = order_loss(np.arange(20),
                                            ((np.ones(K) * (K - 1)) - np.argsort(test_lambda_total_np.reshape(K)))[
                                                top20_index], K)  #
        loss_30lst, top30_loss = order_loss(np.arange(30),
                                            (np.ones(K) * (K - 1) - np.argsort(test_lambda_total_np.reshape(K)))[
                                                top30_index], K)  #

        # mrse = MRSE(test_lambda_total.detach().numpy(), target_test)
        MSELoss = nn.MSELoss()

        L1Loss = nn.L1Loss()
        lmse = LMSE(test_lambda_total_np, target_test)
        mse = MSELoss(test_lambda_total.to(device), torch.tensor(target_test).view(-1, 1))
        l1 = L1Loss(test_lambda_total.to(device), torch.tensor(target_test).view(-1, 1))
        # R_acc = R_Accuracy(test_lambda_total.detach().numpy(), target_test, 0.0005)  # 最后一个参数：误差接受范围
        # 参数设置参考论文：Predicting the Popularity of Online Content with nowledge-enhanced Neural Networks
        lst_k = [5, 5, 5, 5, 6, 6, 6, 7, 7, 7, 7, 8, 8, 8, 8, 9, 9, 9, 9, 10, 10, 10, 15, 15, 15, 15, 15]
        lst_m = [1, 2, 3, 4, 3, 4, 5, 3, 4, 5, 6, 3, 4, 5, 6, 4, 5, 6, 7, 4, 5, 6, 7, 6, 7, 8, 9, 10]
        ap_lst = []
        NDCG_lst = []
        y_argsort = target_test[:20].argsort()
        pred_argsort = test_lambda_total_np.squeeze()[:20].argsort()
        y_argsort_t = y_argsort + 1
        pred_argsort_t = pred_argsort + 1
        for i in range(len(lst_k)):
            ap_lst.append(
                get_ap(y_argsort=y_argsort_t, pred_argsort=pred_argsort_t, k=lst_k[i],
                       m=lst_m[i]))

        map = sum(ap_lst) / len(ap_lst)

        lst_t = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
        for i in range(len(lst_t)):
            NDCG_lst.append(get_NDCG(y_argsort=y_argsort,
                                     pred_argsort=pred_argsort_t,
                                     k=lst_t[i]))

        # roc = auc_calculat(target_test, test_lambda_total_np.squeeze())
        # ndcg = ndcg_score(target_test, test_lambda_total_np)
        print('| mse_loss {:4f} | top10_loss {:4f} | top20_loss {:4f} | top30_loss {:4f}'.format(
            mse, top10_loss, top20_loss, top30_loss))
        print("ap结果：", ap_lst)
        print("NDCG结果：", NDCG_lst)

        if epoch == args.epochs - 1:
            #     #  画出训练loss
            #     w = np.linspace(0, 1, args.epochs)
            #     plt.plot(w, np.array(loss_lst), label="Loss下降曲线")
            #     plt.xlabel("epochs")
            #     plt.ylabel("Loss")
            #     plt.legend(loc="best")
            #     plt.show()
            #
            #
            #     plt.hist(gamma.detach().numpy(), bins=50)
            #     plt.xlabel("gamma", size=20)
            #     plt.savefig('E:\\JLL-work\\visual images\\gamma.png')
            #     plt.cla()
            #     plt.hist(eta.detach().numpy(), bins=50)
            #     plt.xlabel("coe", size=20)
            #     plt.savefig('E:\\JLL-work\\visual images\\coe.png')
            #     plt.cla()

            np.save("test_beta.npy", test_beta.detach().numpy())
            np.save("train_beta.npy", beta_.detach().numpy())
            # 单个事件中所有词的重要得分：是否稀疏
            # num = 4  # 可视化事件num中的词的重要得分
            #
            # beta_event = torch.mul(torch.tensor(beta_), train_masks[num].unsqueeze(1))
            #
            # index_event = np.where(train_masks[num].detach().numpy() != 0)[0]  # 事件num在词袋中的索引
            # beta = list(np.ravel(event(beta_event, index_event)))  # 取出事件num中对应词的重要得分
            #
            # word = event(train_bows, index_event)  # 取出事件num中对应的词
            #
            # for i in range(len(beta)):
            #     plt.bar(word[i], beta[i])
            # plt.savefig('E:\\JLL-work\\visual images\\beta_sparse.png')
            # plt.cla()

            # 绘制词激发的热度图：
            # 强制将激发矩阵对角线置为0
            np.save("Z_matrics.npy", Z_.detach().numpy().round(1))
            # Z_part = Z_[:1000, :1000].detach().numpy().round(1)
            # row, col = np.diag_indices_from(Z_part)
            # Z_part[row, col] = 0
            #
            # with sns.axes_style("white"):
            #     sns.heatmap(Z_part, vmax=1, square=True, cmap="Blues",
            #                 xticklabels=False, yticklabels=False)
            # """
            # vmax, vmin,:图例中最大值和最小值的显示值，没有该参数时默认不显示
            # square：bool类型参数，是否使热力图的每个单元格为正方形，默认为False
            # """
            #
            # plt.title("heatmap", size=15)
            #
            # plt.savefig('E:\\JLL-work\\visual images\\Z.png')
            # plt.cla()
            #
            # # torch.save(beta_, 'beta_train.pt')
            # # beta_last = torch.load('beta_train.pt')
            # print(beta_)
