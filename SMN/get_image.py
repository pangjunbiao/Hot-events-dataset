# !/usr/bin/env python3
# coding: utf-8
import xlrd
import re
import requests
from urllib import error
import os
import time

class get_img:
    def __init__(self):
        self.temp_lst = []
        # 用于根据txt文件得到title
        self.txt_path = 'E:/JLL-work/code/data/data_title2500.txt'
        # 下载图片存放地址
        self.img_data_path = 'E:\\JLL-work\\code\\data\\image_data\\'
        # 根据txt文件中的内容得到img  True时可以正常运行
        self.txt_flag = False

    def get_title_from_txt(self, txt_file):
        # 打开txt文件
        title_list = []
        for line in open(txt_file, "r", encoding='utf-8'):
            title_list.append(line)
        return title_list

    def dowmload_img(self, html, keyword, image_name):
        num = 0
        pic_url = re.findall('"objURL":"(.*?)",', html, re.S)  # 先利用正则表达式找到图片url
        if not pic_url:
            self.temp_lst.append(keyword)
        for each in pic_url:
            print('正在下载第' + str(num + 1) + '张图片，图片地址:' + str(each))
            try:
                if each is not None:
                    pic = requests.get(each, timeout=7)
                else:
                    continue
            except BaseException:
                print('错误，当前图片无法下载')
                continue
            else:
                string = self.img_data_path + image_name + '.jpg'
                fp = open(string, 'wb')
                fp.write(pic.content)
                fp.close()
                num += 1
            if num >= 1:
                return
            # else:
            #     print("没有图片")

    def get_img(self, title_list, image_name):
        headers = {
            'Accept-Language': 'zh-CN,zh;q=0.8,zh-TW;q=0.7,zh-HK;q=0.5,en-US;q=0.3,en;q=0.2',
            'Connection': 'keep-alive',
            'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64; rv:60.0) Gecko/20100101 Firefox/60.0',
            'Upgrade-Insecure-Requests': '1'
        }
        A = requests.Session()
        A.headers = headers

        # 去除特殊字符，只保留汉字，字母、数字
        title_list = re.sub(u"([^\u4e00-\u9fa5\u0030-\u0039\u0041-\u005a\u0061-\u007a])", "", title_list)

        url = 'https://image.baidu.com/search/flip?tn=baiduimage&ie=utf-8&word=' + title_list
        t = 0
        tmp = url
        numPicture = 2
        while t < numPicture:
            try:
                url = tmp + str(t)
                result = A.get(url, timeout=10, allow_redirects=False)
            except error.HTTPError as e:
                print('网络错误，请调整网络后重试')
                t = t + 60
            else:
                print('当前下载的图片主题为:', title_list)
                self.dowmload_img(result.text, title_list, image_name)
                t = t + 60


    def per_data(self, txt_title):
        txt_title = txt_title.replace('”', '')
        txt_title = txt_title.replace('“', '')
        txt_title = txt_title.replace('?', '')
        txt_title = txt_title.replace('？', '')
        txt_title = txt_title.replace('%', '')
        txt_title = txt_title.replace('《', '')
        txt_title = txt_title.replace('》', '')
        txt_title = txt_title.replace('-', '')
        txt_title = txt_title.replace(':', '')
        return txt_title

    def main(self):
        # 从txt中获取标题并爬取图片
        if self.txt_flag:
            # 得到txt中的标题
            title_list = self.get_title_from_txt(self.txt_path)
            for i in range(len(title_list)):
                title = self.per_data(title_list[i])
                title_name = title.rstrip('\n')  # 替换换行符
                image_name = str(i) + '-' + title_name
                self.get_img(title, image_name)

if __name__ == '__main__':
    get_img().main()
