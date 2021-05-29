import os

import torch
from torch.utils.data  import  Dataset,DataLoader,TensorDataset
from torch.autograd import Variable
import numpy as np
from transformers import BertTokenizer
import linecache
from kmp_for_array import kmp

class DealDataset(Dataset):

    def __init__(self,a,b,tokenizer,label2id):
        self.label2id=label2id
        self.tokenizer = tokenizer
        self.input_data = a
        self.tag = b
        self.count = -1  # 这个行代码是为了防止读取空白文件,时候bug.
        for self.count, line in enumerate(open(self.input_data,encoding='utf8')):
            pass  # 这行是计算行数.
        self.count += 1  # +1才对.

    def __getitem__(self, index):
        '''
        修复bug, 注意句首和居中时候编码的不同.

        Args:
            index:

        Returns:

        '''
        text = linecache.getline(self.input_data, index + 1).strip()
        tag = linecache.getline(self.tag, index + 1).strip()
        tag=tag.split(' ')
        # tag=[self.label2id[i] for i in tag]

        #
        # text=all[0]
        # answer=all[1:]
        #
        #
        # text=text.lower()
        # answer=[i.lower() for i in answer]

        # 计算bio的标签位置.

        #Qquestion = '时间'
        encoding = self.tokenizer( text, return_tensors='pt')['input_ids'][0]


        # 我们需要找到tag对应的encoding中的索引位置.



    # 头尾加上O
        '''
        注意英文编码的问题跟中文的区别.
        中文里面以为每一个词的前面都没有空格.所以  我是跟  打我 这2种里面我的编码是不同的.
        英文里面  i see ,      beat i  这2个里面的i编码是相通的. 因为任何东西前面都带空格.
        所以英文直接kmp算法即可. 不用再像汉语分2种情况, 句中和句首区分.


        '''
        out= torch.tensor([2] * encoding.size(0))       # Batch size 1
        for  j in tag:
            obj=self.tokenizer( j, return_tensors='pt')['input_ids'][0][1:-1]
            listFortagIndex=kmp(   encoding,   obj ,return_all=True )
            for kk in listFortagIndex:
                 out[kk:  kk+len(obj) ]   =0  # 补上b
                 out[kk+1:  kk+len(obj) ]   =1  # 补上i

            # print(1)
        out=out.unsqueeze(0)

        # out=tag    # 0 表示other
        # for i in answer:
        #     # ans=self.tokenizer( i, return_tensors='pt')['input_ids'][1:-1]
        #     ans=    self.tokenizer( i, return_tensors='pt')['input_ids'][0][1:-1]
        #     # kaitou = kmp(encoding.numpy(), ans.numpy())
        #     kaitou = kmp(list(encoding.numpy()), list(ans.numpy()))
        #     end = kaitou + len(ans) - 1 # 结尾点所在的坐标.
        #     # 填补b 和i
        #     out[kaitou]=1
        #     for j in range(kaitou+1,end+1):
        #         out[j]=2          # baisohi i
        # print(1111111111111111)



#         input_ids = encoding['input_ids']
#         attention_mask = encoding['attention_mask']
#
#         daan_suoyin = self.tokenizer(answer)['input_ids'][1:-1]
#         all_suoyin = encoding['input_ids']
# # ---------------计算开始和结尾坐标!!!!!!!!!!!!!
#         kaitou = kmp(list(all_suoyin[0].numpy()), daan_suoyin)
#         end = kaitou + len(daan_suoyin) - 1
#         # 输入:question, text, answer 返回索引.
#         if answer == 'no answer':
#             start, end= torch.tensor([-1]),torch.tensor([-1])
#         else:
#             start, end = torch.tensor([kaitou]),torch.tensor([end])
        return  self.tokenizer( text, return_tensors='pt')['input_ids'],self.tokenizer( text, return_tensors='pt')['attention_mask'],out

    def __len__(self):
        return self.count

# if __name__ == '__main__':
#     d=DealDataset('data/time2.txt')
#     print(d[2])

