# 修改成bio代码.


# tokenizer = AutoTokenizer.from_pretrained("ckiplab/bert-base-chinese-ner")
#
# model = AlbertForTokenClassification.from_pretrained("ckiplab/bert-base-chinese-ner")
#
# model_name='ckiplab/bert-base-chinese-ner'


from transformers import AlbertTokenizer, AlbertForTokenClassification  # 版本 3.1
import torch
import torch.nn as nn
tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')


with open("dictforalbert", 'w', encoding='utf-8') as f :
    f.writelines(str(tokenizer.get_vocab()))




model = AlbertForTokenClassification.from_pretrained('albert-base-v2')
model.num_labels=3
model.classifier = nn.Linear(768, 3)
inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
labels = torch.tensor([1] * inputs["input_ids"].size(1)).unsqueeze(0)  # Batch size 1
outputs = model(**inputs, labels=labels)












# ---------------注意先检测 transformer版本    pip3 install  transformers==3.4

# ------transformer 里面 配置都在model里面的参数中!!!!!!!!!!!!!!!




from transformers import AlbertTokenizer, AlbertForTokenClassification
import torch
# tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')
# model = AlbertForTokenClassification.from_pretrained('albert-base-v2')
# inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
# labels = torch.tensor([1] * inputs["input_ids"].size(1)).unsqueeze(0)  # Batch size 1
# outputs = model(**inputs, labels=labels)
# print(1)













seq_max_len = 512  # 下面目标是吧模型参数里面position改成这么长的,管他N方复杂度呢, 能算atttion就行.注意配置里面config也改了.
# config = model.config
# config.max_position_embeddings = seq_max_len
#
# # 乘以负1w,这样就保证了之前的注意力还是最大,不影响之前注意力.不行,这个还是不行.这个是嵌入,不是最后注意力,所以给多少都说不好.所以还是取0靠谱.
# quanzhognold = torch.cat((model.base_model.embeddings.position_embeddings.weight,
#                           torch.zeros(config.max_position_embeddings - 512, config.embedding_size)), dim=0)
#
# model.base_model.embeddings.position_embeddings = nn.Embedding(config.max_position_embeddings, config.embedding_size)
# model.config = config
# model.base_model.embeddings.position_embeddings.weight.data = quanzhognold
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#####nlp4里加上这句报错，拿出来的话，不加报错？？？？？？？？？？？？？？？
#model.base_model.embeddings.position_ids.data = torch.range(0, seq_max_len - 1, dtype=torch.long).view(1, -1)
model.to(device)
if 0:
    if torch.cuda.device_count() > 1:
        print("显卡数量：",torch.cuda.device_count())
        model = torch.nn.DataParallel(model)# if device=='cuda:0' else model

#torch.distributed.init_process_group('nccl',init_method='file:///home/.../my_file',world_size=1,rank=0)
#model = torch.nn.parallel.DistributedDataParallel(model)
#voidful/albert_chinese_xxlarge
#tokenizer = BertTokenizer.from_pretrained('voidful/albert_chinese_xxlarge')
from transformers import pipeline

#from transformers import BartTokenizer, BartForQuestionAnswering
import torch  # 试着在这个上面做finetune


# 可以手动去https://huggingface.co/valhalla/bart-large-finetuned-squadv1#   下载需要的文件.


# question, text = "我是谁", "我是朱光旭啊啊啊啊啊啊啊啊啊啊啊啊"+'啊'*600
# # question, text = "核电占发电量比例最大的是哪个国家?", "在世界主要工业大国中，法国核电的比例最高，核电占国家总发电量的78%，位居世界第二，日本的核电比例为40%，德国为33%，韩国为30%，美国为22%"
# '''
# [cls] 我 是 谁 [sep] 我 是 朱 光 旭 啊啊啊啊啊啊啊啊啊啊啊啊 [sep]
# '''
# encoding = tokenizer(question, text, return_tensors='pt')
# input_ids = encoding['input_ids'].to(device)
# attention_mask = encoding['attention_mask'].to(device)

# input_data = os.path.join('data','time3.txt')







def mycollate_fn(data):
    max_shape=0
    for input_ids, attention_mask, out in data:
        shape = input_ids.shape[1]
        if shape>seq_max_len:
            max_shape=seq_max_len
        elif shape>max_shape:
            max_shape=shape

    input_ids_batch=[]
    attention_mask_batch=[]
    out_batch=[]

    for input_ids, attention_mask, out in data:
        shape=input_ids.shape[1]
        if shape<=seq_max_len: #判断句子长度.补齐到max_shape
            input_ids = torch.cat(
                (input_ids, torch.zeros([1, max_shape - shape ]).to(
                    torch.long)), dim=1).to(torch.long)
            attention_mask = torch.cat(
                (attention_mask, torch.zeros([1, max_shape - shape]).to(
                    torch.long)),dim=1).to(torch.long)
        else:
            print('===========')
            input_ids = input_ids[:,:seq_max_len].to(torch.long)
            attention_mask = attention_mask[:,:seq_max_len].to(torch.long)

        input_ids_batch.append(input_ids)
        attention_mask_batch.append(attention_mask)
        out_batch.append(out)

    #return input_ids_batch,attention_mask_batch,start_batch,end_batch

    return torch.cat(input_ids_batch,dim=0),\
           torch.cat(attention_mask_batch,dim=0),\
           torch.cat(out_batch,dim=0)


# ------------下面我们做模型拼接:-------拼接不了, 旧模型输出1024, 新模型输出2048.完全不一样的.

#
# model_name2='wptoux/albert-chinese-large-qa'
# # 注意这个模型里面没有大写英文字母.所以需要我们手动转化.
#
#
#
#
#
# #input_data = os.path.join('data','me_train.txt')
# model2 = AutoModelForQuestionAnswering.from_pretrained(model_name2)  # 64mb
#
#
# model.qa_outputs=model2.qa_outputs
# 把model2 拼接到model上.


print(1111)






from torch.utils.data import DataLoader, TensorDataset

label2id={'B':0,'I':1,'O':2}



from DealDataset import DealDataset
if 1:
    print('start_train')

    # 开启finetune模式 ,,,,,,,C:\Users\Administrator\.PyCharm2019.3\system\remote_sources\-456540730\-337502517\transformers\data\processors\squad.py 从这个里面进行抄代码即可.

    batch_size=4
    a='data/ner1.txt'
    b='data/ner2.txt'
    dealDataset = DealDataset(a,b,tokenizer,label2id)
    train_loader = DataLoader(dataset=dealDataset,
                              batch_size=batch_size,
                              shuffle=True,
                              pin_memory=True,
                              collate_fn=mycollate_fn)
    n_epoch=25
    cold=10
    cold_lr=3e-4
    lr=3e-5
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)


    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=100, verbose=True)
    if cold > 0:
        for param_group in optimizer.param_groups:
            param_group['lr'] = cold_lr
        for param in model.base_model.parameters():
            param.requires_grad = False


        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        #     optimizer, factor=0.1, patience=10)
        # optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    for i in range(n_epoch):
        model.train()
        if i == cold:
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            for param in model.base_model.parameters():
                param.requires_grad = True
        print('当前epoch学习率',optimizer.param_groups[0]['lr'])
        #for input_ids,attention_mask,start,end in train_loader:
        for  input_ids_batch,attention_mask_batch,out in train_loader:
            inputs = {
                "input_ids": input_ids_batch.to(device),
                "attention_mask": attention_mask_batch.to(device),
                "labels": out.to(device),

            }
            outputs = model(**inputs)
            loss = outputs[0]
            ####多gpu时返回多个loss，需要取均值
            loss=loss.mean()
            print(loss,"当前loss")
            loss.backward(loss.clone().detach())
            #loss.backward()
            optimizer.step()
            model.zero_grad()
            scheduler.step(loss)  # 加入策略.
            #early_stopping(loss, model)

            # if early_stopping.early_stop:
            # # if loss.item()<0.001:
            #     print("early stop!!")
            #     print("train_over!!!!!!!!!!")
            #     torch.save(model, r'models/time.pkl')
            #     return
        print('epoch over:',i)
        #对于最后一轮直接做测试:
        if i==n_epoch-1:
            model.eval()
            outputs = model(**inputs)
            print('输出outputs',outputs[1])
            out=outputs[1]
            a=outputs[1].argmax(dim=-1)
            print('我们输出的标签',a)
            print('ground_true',inputs['labels'])
            # 根据0,1,2抽取即可.
            # 下面解析out

            # input_ids=inputs['input_ids']
            # start_scores=outputs[1]
            # end_scores=outputs[2]
            # print('下面打印最后一个epoch的一个batch考题.')
            # for a,b,c in zip(input_ids,start_scores,end_scores):
            #     for dex in range(len(a)):
            #         if a[dex] == tokenizer.encode('[SEP]')[1]:
            #             break
            #     b[ :dex + 1] = -float("inf")
            #     c[ :dex + 1] = -float("inf")
            #
            #     all_tokens = tokenizer.convert_ids_to_tokens(a)
            #     start = torch.argmax(b)
            #     end = torch.argmax(c) + 1
            #     print(start, end, 'for  debug')
            #     if end < start:
            #         end, start = start, end
            #
            #     answer = ' '.join(all_tokens[start:end])
            #     answer = tokenizer.convert_tokens_to_ids(answer.split())
            #     answer = tokenizer.decode(answer)
            #
            #     print(answer)



        # if i%100==0:
        #     #torch.save(model, r'models/checkpoint_model_epoch_{}.pth.tar'.format(i))
        #     torch.save(model, r'models/time.pkl')
        #     print('第{}次迭代完成！'.format(i))
    print("train_over!!!!!!!!!!")






