#加载3-4分钟时间
import os
os.system('pip install gensim')
os.system('pip install jieba')

import jieba
import os
import numpy as np
from gensim.models.keyedvectors import KeyedVectors

# 词库路径
voc_model_path = '../sgns.target.word-character.char1-2.dynwin5.thr10.neg5.dim300.iter5.bz2'

# 加载词库
print('Start loading voc_model!')
voc_model = KeyedVectors.load_word2vec_format(voc_model_path, 
                                              binary=False, 
                                              unicode_errors='ignore')
print('Success loading voc_model!')

class mydataset():
    label_map = {"正向评价": 1 ,"负向评价": 0}     
    
    def __init__(self, path, voc_model, mode="train"):
        self.mode = mode
        self.path = path
        self.docs, self.labels = [], []
        # self.punctuation = '!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~，。！；‘、^"“”⚘❀༵' #标点符号
        self.embeddings = []

        #词向量模型
        self.model = voc_model
        
        self._load()

    def _load(self):
        
        if self.mode == 'train':
            f = open(os.path.join(self.path, self.mode, 'data.txt')) # datatest.txt 用于测试
            
            for line in f:
                embedding = np.zeros((100, 300))
                label = line.split(',', 1)[0]

                text = line.split(',', 1)[-1]
                text = text.strip('\n')

                # for i in self.mypunctuation:
                #     text = text.replace(i,'')
                tokens = jieba.lcut(text, cut_all=False)  # 得到分词组  
                # print('text', text, 'tokens', tokens)
                for i, token in enumerate(tokens):
                    if i == 100:
                        break
                    try:
                        # print('token:', self.model.get_vector(token))
                        embedding[i] = np.array(self.model.get_vector(token))
                    except KeyError as kr:
                        embedding[i] = np.random.rand(300)
                        # print('Now exiting key not present:', token)
                # while i < 99:
                #     embedding.append(np.zeros((100,), np.float32))
                #     i += 1 
                # print('len of embedding:', len(embedding))        
                self.embeddings.append(embedding)
                self.labels.append(int(label))
                
        
        else :
            f = open(os.path.join(self.path, self.mode, 'test.txt'))
            for line in f:
                embedding = np.zeros((100, 300))
                text = line.strip('\n')
                tokens = jieba.lcut(text, cut_all=False)  # 得到分词组            
                for i, token in enumerate(tokens):
                    if i == 100:
                        break
                    try:
                        # print('token:', self.model.get_vector(token))
                        embedding[i] = np.array(self.model.get_vector(token))
                    except KeyError as kr:
                        embedding[i] = np.random.rand(300)
                        # print('Now exiting key not present:', token)

                self.embeddings.append(embedding)
                self.labels.append(-1) # 没有label
           
    def __getitem__(self, idx):
        return self.embeddings[idx], self.labels[idx]

    def __len__(self):
        return len(self.embeddings)
    


