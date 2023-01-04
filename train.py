from dataset import mydataset
from model import RNN
import mindspore.dataset as ds
import mindspore as ms
import mindspore.nn as nn
from tqdm import tqdm
import numpy as np
import os
os.system('pip install gensim')
from gensim.models.keyedvectors import KeyedVectors

# 日志保存
import sys
class Logger(object):
    def __init__(self, filename='default.log', stream=sys.stdout):
        self.terminal = stream
        self.log = open(filename, 'a')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.terminal.flush()  # 不启动缓冲,实时输出
        self.log.flush()

    def flush(self):
        pass

import argparse
parser = argparse.ArgumentParser(description='manual to this script')
parser.add_argument("--voc_model_path", type=str, default="../sgns.target.word-character.char1-2.dynwin5.thr10.neg5.dim300.iter5.bz2", help='词向量模型')
parser.add_argument("--datapath", type=str, default='../data')
parser.add_argument("--cache_dir", type=str, default='./checkpoint')
parser.add_argument("--save_log_dir", type=str, default='./train_log.log')
args = parser.parse_args()


sys.stdout = Logger(args.save_log_dir, sys.stdout)
sys.stderr = Logger(args.save_log_dir, sys.stderr)


def train_one_epoch(model, train_dataset, epoch=0):
    model.set_train()
    total = train_dataset.get_dataset_size()
    loss_total = 0
    step_total = 0
    with tqdm(total=total) as t:
        t.set_description('Epoch %i' % epoch)
        for i in train_dataset.create_tuple_iterator():
            loss = model(*i)
            loss_total += loss.asnumpy()
            step_total += 1
            t.set_postfix(loss=loss_total/step_total)
            t.update(1)
            
def binary_accuracy(preds, y):
    """
    计算每个batch的准确率
    """

    # 对预测值进行四舍五入
    rounded_preds = np.around(preds)
    correct = (rounded_preds == y).astype(np.float32)
    acc = correct.sum() / len(correct)
    return acc

def evaluate(model, test_dataset, criterion, epoch=0):
    total = test_dataset.get_dataset_size()
    epoch_loss = 0
    epoch_acc = 0
    step_total = 0
    model.set_train(False)

    with tqdm(total=total) as t:
        t.set_description('Epoch %i' % epoch)
        for i in test_dataset.create_tuple_iterator():
            predictions = model(i[0])
            loss = criterion(predictions, i[1])
            epoch_loss += loss.asnumpy()

            acc = binary_accuracy(predictions.asnumpy(), i[1].asnumpy())
            epoch_acc += acc

            step_total += 1
            t.set_postfix(loss=epoch_loss/step_total, acc=epoch_acc/step_total)
            t.update(1)

    return epoch_loss / total

if __name__ == '__main__':
    
    
    # 词库路径
    voc_model_path = args.voc_model_path

    # 加载词库
    print('Start loading voc_model!')
    voc_model = KeyedVectors.load_word2vec_format(voc_model_path, 
                                                  binary=False, 
                                                  unicode_errors='ignore')
    print('Success loading voc_model!')
    
    
    
    ms.set_context(mode=ms.PYNATIVE_MODE, device_target ="Ascend")

    # 数据加载
    # it may take 6 mins ！
    datapath = args.datapath
    data_train = mydataset(datapath, voc_model, 'train')


    train_loader = ds.GeneratorDataset(data_train, column_names=["embedding", "label"], shuffle=True)

    type_cast_op = ds.transforms.TypeCast(ms.float32)
    train_loader = train_loader.map(operations=[type_cast_op], input_columns=['label'])
    train_loader = train_loader.map(operations=[type_cast_op], input_columns=['embedding'])

    train_loader, val_loader = train_loader.split([0.8, 0.2])

    train_loader = train_loader.batch(64, drop_remainder=True)
    val_loader = val_loader.batch(64, drop_remainder=True)

    # 超参数
    hidden_size = 256
    embedding_dim = 300
    output_size = 1
    num_layers = 2
    bidirectional = True
    dropout = 0.5
    lr = 0.001
    net = RNN(embedding_dim, hidden_size, output_size, num_layers, bidirectional, dropout)

    # 训练配置
    loss = nn.BCELoss(reduction='mean')
    net_with_loss = nn.WithLossCell(net, loss)
    optimizer = nn.Adam(net.trainable_params(), learning_rate=lr)
    train_one_step = nn.TrainOneStepCell(net_with_loss, optimizer)


    cache_dir = args.cache_dir
    os.makedirs(cache_dir, exist_ok=True)
    num_epochs = 100
    best_valid_loss = float('inf')
    ckpt_file_name = os.path.join(cache_dir, 'sentiment-analysis.ckpt')

    for epoch in range(num_epochs):
        train_one_epoch(train_one_step, train_loader, epoch)
        valid_loss = evaluate(net, val_loader, loss, epoch)

        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            ms.save_checkpoint(net, ckpt_file_name)
