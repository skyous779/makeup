import mindspore.dataset as ds
from dataset import mydataset
from tqdm import tqdm
import mindspore as ms
from model import RNN
import numpy as np
import os
os.system('pip install gensim')
from gensim.models.keyedvectors import KeyedVectors

import argparse
parser = argparse.ArgumentParser(description='manual to this script')
parser.add_argument("--voc_model_path", type=str, default="../sgns.wiki.char.bz2", help='词向量模型')
parser.add_argument("--datapath", type=str, default='../data')
parser.add_argument("--ckpt_file_name", type=str, default='../sentiment-analysis.ckpt', help='推理模型')
parser.add_argument("--result_dir", type=str, default='./result.txt', help='结果保存路径')


args = parser.parse_args()


def predict_result(model, test_dataset):
    total = test_dataset.get_dataset_size()
    epoch_loss = 0
    epoch_acc = 0
    step_total = 0
    model.set_train(False)
    result = []
    with tqdm(total=total) as t:
        # t.set_description('Epoch %i' % epoch)
        for i in test_dataset.create_tuple_iterator():
            predictions = model(i[0])
            predictions = predictions.asnumpy()
            result.append(predictions)
            t.update(1)

    return result   


if __name__ == '__main__':
    
    # 超参数
    hidden_size = 256
    embedding_dim = 300
    output_size = 1
    num_layers = 2
    bidirectional = True
    dropout = 0.5
    lr = 0.001
    
    # 词库路径
    voc_model_path = args.voc_model_path

    # 加载词库
    print('Start loading voc_model!')
    voc_model = KeyedVectors.load_word2vec_format(voc_model_path, 
                                                  binary=False, 
                                                  unicode_errors='ignore')
    print('Success loading voc_model!')

    ms.set_context(mode=ms.PYNATIVE_MODE, device_target ="Ascend")
    ckpt_file_name = args.ckpt_file_name
    param_dict = ms.load_checkpoint(ckpt_file_name)
    net = RNN(embedding_dim, hidden_size, output_size, num_layers, bidirectional, dropout)
    ms.load_param_into_net(net, param_dict)


    datapath = args.datapath
    test_dataset = mydataset(datapath, voc_model, 'test')
    test_dataset = ds.GeneratorDataset(test_dataset, column_names=["embedding", "label"], shuffle=False)

    type_cast_op = ds.transforms.TypeCast(ms.float32)
    test_dataset = test_dataset.map(operations=[type_cast_op], input_columns=['label'])
    test_dataset = test_dataset.map(operations=[type_cast_op], input_columns=['embedding'])

    test_dataset = test_dataset.batch(1, drop_remainder=False)



    
    


    myresult = predict_result(net, test_dataset)

    # 结果写入txt
    bool_results = myresult.copy()
    f1 = open(args.result_dir,'w', encoding='utf-8')
    for i, result in enumerate(myresult):
        if result > 0.5:
            f1.write('1')
        else :
            f1.write('0')
        f1.write('\n')
    f1.close()