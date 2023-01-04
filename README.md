# 第四届MindCon-外卖评论文本分类
 
```text 
 ├── data
        ├─train
        │   ├ data.txt
        │   └─label.txt    
        ├─test
            └─test.txt  
```
## 训练脚本
```bash
python train.py --voc_model_path ../sgns.wiki.char.bz2 --datapath ./data --cache_dir ./checkpoint  --save_log_dir ./train_log.log
```
- voc_model_path: 词向量模型
- datapath: 数据集路径
- cache_dir: 权重模型保存文件夹
- save_log_dir：日志保存文件名



## 推理脚本
```bash
python predict.py --voc_model_path ../sgns.wiki.char.bz2 --datapath ./data --ckpt_file_name ../sentiment-analysis.ckpt  --result_dir ./result.txt
```
- voc_model_path: 词向量模型
- datapath: 数据集路径
- ckpt_file_name： 推理模型
- result_dir ： 推理结果保存文件名