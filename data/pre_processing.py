import pandas as pd 
import math

root='F:/study/sem6/NLP/bert_new/bert_new/data'
data=pd.read_csv(root+'/simplifyweibo_4_moods.csv')
label_flag=[0,1,2,3]
class_num=[0,0,0,0]

split_num=[0.7, 0.2, 0.1]
label_info = data["label"].unique()
print(label_info)
for temp_info in label_info:
    temp_data = data[data['label'].isin([temp_info])]
    exec("data%s = temp_data" %temp_info)

for i in range(4):
    exec("temp=data%s" %i)
    class_num[i] = temp.shape[0]
    train_num = math.ceil(0.7*class_num[i])
    test_num=math.ceil(0.2*class_num[i])
    exec("data_train%s=temp[0:train_num]" %i)
    exec("data_test%s=temp[train_num:train_num+test_num]" %i)
    exec("data_dev%s=temp[train_num+test_num:class_num[i]]" %i)

train_set=pd.DataFrame()
test_set=pd.DataFrame()
dev_set=pd.DataFrame()
for i in range(4):
    exec("train_set=train_set.append(data_train%s)" %i)
    exec("test_set=test_set.append(data_test%s)" %i)
    exec("dev_set=dev_set.append(data_dev%s)" %i)
print(dev_set)

order=['review','label']
train_set=train_set[order]
test_set=test_set[order]
dev_set=dev_set[order]

train_set.to_csv(root+'/weibo2_train.tsv', sep='\t', index = False, header = None)
test_set.to_csv(root+'/weibo2_test.tsv', sep='\t', index = False, header = None)
dev_set.to_csv(root+'/weibo2_dev.tsv', sep='\t', index = False, header = None)