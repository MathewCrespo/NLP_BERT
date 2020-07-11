import pandas as pd
import math
# 1 -- positive 0-- negative    1...1,0...0
root = 'F:/study/sem6/NLP/bert/bert/data/weibo_senti_100k/'
weibo = pd.read_csv(root+'weibo_senti_100k.csv')

# get number of positive and negative
positive_num = weibo[weibo.label== 1].shape[0]
negative_num = weibo[weibo.label== 0].shape[0]

# split the data set
    # positive
positive_train_num = math.ceil(0.7*positive_num)
positive_dev_num = math.ceil(0.1*positive_num)

positive_train_set = weibo[0: positive_train_num]
positive_dev_set = weibo[positive_train_num: positive_train_num+positive_dev_num]
positive_test_set = weibo[positive_train_num+positive_dev_num:positive_num]
    #negative
negative_train_num = math.ceil(0.7* negative_num)
negative_dev_num = math.ceil(0.1*negative_num)

negative_train_set = weibo[positive_num:positive_num + negative_train_num]
negative_dev_set = weibo[positive_num + negative_train_num: positive_num + negative_train_num + negative_dev_num ]
negative_test_set = weibo[positive_num + negative_train_num + negative_dev_num : positive_num + negative_num]
# trains/tests merge together
train_set = positive_train_set.append(negative_train_set)
test_set = positive_test_set.append(negative_test_set)
dev_set = positive_dev_set.append(negative_dev_set)

order=['review','label']
train_set=train_set[order]
test_set=test_set[order]
dev_set=dev_set[order]

train_set.to_csv(root+'train.0', sep='\t', index = False, header = None)
test_set.to_csv(root+'test.0', sep='\t', index = False, header = None)
dev_set.to_csv(root+'dev.0', sep='\t', index = False, header = None)