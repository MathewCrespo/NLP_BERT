import pandas
import json
import matplotlib.pyplot as plt
import numpy as np
import os

root='F:/study/sem6/NLP/log'

log = ['log_waimai', 'log_weibo', 'log_weibo4', 'weibo22', 'log_Chicorp', 'log_LCQMC']

for file_name in log:
    log_txt = open(root + '/'+ file_name + '.txt')
    iter_file = iter(log_txt)
    loss_all=[]
    acc_all=[]


    for line in iter_file:
        if line.find("lr")>-1 & line.find("loss")>-1:
            loss=line.split("loss=")[1]
            loss=loss.split(", lr")[0]
            loss_all.append(float(loss))
            acc = line.split("accuracy:")[1]
            acc = acc.split("\n")[0]
            acc_all.append(float(acc)) 
    loss = np.array(loss_all)
    acc = np.array(acc_all)
    plt.plot(acc)
    plt.xlabel('iterations')
    plt.ylabel('Metric:Accuracy')
    plt.title('Accuracy')

    plt.savefig(root+'/'+ file_name+ '_acc.png')
    plt.close()

    plt.plot(loss)
    plt.xlabel('iterations')
    plt.ylabel('Metric:Loss')
    plt.title('Loss')
    plt.savefig(root+'/'+ file_name+'_loss.png')
    plt.close()

