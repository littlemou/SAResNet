import os
import numpy as np
import tensorflow as tf

def parse(seq,label):
    # print(seq)
    seq = tf.reshape(seq, [1, 101, 4])
    seq = tf.cast(seq, dtype=tf.float32)
    return seq,label

def embed(seq, mapper, worddim):
    mat = np.asarray([mapper[element] if element in mapper else np.random.rand(worddim)*2-1 for element in seq])
    return mat

def onehot(index):
    if(index == '0'):
        return [1, 0]
    elif(index == '1'):
        return [0, 1]

def  digital(n):
    res=list(map(int,str(n)))
    res =np.array(res) #转为array格式
    return res

def dna2Sequence(dna):
    mapper = {'A': [1, 0, 0, 0], 'C': [0, 1, 0, 0], 'G': [0, 0, 1, 0], 'T': [0, 0, 0, 1], 'N': [1, 1, 1, 1]}
    mat = embed(dna, mapper, len(mapper['A']))
    result = mat.reshape(-1)  # 打成一维数组
    result = result.astype(int)
    return result

def get_data(file_open):
    res=[]
    res_label=[]
    temp = ""
    num=0
    for line in file_open.readlines():
        if (line[-2] == '}'):
            temp += line
            temp=temp[:-1]
            temp = temp.strip('{')
            temp = temp.strip('}')
        else:
            temp += line
            continue
        seq = temp.split(' ')[0]
        label = temp.split(' ')[1]
        res.append(digital(seq))
        res_label.append(onehot(label))
        temp = ""
        # num+=1
        # # print(label,onehot(label))
        # #测试_______________
        # if(num==100):
        #     break
        # _______________
    return res,res_label,num

def load_data(data_path):
    train_filename = os.path.join(data_path, 'train.data')
    test_filename = os.path.join(data_path, 'test.data')
    valid_filename = os.path.join(data_path, 'valid.data')

    train_data = open(train_filename, 'r')
    test_data = open(test_filename, 'r')
    valid_data = open(valid_filename, 'r')

    train,train_label,train_num=get_data(train_data)
    test,test_label,test_num=get_data(test_data)
    valid,valid_label,valid_num=get_data(valid_data)

    train_data.close()
    test_data.close()
    valid_data.close()
    return (train, train_label), (test, test_label), (valid, valid_label)

def load_data_evaluate(data_path):
    test_filename = os.path.join(data_path, 'test.data')
    test_data = open(test_filename, 'r')
    test,test_label,test_num=get_data(test_data)
    test_data.close()
    return  (test, test_label)
