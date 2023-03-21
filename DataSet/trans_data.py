import os
import numpy as np
# numpy.ndarray

# def embed(seq, mapper, worddim):
#     mat = np.asarray([mapper[element] if element in mapper else np.random.rand(worddim)*2-1 for element in seq])
#     return mat
#
# def dna2Sequence(dna):
#     mapper = {'A': [1, 0, 0, 0], 'C': [0, 1, 0, 0], 'G': [0, 0, 1, 0], 'T': [0, 0, 0, 1], 'N': [1, 1, 1, 1]}
#     mat = embed(dna, mapper, len(mapper['A']))
#     result = mat.reshape(-1)  # 打成一维数组
#     result = result.astype(int)
#     return result


def embed(seq, mapper):
    seqStr=""
    for element in seq:
        seqStr+=mapper[element]
    return seqStr
def dna2Sequence(dna):
    mapper = {'A':"1000", 'C': "0100", 'G':"0010",'T':"0001",'N':"1111"}
    mat = embed(dna, mapper)
    return mat

def trans_data(data_path,save_path):
    train_filename = os.path.join(data_path,"train.data")
    test_filename = os.path.join(data_path,"test.data")
    valid_filename = os.path.join(data_path,"valid.data")

    train_transfile = os.path.join(save_path,"train.data")
    test_transfile  = os.path.join(save_path,"test.data")
    valid_transfile  = os.path.join(save_path,"valid.data")

    train_data = open(train_filename, 'r')
    test_data = open(test_filename, 'r')
    valid_data = open(valid_filename, 'r')

    train_transfile =open(train_transfile, 'w')
    test_transfile =open(test_transfile, 'w')
    valid_transfile = open(valid_transfile, 'w')

    train = []
    test = []
    valid = []
    train_num=0
    test_num=0
    valid_num=0
    for line in train_data.readlines():
        seq = line.split(' ')[1]
        label = line.split(' ')[2][:-1]
        temp = "{" + str(dna2Sequence(seq)) + ' ' + str(label) + "}\n"
        train_transfile.write(temp)
        train_num+=1
    print("——————————训练数据转换完成")

    for line in test_data.readlines():
        seq = line.split(' ')[1]
        label = line.split(' ')[2][:-1]
        temp = "{" + str(dna2Sequence(seq)) + ' ' + str(label) + "}\n"
        test_transfile.write(temp)
        test_num+=1
    print("——————————测试数据转换完成")

    for line in valid_data.readlines():
        seq = line.split(' ')[1]
        label = line.split(' ')[2][:-1]
        temp = "{"+str(dna2Sequence(seq)) + ' ' + str(label) + "}\n"
        valid_transfile.write(temp)
        valid_num+=1
    print("——————————验证数据转换完成")
    train_data.close()
    test_data.close()
    valid_data.close()
    train_transfile.close()
    test_transfile.close()
    valid_transfile.close()
    return train_num,test_num,valid_num

if __name__=='__main__':
    #转换全局数据
    all_data_path = os.path.join(os.getcwd(),"data_pretrain","150_global")
    original_data_path=os.path.join(all_data_path,"original_data")
    global_save_path=os.path.join(all_data_path,"trans_data")
    train_num,test_num,valid_num=trans_data(original_data_path,global_save_path)
    print(train_num,test_num,valid_num)
    #转换子数据
    sub_data_path =os.path.join(os.getcwd(),"150_ChIP-seq_Datasets_addValid")
    save_path = os.getcwd()
    sub_save_path = os.path.join(save_path, "data_sub")
    for file in os.listdir(sub_data_path):
        now_data_path=os.path.join(sub_data_path,file)
        now_save_path=os.path.join(sub_save_path,file)
        if not os.path.exists(now_save_path):
            os.makedirs(now_save_path)
        train_num, test_num, valid_num = trans_data(now_data_path, now_save_path)
        print(train_num, test_num, valid_num)

