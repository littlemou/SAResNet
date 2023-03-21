import os
from random import sample

rootPath = os.path.join(os.getcwd(),"150_ChIP-seq_Datasets")
# dirs=os.listdir(rootPath)

for now_dir in os.listdir(rootPath):
    now_dirpath=os.path.join(rootPath,now_dir)#每个类别路径
    sub_train_file_path= os.path.join(os.getcwd(), "150_ChIP-seq_Datasets_addValid", now_dir)
    sub_test_file_path = os.path.join(os.getcwd(), "150_ChIP-seq_Datasets_addValid", now_dir)
    sub_valid_file_path= os.path.join(os.getcwd(), "150_ChIP-seq_Datasets_addValid", now_dir)
    if not os.path.exists(sub_train_file_path):
        os.makedirs(sub_train_file_path)
    if not os.path.exists(sub_test_file_path):
        os.makedirs(sub_test_file_path)
    if not os.path.exists(sub_valid_file_path):
        os.makedirs(sub_valid_file_path)
    sub_train_filename = os.path.join(sub_train_file_path,"train.data")
    sub_test_filename = os.path.join(sub_test_file_path,"test.data")
    sub_valid_filename = os.path.join(sub_valid_file_path,"valid.data")

    train_data = open(sub_train_filename, 'w')
    test_data = open(sub_test_filename, 'w')
    valid_data = open(sub_valid_filename, 'w')
    train_data.truncate(0)
    test_data.truncate(0)
    valid_data.truncate(0)

    train_fre = []
    test_fre = []
    valid_fre = []
    for file in os.listdir(now_dirpath):#每个data
        now_filepath=os.path.join(now_dirpath,file)
        # print(file)
        if(file=="test.data"):
            now_file=open(now_filepath,'r')
            lines=now_file.readlines()
            test_fre.extend(lines)
            # test_data.writelines(lines)
        if (file == "train.data"):
            now_file = open(now_filepath, 'r')
            lines = now_file.readlines()
            train_fre.extend(lines)
            # test_data.writelines(lines)

    train_num=int(len(train_fre)*0.8)#随机选取数据
    valid_num=int(len(train_fre)*0.2)
    test_num=int(len(test_fre))

    train=sample(train_fre,train_num)#0.8
    valid =sample(list(set(train_fre).difference(set(train))),valid_num)
    test=sample(test_fre,test_num)

    train_data.writelines(train)
    test_data.writelines(test)
    valid_data.writelines(valid)

    print(train_num,valid_num,test_num)
    train_data.close()
    test_data.close()
    valid_data.close()