import os
from random import sample

rootPath = os.path.join(os.getcwd(),"150_ChIP-seq_Datasets")
dirs=os.listdir(rootPath)

global150_train_filename = os.path.join(os.getcwd(), "data_pretrain","150_global", "original_data",'train.data')
global150_test_filename = os.path.join(os.getcwd(), "data_pretrain","150_global", "original_data",'test.data')
global150_valid_filename = os.path.join(os.getcwd(), "data_pretrain","150_global", "original_data",'valid.data')

train_data=open(global150_train_filename,'w')
test_data=open(global150_test_filename,'w')
valid_data=open(global150_valid_filename,'w')
train_data.truncate(0)
test_data.truncate(0)
valid_data.truncate(0)

train_fre=[]
test_fre=[]
valid_fre=[]

# train=[]
for now_dir in dirs:
    now_dirpath=os.path.join(rootPath,now_dir)
    m = 99999999
    for file in os.listdir(now_dirpath):
        now_filepath=os.path.join(now_dirpath,file)
        if(file=="test.data"):
            now_file=open(now_filepath,'r')
            lines=now_file.readlines()
            test_fre.extend(lines)
            # test_data.writelines(lines)
        num=0

        if (file == "train.data"):
            now_file = open(now_filepath, 'r')
            lines = now_file.readlines()
            large=5000
            small=500
            if(len(lines)>large):
                # print(len(lines))
                now=sample(lines, large)
                train_fre.extend(now) # 0.1*0.8
                # print(len(lines))
                if((len(lines)-large)>small):
                    valid_fre.extend(sample(list(set(lines).difference(set(now))),small))
                else:
                    valid_fre.extend(sample(list(set(lines).difference(set(now))), len(lines)-large))
            else:
                train_fre.extend(lines)
                valid_fre.extend(sample(lines, small))
                # valid_fre.extend(sample(list(set(lines).difference(set(now))), 800))

            # test_data.writelines(lines)
train_num=int(len(train_fre))#随机选取数据
valid_num=int(len(valid_fre))
test_num=int(len(test_fre)*0.07)
#
# train=sample(train_fre,train_num)#0.1*0.8
# valid =sample(list(set(train_fre).difference(set(train))),valid_num)
test_fre=sample(test_fre,test_num)

train_data.writelines(train_fre)
test_data.writelines(test_fre)
valid_data.writelines(valid_fre)

print(train_num,test_num,valid_num)
train_data.close()
test_data.close()
valid_data.close()