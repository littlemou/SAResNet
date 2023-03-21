import os

from matplotlib import pyplot as plt
import numpy as np
from data_loader.load_data import load_data
from models.SAResNet_Model import SAResnetModel
from Trainer.SAResNetTrainer import SAResNetTrainer
from Evaluater.Evaluater import Evaluater
import sys
from Evaluater.util_evaluate import *
from Trainer.util_train import second_train


def main(trainData_Path,ckpt_save_path,kernal_channel=64,fc_num = 32,batch_size = 128,epoch = 10,is_pretrain = True):
    if(is_pretrain):
        now_ckpt_save_path = os.path.join(ckpt_save_path, "pretrain_model_batchSize" + str(batchSize))
        # now_ckpt_save_path = ckpt_save_path
        if not os.path.exists(now_ckpt_save_path):
            os.makedirs(now_ckpt_save_path)
        model = SAResnetModel(kernal_channel=kernal_channel, fc_num=fc_num)
        trainer = SAResNetTrainer(model=model, data_path=trainData_Path, batch_size=batch_size,learning_rate=0.0001)
        trainer.train_save(epochs=epoch, ckpt_save_path=now_ckpt_save_path)
    else:
        num = 0
        for file in os.listdir(trainData_Path):
            num += 1
            print(str(num) + "]当前file：" + file)
            now_trainData_Path = os.path.join(trainData_Path, file)
            now_ckpt_save_path = os.path.join(ckpt_save_path, file)
            ckpt_path = os.path.join(ckpt_save_path, "pretrain_model_batchSize" + str(batchSize),
                                     "SAResNet.ckpt")  # 加载前一个模型
            if not os.path.exists(now_ckpt_save_path):
                os.makedirs(now_ckpt_save_path)
            else:
                continue
            # else:  # 存在，若准确率》0.75则跳过，否则训练
            #     xls_path = os.path.join(now_ckpt_save_path, "SAResNet" + "_batchSize=256_learnRate=5e-05.xls")
            #     if(os.path.exists(xls_path)):
            #         workBook = xlrd.open_workbook(xls_path)
            #         work_sheet = workBook.sheet_by_name("Sheet1")
            #         if (work_sheet.cell_value(work_sheet.nrows - 1, 2) > 0.75):
            #             print("跳过")
            #             continue
            #     xls_path2 = os.path.join(now_ckpt_save_path, "SAResNet" + "_batchSize=1024_learnRate=0.0001.xls")
            #     if (os.path.exists(xls_path2)):
            #         workBook = xlrd.open_workbook(xls_path2)
            #         work_sheet = workBook.sheet_by_name("Sheet1")
            #         if (work_sheet.cell_value(work_sheet.nrows - 1, 2) > 0.75):
            #             print("跳过")
            #             continue
            # model = loadModel(ckpt_path, kernal_channel=kernal_channel, fc_num=fc_num)#迁移学习
            model = SAResnetModel(kernal_channel=kernal_channel, fc_num=fc_num)#直接学习
            trainer = SAResNetTrainer(model=model, data_path=now_trainData_Path, batch_size=128, learning_rate=0.00001)
            trainer.train_save(epochs=15, ckpt_save_path=now_ckpt_save_path)

def TorF(temp):
    if (temp== '1'):
        return True
    elif (temp == '0'):
        return False
if __name__=='__main__':
    curPath = os.path.abspath(os.path.dirname(__file__))
    sys.path.append(os.path.split(curPath)[0])
    rootPath= os.path.split(curPath)[0]
    dataSet_path = os.path.join(rootPath,"DataSet")

    is_training=True
    is_pretrain=True

    is_evaluate=False

    batchSize=128
    kernal_channel=64
    if(is_training):
        if(is_pretrain):
            trainData_path = os.path.join(dataSet_path,"data_pretrain","150_global","trans_data")
        else:
            trainData_path = os.path.join(dataSet_path,"data_sub")
        ckpt_save_path = os.path.join(rootPath,"save_weights")

        # now_ckpt_save_path=os.path.join(ckpt_save_path, "pretrain_model_batchSize"+str(batchSize))
        now_ckpt_save_path=ckpt_save_path
        main(trainData_path,now_ckpt_save_path,kernal_channel=kernal_channel,fc_num = 32,batch_size = batchSize,epoch = 20,is_pretrain = is_pretrain)

    if(is_evaluate):
        globalData_Path =  os.path.join(dataSet_path,"data_pretrain","150_global","trans_data")
        subData_path= os.path.join(dataSet_path,"data_sub")
        ckpt_save_path = os.path.join(rootPath,"save_weights")
        ckpt_path = os.path.join(ckpt_save_path, "pretrain_model_batchSize"+str(batchSize), "SAResNet.ckpt")
        model=loadModel(ckpt_path,kernal_channel=kernal_channel,fc_num=32)
        paintGlobal_Transfer_AUC(globalCkpt_path=ckpt_path,subCkpt_path=ckpt_save_path,subData_path=subData_path,title="global_Transfer")

    # is_pretrain=True
    # data_path = r"D:\Python\PyCharm\workSpace\SAResNet_tensorflow2\DataSet\data_pretrain\150_global\trans_data"
    # ckpt_save_path = r"D:\Python\PyCharm\workSpace\SAResNet_tensorflow2\save_weights"
    # ckpt_name = "SAResNet.ckpt"
    # ckpt_path = os.path.join(ckpt_save_path, "pretrain_model",ckpt_name)
    # kernal_channel = 64
    # fc_num = 32
    # batch_size = 64
    # epoch = 10
    # for file in os.listdir(data_path):
    #     print(file)
    #     now_data_file=os.path.join(data_path,file)
    #     (train, train_label), (test, test_label), (valid, valid_label) = load_data(data_path)
    #     print("——————————加载数据结束")
    #     if(is_pretrain):
    #         now_ckpt_save_path = os.path.join(ckpt_save_path,"pretrain_model")
    #         model = SAResnetModel(kernal_channel=kernal_channel, fc_num=fc_num)
    #     else:
    #         now_ckpt_save_path = os.path.join(ckpt_save_path,file)
    #         model=SAResnetModel(kernal_channel=kernal_channel,fc_num=fc_num)
    #         model.load_weights(filepath=ckpt_path)
    #         model.trainable=True
    #         model.summary()
    #
    #     trainer = SAResNetTrainer(model=model, data_path=data_path, batch_size=batch_size)
    #     trainer.train_save(epochs=epoch, ckpt_save_path=now_ckpt_save_path, ckpt_name=ckpt_name)







