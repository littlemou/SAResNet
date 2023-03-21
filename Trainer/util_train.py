import os
import sys
import xlrd

from Trainer.SAResNetTrainer import SAResNetTrainer
from models.SAResNet_Model import SAResnetModel
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
#对子模型中数据量较小的进行重新训练，batchsize由1024降为256
def second_train(threshold,kernal_channel,fc_num,batch_size):
    col=2
    m=0
    num=0
    print(rootPath)
    name_path=os.path.join(rootPath, "DataSet","data_sub")
    ckpt_save_path = os.path.join(rootPath, "save_weights")
    xls_name1="SAResNet_batchSize=1024_learnRate=0.0001.xls"
    xls_name2 = "SAResNet_batchSize=256_learnRate=5e-05.xls"
    for file in os.listdir(name_path):
        now_path=os.path.join(ckpt_save_path,file)
        if(os.path.exists(os.path.join(now_path,xls_name2))):
            # print("喀喀喀")
            now_xls=os.path.join(now_path,xls_name2)
        else:
            now_xls = os.path.join(now_path, xls_name1)
        print(now_xls)
        workBook=xlrd.open_workbook(now_xls)
        work_sheet=workBook.sheet_by_name("Sheet1")
        # a=work_sheet.cell()
        for i in range(1,work_sheet.nrows):
            acc=work_sheet.cell_value(i,col)
            m=max(acc,m)
        # print(m)
        if(m<threshold):
            num+=1
            # trainData_Path=os.path.join(name_path,file)
            # now_ckpt_save_path=os.path.join(ckpt_save_path,file)
            # model = SAResnetModel(kernal_channel=kernal_channel, fc_num=fc_num)
            # trainer = SAResNetTrainer(model=model, data_path=trainData_Path, batch_size=batch_size,learning_rate=0.00005)
            # trainer.train_save(epochs=30, ckpt_save_path=now_ckpt_save_path)
        print(num)
        m=0

if __name__=='__main__':
    second_train(0.6,64,32,256)