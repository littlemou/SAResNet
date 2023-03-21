import os
import xlrd
import xlwt
from matplotlib import pyplot as plt
import numpy as np
from models.SAResNet_Model import SAResnetModel
from Evaluater.Evaluater import Evaluater
# from Evaluater import Evaluater
import sys
curpath=os.path.abspath(os.path.dirname(__file__))
curpath=os.path.split(curpath)[0]
logpath=os.path.join(curpath,"log")

def loadModel(ckpt_path,kernal_channel=64, fc_num=32):
    model = SAResnetModel(kernal_channel=kernal_channel, fc_num=fc_num)
    model.load_weights(filepath=ckpt_path)
    model.trainable = True
    model.build((None, 1, 101, 4))  # when using subclass model
    return model

def evalute(model, testData_path):
    evaluater = Evaluater(model=model, data_path=testData_path)
    Accuary, Precision, Recall, F1, AUC = evaluater.test()
    return Accuary, Precision, Recall, F1, AUC


def paintAUC(x_list, y_list,x_label,y_label ,title):
    # plt.figure(figsize=(8, 8))  # 图的大小
    x = np.linspace(0, 1)
    y = x
    plt.plot(x, y, linestyle='dashed')  # 表示绘制红色、圆圈
    for i in range(len(x_list)):
        plt.scatter(x_list[i], y_list[i], marker='+', color='blue')
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    # plt.show()


def globalEvaluateSubData(model, subData_path):
    auc_list = []
    ave_accuary = 0
    ave_precision = 0
    ave_recall = 0
    ave_f1 = 0
    ave_auc = 0
    num = 0
    out_book = xlwt.Workbook()
    out_sheet = out_book.add_sheet('Sheet1')
    out_sheet.write(0, 0, "subDataName")
    out_sheet.write(0, 1, "Accuary")
    out_sheet.write(0, 2, "Precision")
    out_sheet.write(0, 3, "Recall")
    out_sheet.write(0, 4, "F1")
    out_sheet.write(0, 5, "AUC")
    out_book.save(os.path.join(logpath,"global","globalEvaluateSubData.xls"))
    for file in os.listdir(subData_path):
        print(str(num)+"]当前file："+file)
        num += 1
        now_file = os.path.join(subData_path, file)
        evaluater = Evaluater(model=model, data_path=now_file)
        Accuary, Precision, Recall, F1, AUC = evaluater.testModel()
        ave_accuary += Accuary
        ave_precision += Precision
        ave_recall += Recall
        ave_f1 += F1
        ave_auc += AUC
        auc_list.append(AUC)

        out_sheet.write(num, 0, file)
        out_sheet.write(num, 1, Accuary)
        out_sheet.write(num, 2, Precision)
        out_sheet.write(num, 3, Recall)
        out_sheet.write(num, 4, F1)
        out_sheet.write(num, 5, AUC)
        out_book.save(os.path.join(logpath,"global","globalEvaluateSubData.xls"))
    ave_accuary = ave_accuary / num
    ave_precision = ave_precision / num
    ave_recall = ave_recall / num
    ave_f1 = ave_f1 / num
    ave_auc = ave_auc / num
    #写入平均
    num+=1
    out_sheet.write(num, 0, "平均值")
    out_sheet.write(num, 1, ave_accuary)
    out_sheet.write(num, 2, ave_precision)
    out_sheet.write(num, 3, ave_recall)
    out_sheet.write(num, 4, ave_f1)
    out_sheet.write(num, 5, ave_auc)
    out_book.save(os.path.join(logpath,"global","globalEvaluateSubData.xls"))
    return ave_accuary, ave_precision, ave_recall, ave_f1, ave_auc, auc_list


def paintGlobal_Transfer_AUC(globalCkpt_path, subCkpt_path, subData_path, title):
    global_model = loadModel(globalCkpt_path, kernal_channel=64, fc_num=32)
    print("————————————预测global模型ing")
    ave_accuary, ave_precision, ave_recall, ave_f1, ave_auc, auc_list = globalEvaluateSubData(global_model,
                                                                                              subData_path)  # 全局模型预测子数据
    return
    # return
    subAUC_list = []
    ave_accuary = 0
    ave_precision = 0
    ave_recall = 0
    ave_f1 = 0
    ave_auc = 0
    num = 0

    out_book = xlwt.Workbook()
    out_sheet = out_book.add_sheet('Sheet1')
    out_sheet.write(0, 0, "TransferModelName")
    out_sheet.write(0, 1, "Accuary")
    out_sheet.write(0, 2, "Precision")
    out_sheet.write(0, 3, "Recall")
    out_sheet.write(0, 4, "F1")
    out_sheet.write(0, 5, "AUC")
    out_book.save(os.path.join(logpath,"transfer","transferEvaluateSubData.xls"))
    for file in os.listdir(subData_path):
        print(str(num) + "]当前file：" + file)
        now_ckpt = os.path.join(subCkpt_path, file, "SAResNet.ckpt")
        now_model = loadModel(now_ckpt, kernal_channel=128, fc_num=32)
        now_file = os.path.join(subData_path, file)
        sub_evaluater = Evaluater(now_model, now_file)
        Accuary, Precision, Recall, F1, AUC = sub_evaluater.testModel()
        ave_accuary += Accuary
        ave_precision += Precision
        ave_recall += Recall
        ave_f1 += F1
        ave_auc += AUC
        num += 1
        subAUC_list.append(AUC)

        out_sheet.write(num, 0, file)
        out_sheet.write(num, 1, Accuary)
        out_sheet.write(num, 2, Precision)
        out_sheet.write(num, 3, Recall)
        out_sheet.write(num, 4, F1)
        out_sheet.write(num, 5, AUC)
        out_book.save(os.path.join(logpath,"transfer","transferEvaluateSubData.xls"))
    ave_accuary = ave_accuary / num
    ave_precision = ave_precision / num
    ave_recall = ave_recall / num
    ave_f1 = ave_f1 / num
    ave_auc = ave_auc / num

    # 写入平均
    num += 1
    out_sheet.write(num, 0, "平均值")
    out_sheet.write(num, 1, ave_accuary)
    out_sheet.write(num, 2, ave_precision)
    out_sheet.write(num, 3, ave_recall)
    out_sheet.write(num, 4, ave_f1)
    out_sheet.write(num, 5, ave_auc)
    out_book.save(os.path.join(logpath,"transfer","transferEvaluateSubData.xls"))

    paintAUC(auc_list, subAUC_list, title)
    return ave_accuary, ave_precision, ave_recall, ave_f1, ave_auc, auc_list

def paintGlobal(globalauc_list,x_label,y_label,title):
    x_list=[i for i in range(1,151)]
    # plt.figure(figsize=(8, 8))  # 图的大小
    x = 1
    y = 0.5*x
    plt.plot(x, y, linestyle='dashed')  # 表示绘制红色、圆圈
    for i in range(len(x_list)):
        plt.scatter(x_list[i], globalauc_list[i], marker='+', color='blue')
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    # plt.show()

# def getTransferXls():
#     path=os.path.join(curpath,"save_weights")
#     for file in os.listdir(path):
#         now_ckpt_save_path=os.path.join(path,file)
#         xls_path = os.path.join(now_ckpt_save_path, "SAResNet" + "_batchSize=256_learnRate=5e-05.xls")
#         if (os.path.exists(xls_path)):
#             xls_path=xls_path
#         else:
#             xls_path = os.path.join(now_ckpt_save_path, "SAResNet" + "_batchSize=1024_learnRate=0.0001.xls")
#         workBook = xlrd.open_workbook(xls_path)
#         work_sheet = workBook.sheet_by_name("Sheet1")
#         m=0
#         for i in range(1,work_sheet.nrows):
#             t=work_sheet.cell_value(i,2)
#             m=max(t,m)
#         w2=xlwt.Workbook()
#
#         workBook2 = .open_workbook(os.path.join(logpath, "transfer", "transferEvaluateSubData.xls"))
#         work_sheet2=workBook2.sheet_by_name("Sheet1")
#         work_sheet2

if __name__=='__main__':
    # workBook1=xlrd.open_workbook(os.path.join(logpath,"global","globalEvaluateSubData.xls"))
    workBook_saresnet = xlrd.open_workbook(os.path.join(logpath,"transfer","transferEvaluateSubData.xls"))
    workBook_zeng = xlrd.open_workbook(r"D:\Python\PyCharm\workSpace\SAResNet_tensorflow2\Comparison_Model\CNN_Zeng\log\transfer\transferEvaluateSubData.xls")
    workBook_deeptf = xlrd.open_workbook(r"D:\Python\PyCharm\workSpace\SAResNet_tensorflow2\Comparison_Model\DeepTF\log\transfer\transferEvaluateSubData.xls")
    workBook_luo =xlrd.open_workbook(r"D:\Python\PyCharm\workSpace\SAResNet_tensorflow2\Comparison_Model\Exception_luo\log\transfer\transferEvaluateSubData.xls")

    sheet_saresnet=workBook_saresnet.sheet_by_name("Sheet1")
    sheet_zeng=workBook_zeng.sheet_by_name("Sheet1")
    sheet_deeptf= workBook_deeptf.sheet_by_name("Sheet1")
    sheet_luo = workBook_luo.sheet_by_name("Sheet1")

    col=5
    global_list=[]
    transfer_lsit=[]
    # print(work_sheet1.nrows-1)
    num=0
    num_saresnet=[0 for i in range(0,5)]
    num_zeng= [0 for i in range(0, 5)]
    num_deeptf = [0 for i in range(0, 5)]
    num_luo  = [0 for i in range(0, 5)]

    transfer_saresnet = []
    transfer_zeng = []
    transfer_deeptf = []
    transfer_luo= []

    for i in range(1,sheet_saresnet.nrows-1):
        t1=sheet_saresnet.cell_value(i,col)
        t2=sheet_zeng.cell_value(i,col)
        t3 = sheet_deeptf.cell_value(i, col)
        t4 = sheet_luo.cell_value(i, col)

        # global_list.append(t1)
        transfer_saresnet.append(t1)
        transfer_zeng.append(t2)
        transfer_deeptf.append(t3)
        transfer_luo.append(t4)

        # print(t1)
        # print(int(t1*10-5))

        num_saresnet[int(t1*10-5)]+=1
        num_zeng[int(t2 * 10 - 5)] += 1
        num_deeptf[int(t3 * 10 - 5)] += 1
        num_luo[int(t4 * 10 - 5)] += 1

        # if()
    print(num)
        # print(work_sheet1.cell_value(i,0))

    # paintAUC(global_list,transfer_lsit,"Global AUC","Transfer AUC","Global-Transfer AUC")
    # plt.figure(figsize=(16, 8))
    # plt.subplot(1, 2, 1)
    # paintGlobal(global_list,"150 subDateSet","Global AUC","(A)150 Global AUC")
    # plt.subplot(1, 2,2)
    # paintGlobal(transfer_lsit, "150 subDateSet", "Transfer AUC", "(B)150 Transfer AUC")
    # plt.show()
    #
    # plt.figure(figsize=(16, 8))
    # plt.subplot(1, 2, 1)
    # paintAUC(global_list,transfer_lsit,"Global AUC","Transfer AUC","(A)Global-Transfer Contrast")

    # plt.subplot(1, 2, 2)
    # labels = ['Global AUC', 'Transfer AUC']
    # plt.boxplot([global_list,transfer_lsit], labels=labels)
    # plt.title("(B)Global-Transfer AUC Boxplots")
    # plt.show()
    #
    # ##绘制多个条形图
    # a = ["Without self-attention module", "With self-attention module"]
    # b=["Accuracy","Precision","Recall","F1","AUC"]
    # b_14 = [0.740,0.776,0.717,0.745,0.762]
    # b_15 = [0.777,0.791,0.752,0.770,0.783]
    # bar_width = 0.2
    # x_14 = list(range(len(b)))
    # x_15 = [i + bar_width for i in x_14]
    #
    #
    # plt.bar(x_14, b_14, width=0.2, label=a[1])
    # plt.bar(x_15, b_15, width=0.2, label=a[0])
    #
    # ##设置x轴刻度
    # plt.xticks(x_15, b )
    # plt.legend(a, loc=0)
    # plt.title("Comparison of with or without self-attention module")
    # plt.show()

    print(np.average(transfer_luo))
    labels = ["SAResNet", "CNN_Zeng","DeepTF","Expectation_Luo"]
    # flierprops = dict(marker='-', markersize=8, markeredgewidth=2.5, linestyle='none', markeredgecolor='r')  # 异常值的样式
    plt.boxplot([transfer_saresnet,transfer_zeng,transfer_deeptf,transfer_luo], labels=labels)
    plt.title("Each Model's AUC Boxplots")
    plt.show()


    a = ["SAResNet", "CNN_Zeng","DeepTF","Expectation_Luo"]
    b = ["0.5-0.6", "0.6-0.7", "0.7-0.8", "0.8-0.9", "0.9-1.0"]
    bar_width = 0.2
    x_14 = list(range(len(b)))
    x_15 = [i + bar_width for i in x_14]
    x_16 = [i + 2*bar_width for i in x_14]
    x_17 = [i + 3*bar_width for i in x_14]


    plt.bar(x_14, num_saresnet, width=0.2, label="SAResNet Model AUC")
    plt.bar(x_15, num_zeng, width=0.2, label="CNN_Zeng Model AUC")
    plt.bar(x_16, num_deeptf, width=0.2, label="DeepTF Model AUC")
    plt.bar(x_17, num_luo, width=0.2, label="Expectation_Luo Model AUC")

    ##设置x轴刻度
    plt.xticks(x_15, b)
    plt.legend(a, loc = 0)
    plt.title("Comparison of each Model's AUC")
    plt.show()

