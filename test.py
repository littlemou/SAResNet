import os
import sys
from random import sample
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from sklearn.metrics import roc_auc_score,roc_curve,auc,precision_recall_curve
# a=tf.constant([1,2,3,4],shape=[2,2])
# b=tf.constant([5,6,7,8],shape=[2,2])
# f = tf.matmul(a,b)#矩阵相乘
# print(f)
# N = tf.size(f, out_type=tf.int32)
# print(N)
# N = tf.cast(N, dtype=tf.int32)#数据类型转换为float
# print(N)
# f = f/N
# print(f)

# tf_path = os.path.join(os.getcwd(),"DataSet","tfrecords_30","wgEncodeAwgTfbsBroadH1hescCtcfUniPk","test.tfrecords")
# data=tf.data.TFRecordDataset(tf_path)
# num=0
# for raw_record in data:
#     num+=1
# print(num)
#
# data_path=r"D:\Python\PyCharm\workSpace\SAResNet_tensorflow2\DataSet\tfrecords_pretrain\150_global"
# train_filename = os.path.join(data_path,"train.data")
# # create_dirs(train_filename)
# train_transfile = os.path.join(data_path, "trans_data", 'train.data')
#
# train_data = open(train_filename, 'r')
# train_trans=open(train_transfile, 'w')
#
# temp=['>chr3:72127988-72128088_shuf CTGGTGCTATAGGTTGACTTTTCCAACTGGGAACATATCACATGTTTTTTCCTGGGTTTGGTGGTGAACAATGGGCACTGTCAAACCAAAACTGGGTAGTG 0\n', '>chr8:87905277-87905377 CTGGAATGAAGCCATTTTTTATTTTGAGGTTTCCTCTGATGACCACTAGCTGGCACTCTCAATCCAGAACAGCCCTGTCTTTTGGGATGGCAGAAAATGTG 1\n']
# train_trans.writelines(temp)
#
# train_trans.close()



t="[1 2 3] 1"
# print(digital("[1,2,3]"))

# res=t.strip('[')
# res=res.strip(']')
# res=res.split(' ')
# print(type(res))
# print(res)
# res=list(map(int, res))
# print(res)
# print(t[-1])
# train_trans=open(train_transfile, 'r')
# print(list(train_trans.readlines()))

def  digital(n):
    return list(map(int,str(n)))
def embed(seq, mapper):
    seqStr=""
    for element in seq:
        seqStr+=mapper[element]
    return seqStr
def dna2Sequence(dna):
    mapper = {'A':"1000", 'C': "0100", 'G':"0010",'T':"0001"}
    mat = embed(dna, mapper)
    return mat

# t=dna2Sequence("GGGTGGACAGGCGAGTGCGGGGTCTGCGCGCTCGAGACGAGCCGAGCCGCGGGGCGCGCCGAAGGAGTGCGGGATCTCTTGGCGGCCTGGGGACCCGCAGG")
#
# d=digital(t)
#
# print(d)
# a=np.array(d)
# print(a)


# a=tf.constant([i for i in range(1,6+1)])
# a=tf.reshape(a,[-1,2,3])
# print(a)
# b=tf.constant([i for i in range(1,6+1)])
# b=tf.reshape(b,[-1,2,3])
# print(b)
# c=tf.matmul(a,b)
# print(c)


# import tensorflow as tf
# import numpy as np
#
# # 加载数据集
# x_train = np.random.random((1000, 10))
# y_train = np.random.randint(0, 2, size=(1000,))
#
# # 创建数据集
# dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
# dataset=dataset.batch(batch_size=128)
# # 遍历数据集
# for features, label in dataset:
#     # 处理每个小块的数据
#     print(features.numpy().shape, label.numpy().shape)
#
# data_path = r"D:\Python\PyCharm\workSpace\SAResNet_tensorflow2\DataSet\data_pretrain\150_global\trans_data"
# n=os.path.join(data_path,"abc")
# print(n)

# output=[[0.3,0.8],[0.6,0.4]]
# output=tf.argmax(output,axis=1)
# output = tf.cast(output, dtype=tf.int32)
# print(output)
#
# a=1
# b=3
# print(a/b)


# def calAUC(prob,labels):
#   f = list(zip(prob,labels))
#   rank = [values2 for values1,values2 in sorted(f,key=lambda x:x[0])]
#   rankList = [i+1 for i in range(len(rank)) if rank[i]==1]
#   posNum = 0
#   negNum = 0
#   for i in range(len(labels)):
#     if(labels[i]==1):
#       posNum+=1
#     else:
#       negNum+=1
#   auc = 0
#   auc = (sum(rankList)- (posNum*(posNum+1))/2)/(posNum*negNum)
#   print(auc)
#   return auc
#
# label=[1,0,1,0]
# pre=[0.5,0.2,0.3,0.6]
# calAUC(pre,label)
#
#
# # a,b,c=precision_recall_curve(label,pre)
# # print(a,b,c)
# n=5
# while(n):
#     n+=1
#     print(n)

# y_2048=[0.6698,0.6968,0.7081,0.7159,0.7209,0.7266,0.7289,0.7308,0.7324,0.7332]
# y_1024=[0.6836,0.7064,0.7223,0.7218,0.7313,0.7008,0.7059,0.7113,0.7139,0.7168]
# y_4096=[0.6836,0.7064,0.7223,0.7313,0.7302,0.7454,0.7493,0.7522,0.7536,0.7564]
# x=[i for i in range(1,31,3)]
# # .49990625271537237 0.683719309065565 0.41930398609729913 0.27253019724337374 0.39478909357000874
# y2048_150auc=[0.37825497531695484, 0.34891135318438216, 0.3768065541677775, 0.3833195027368622, 0.34006990046532837, 0.38121124257284444, 0.40467598141428734, 0.33373320004086404, 0.4996279144193818, 0.35363639615922726, 0.36636078042328046, 0.3880170922032128, 0.39618287770346905, 0.37224538014164443, 0.38910257207898147, 0.40113769028509605, 0.3371783268015098, 0.3457784151719071, 0.39184957124644915, 0.5048900540664326, 0.3412075760705852, 0.3748027150759425, 0.3639736573259239, 0.4040765397361893, 0.3725667828106853, 0.41793905821438615, 0.35531604891815616, 0.3696091644204852, 0.3331817171946466, 0.49290188443037064, 0.36000036971601734, 0.3950050968399592, 0.3682062143971926, 0.35910731211088853, 0.433138259630935, 0.44774218916688857, 0.4369494648037314, 0.3579531039271455, 0.36150535892889624, 0.4351979535214493, 0.4193868091860805, 0.45620919132657833, 0.3982745924892195, 0.40705290897862145, 0.3491916638470223, 0.3704962999280931, 0.38610194251179353, 0.49677715433389863, 0.5306520098895153, 0.33618305657623515, 0.441390060686797, 0.3761163262250131, 0.45689173746879264, 0.40731664991143657, 0.44899573159172496, 0.42019457881526845, 0.4497153445405231, 0.45520391638130325, 0.49216528541268056, 0.3460231991432817, 0.3849174933405809, 0.43676855728508496, 0.4933510090405274, 0.4490819610070036, 0.3780909990108804, 0.3715121624321568, 0.3935010546196094, 0.4025078549696547, 0.3578044218721489, 0.43993865473547433, 0.3457174707593771, 0.36503895754970445, 0.45385202963379395, 0.3394085560840475, 0.32601777918016717, 0.3649197976758835, 0.39697337821070067, 0.5414240863631992, 0.33781555812640773, 0.37344506538280897, 0.38012790499777205, 0.39947403753414995, 0.35839045076589326, 0.340197055118203, 0.3372397837899572, 0.38968807424094476, 0.3905696132847394, 0.373879694225766, 0.365325293463564, 0.3918163021918036, 0.41829597414531744, 0.3728647396179467, 0.3392650149291442, 0.3307271802695143, 0.3798795360075119, 0.3890782097821486, 0.38873205785529963, 0.3958964024481265, 0.3611162373467417, 0.4375831507558159, 0.34127100340447336, 0.5400782263576287, 0.39102684942498317, 0.3202776315848661, 0.5279117153258311, 0.5101489617180499, 0.385298474780267, 0.5089945204931281, 0.3371968209195129, 0.35319652185750505, 0.5289967760758907, 0.3289501853746548, 0.3208181785281291, 0.28716576774036434, 0.2846278425424771, 0.5140713192641506, 0.519263995874317, 0.333603364497742, 0.30578485021490204, 0.3104379142514736, 0.30593418418197693, 0.3110074944526846, 0.3087397766535944, 0.5182630529929981, 0.31444538766270513, 0.4886529709062012, 0.31051537752916236, 0.4085814354440537, 0.378556801994302, 0.3328417931680063, 0.517598194975992, 0.31327920985736235, 0.3199061027603324, 0.5205496681840279, 0.34010785068982424, 0.3397017707362535, 0.3126298581448083, 0.5220033474163266, 0.5334128125672429, 0.4109677265462872, 0.3199961059190032, 0.5056851106680484, 0.5032394981601871, 0.2971730395619922, 0.4053012858014165, 0.3201684440068968, 0.5122520823029011, 0.3240543041172912, 0.5189247967625527, 0.3448113639727389]
# def paintScatter(x_list,y_list,title):
#     plt.figure(figsize=(8, 8))  # 图的大小
#     # x = np.linspace(0,1)
#     # y=np.linspace(0,1)
#     # plt.plot(x,y, linestyle='dashed')  # 表示绘制红色、圆圈
#     for i in range(len(x_list)):
#         plt.scatter(x_list[i],y_list[i],marker='+',color='blue')
#     plt.title(title)
#     plt.show()
#
# def paintPlot(x_list,y_list,title):
#     plt.figure(figsize=(8, 8))  # 图的大小
#     # x = np.linspace(0,1)
#     # y=np.linspace(0,1)
#     plt.plot(x_list,y_list)  # 表示绘制红色、圆圈
#     plt.title(title)
#     plt.show()
#
# paintPlot(x,y_2048,"Accuracy(batchSize=2048)")
# paintPlot(x,y_1024,"Accuracy(batchSize=1024)")
# paintPlot(x,y_4096,"Accuracy(batchSize=4096)")
# num=[x for x in y2048_150auc if(x>0.5)]
# print(len(num))
# x_150auc=[i for i in range(1,151)]
# paintScatter(x_150auc,y2048_150auc,"AUC(batchSize=2048)")
#
# label=[1,1,1,1,1,1]
# prob=[0,0,0,0,0,0]


# curPath = os.path.abspath(os.path.dirname(__file__))
# curPath=os.path.split(curPath)[0]
# xlsSave_path = os.path.join(curPath, "log")
# xlsSave_path = os.path.join(xlsSave_path,"SAResNet"+"_batchSize="+str(0)+"_learnRate="+str(0)+".xls")
# # xlsSave_path + "SAResNet" + "_batchSize=" + str(0) + "_learnRate=" + str(
# #     0) + ".xls"
# print(xlsSave_path)

# from Trainer.util_train import second_train
# second_train(0.7,64,32,128)

# tf.keras.layers.LSTM(
#     units,
#     activation="tanh",
#     use_bias=True,
#     dropout=0.0,
#     return_sequences=False,
# )
# from tensorflow import keras
# from tensorflow.keras import layers
# rnn_units = 128
# vocab_size = 65
#
# # 构造输入，输入维度为句子的维度
# _input = keras.Input(shape=(64,10))
#
# # 构造LSTM单元，rnn_units 为h_i的输出维度
# # LSTM单元会自动匹配句子的长度64，生成64个单元A，也会得到输出h_1~h_64
# # return_sequences=True表示返回所有的h_i
# x = layers.LSTM(rnn_units,return_sequences=True, recurrent_initializer='orthogonal')(_input)
#
# # 在来一层LSTM，每个h_i的输出维度为90
# x = layers.LSTM(90,return_sequences=True,recurrent_initializer='orthogonal')(x)
#
# # 进入全连接层，vocab_size表示所有的词汇量的大小,每个timestep共享同一组参数。
# output = layers.Dense(vocab_size,activation='softmax')(x)

from data_loader.load_data import *
curPath = os.path.abspath(os.path.dirname(__file__))
sys.path.append(os.path.split(curPath)[0])
rootPath= os.path.split(curPath)[0]
# rootPath=rootPath.split("Comparison_Model")[0]
dataSet_path = os.path.join(rootPath,"DataSet")
# print(rootPath.split("Comparison_Model")[0])
data_path= r"D:\Python\PyCharm\workSpace\SAResNet_tensorflow2\DataSet\data_pretrain\150_global\trans_data"
(train, train_label), (test, test_label), (valid, valid_label) = load_data(data_path)
# sentence=[["我好饿"],["想吃饭"]]
print(type(valid))
print(len(train))
