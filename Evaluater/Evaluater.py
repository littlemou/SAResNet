import os
import tensorflow as tf
from data_loader.load_data import load_data_evaluate,parse
from models.SAResNet_Model import SAResnetModel
from tensorflow import keras
from sklearn.metrics import roc_auc_score,roc_curve,auc,precision_recall_curve
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
class Evaluater():
    def __init__(self, model, data_path,batch_size=5000):
        self.model=model
        (self.test, self.test_label)= load_data_evaluate(data_path)
        self.test_num = len(self.test)
        self.batch_size = batch_size
        test_dataset = tf.data.Dataset.from_tensor_slices((self.test, self.test_label))
        self.test_dataset = test_dataset.map(parse, num_parallel_calls=tf.data.experimental.AUTOTUNE).batch(batch_size=self.batch_size)#一次性全部

    def test_step(self, inputs, labels):
        output = self.model(inputs, training=False)  # 前向计算
        labels = tf.argmax(labels, axis=1)
        labels = tf.cast(labels, dtype=tf.int32)
        output=tf.argmax(output,axis=1)
        output = tf.cast(output, dtype=tf.int32)

        auc=roc_auc_score(labels,output)
        acc = accuracy_score(labels, output)
        precision = precision_score(labels, output)
        recall = recall_score(labels, output)
        f1 = f1_score(labels, output)

        return acc,precision,recall,f1,auc
    def testModel(self):
        Accuary=0
        Precision=0
        Recall=0
        F1=0
        AUC=0
        num=0
        for step, (inputs, labels) in enumerate(self.test_dataset):
            # print("————————————预测步骤")
            num+=1
            acc, precision, recall, f1,auc=self.test_step(inputs, labels)
            AUC += auc
            Accuary+=acc
            Precision+=precision
            Recall+=recall
            F1+=f1
        Accuary = Accuary/num
        Precision = Precision/num
        Recall = Recall/num
        F1 = F1/num
        AUC = AUC/num
        return Accuary, Precision, Recall, F1 ,AUC