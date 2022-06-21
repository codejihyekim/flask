from pickletools import optimize
from turtle import shape
import pandas as pd
import numpy as np
import os
import sys
from icecream import ic
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from config import basedir
import tensorflow.compat.v1 as tf

class Solution:
    def __init__(self) -> None:
        self.basedir = os.path.join(basedir, 'model')
        self.df = None
        self.x_data = None
        self.y_data = None

    def create_print(self,avg_temp,min_temp,max_temp,rain_fall):
        print(f'훅에 전달된 avg_temp: {avg_temp}, min_temp: {min_temp}, max_temp: {max_temp}, rain_fall: {rain_fall}')
        result = f'{avg_temp}, {min_temp}, {max_temp}, {rain_fall}'
        return result

    def hook(self):
        self.create_model()

    def proprecessing(self):
        data_path = './data/price_data.csv'
        self.df = pd.read_csv(data_path, encoding='UTF-8', thousands=',')   
        #year,avg_temp,min_temp,max_temp,rain_fall,avgPrice
        xy = np.array(self.df, dtype=np.float32)
        #ic(xy)
        self.x_data = xy[:, 1:-1] #앞은 행, 뒤는 열로 구분하여 코딩 
        #ic(x_data)
        self.y_data = xy[:, [-1]]
        #ic(y_data)
    
    def create_model(self):
        #텐서모델 초기화(모델템플릿 생성)
        #학률변수 데이터 
        self.proprecessing()
        #선형식(가설)제작 y = Wx + b
        X = tf.placeholder(tf.float32, shape=[None, 4]) #외부에서 주입되는 값(x_data)
        Y = tf.placeholder(tf.float32, shape=[None, 1]) #외부에서 주입되는 값(y_data)
        W = tf.Variable(tf.random_normal([4, 1]), name='weight') #변하는 값 
        b = tf.Variable(tf.random_normal([1]), name='bias')
        hypothesis = tf.matmul(X,W) + b
        #손실함수
        cost = tf.reduce_mean(tf.square(hypothesis - Y))
        #최적화알고리즘
        optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.000005)
        train = optimizer.minimize(cost)
        #세션생성
        sess = tf.Session()
        sess.run(tf.global_variables_initializer())
        #트레이닝 
        for step in range(100000):
            cost_, hypo_, _= sess.run([cost, hypothesis, train],
                                        feed_dict={X: self.x_data, Y: self.y_data})
            if step % 500 == 0:
                print('# %d 손실비용: %d'%(step, cost_))
                print('- 배추가격: %d '%(hypo_[0]))
        #모델저장
        saver = tf.train.Saver() #저장
        saver.save(sess, os.path.join(self.basedir,'cabbage', 'cabbage.ckpt'),global_step=1000)
        print('저장완료')
    
    def load_model(self, avg_temp,min_temp,max_temp,rain_fall): #모델로드
        tf.disable_v2_behavior()
        X = tf.placeholder(tf.float32, shape=[None, 4]) #외부에서 주입되는 값(x_data)
        W = tf.Variable(tf.random_normal([4, 1]), name='weight') #변하는 값 
        b = tf.Variable(tf.random_normal([1]), name='bias')
        saver = tf.train.Saver()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            saver.restore(sess, os.path.join(self.basedir,'cabbage', 'cabbage.ckpt-1000'))
            data = [[avg_temp,min_temp,max_temp,rain_fall]]
            arr = np.array(data, dtype=np.float32)
            dict =sess.run(tf.matmul(X,W)+b, {X: arr[0:4]})
            print(dict)
        return int(dict[0])

    
if __name__ =='__main__':
    tf.disable_v2_behavior()
    s = Solution()
    s.hook()