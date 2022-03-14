from numpy.lib.function_base import gradient
import tensorflow.compat.v1 as tf
import pandas as pd
tf.disable_v2_behavior()

url = "https://raw.githubusercontent.com/Codecademy/datasets/master/streeteasy/manhattan.csv"
manhattan = pd.read_csv(url)  #맨해튼 주택 정보 csv파일 불러오기

x1 = manhattan[['size_sqft']] #면적
x2 = manhattan[['min_to_subway']] #지하철까지 걸리는 시간
y_data = manhattan[['rent']] #가격

w1 = tf.Variable(tf.random.uniform([1], 0, 1000, dtype = tf.float64, seed=0)) #임의로 초기값
w2 = tf.Variable(tf.random.uniform([1], 0, -10, dtype = tf.float64, seed=0))
b = tf.Variable(tf.random.uniform([1], 0, 100, dtype = tf.float64, seed=0))

y = w1*x1 + w2*x2 + b #가설함수

cost = tf.sqrt(tf.reduce_mean(tf.square(y-y_data))) #손실함수
alpha = 0.5

gradient_descent = tf.train.GradientDescentOptimizer(alpha).minimize(cost) #경사하강법

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for step in range(5001): #5000번 시행
        sess.run(gradient_descent)
        if step % 100 == 0: #100번 단위로 출력
            print("Epoch : %.f, cost = %.04f, w1 = %.4f, w2 = %.4f, b = %.4f" 
            % (step, sess.run(cost), sess.run(w1), sess.run(w2), sess.run(b))) 
            #학습, 손실, 가중치1, 가중치2, 편향 출력

