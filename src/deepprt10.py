import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import math

#심층 다층 퍼셈트론을 mnist 데이터셋에 훈련해보세요.
#98% 이상의 정확도를 얻을 수 있는지 확인해보세요.
from sklearn.datasets import fetch_openml
(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

X_valid = X_train[:5000]/255.0
X_train = X_train[5000:]/255.0
X_test = X_test/255.0
y_valid = y_train[:5000]
y_train = y_train[5000:]

class ExponentialLearningRate(keras.callbacks.Callback):
    def __init__(self,factor):
        self.factor=factor
        self.rates=[]
        self.losses=[]

    def on_batch_end(self,batch,logs):
        self.rates.append(keras.backend.get_value(self.model.optimizer.lr))
        self.losses.append(logs['loss'])
        keras.backend.set_value(self.model.optimizer.lr, self.model.optimizer.lr * self.factor)



model=keras.models.Sequential()
model.add(keras.layers.Flatten(input_shape=[28,28]))
model.add(keras.layers.Dense(300,activation='relu'))
model.add(keras.layers.Dense(100,activation='relu'))
model.add(keras.layers.Dense(10,activation='softmax'))

model.summary()

sgd = tf.keras.optimizers.SGD(learning_rate=0.001)
model.compile(loss='sparse_categorical_crossentropy',
              optimizer=sgd,
              metrics=['accuracy'])
expon_lr=ExponentialLearningRate(factor=1.005)
##텐서보드
# import os
# root_logdir=os.path.join(os.curdir,"my_logs")
#
# def get_run_logdir():
#     import time
#     run_id=time.strftime("run_%Y_%m_%d-%H_%M_%S")
#     return os.path.join(root_logdir,run_id)
#
# run_logdir=get_run_logdir()


checkpoint_cb=keras.callbacks.ModelCheckpoint(r"C:\Users\윤유진\OneDrive - 데이터마케팅코리아 (Datamarketingkorea)\문서\my_keras_model.h5",save_best_only=True)
early_stopping_cb=keras.callbacks.EarlyStopping(patience=10,restore_best_weights=True)
tensorboard_cb=keras.callbacks.TensorBoard(r'C:\Users\윤유진\OneDrive - 데이터마케팅코리아 (Datamarketingkorea)\문서\tensor_log')

history=model.fit(X_train,y_train,
                  # batch_size=10,(128,256,512)
                  epochs=1,
                  validation_data=(X_valid,y_valid),
                  callbacks=[expon_lr,
                             checkpoint_cb,
                             early_stopping_cb,
                             tensorboard_cb])


plt.plot(expon_lr.rates, expon_lr.losses)
plt.gca().set_xscale('log')
plt.hlines(min(expon_lr.losses), min(expon_lr.rates), max(expon_lr.rates))
plt.axis([min(expon_lr.rates), max(expon_lr.rates), 0, expon_lr.losses[0]])
plt.grid()
plt.xlabel("Learning rate")
plt.ylabel("Loss")


model.evaluate(X_test, y_test)

#이 장에서 사용한 방법을 사용해 최적의 학습률을 찾아보세요.
#(학습률을 지수적으로 증가시키면서 손실을 그래프로 그립니다. 그다음 손실이 다시 증가하는 지점을 찾습니다)


