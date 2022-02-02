from keras.datasets import cifar10
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import math
#CIFAR10 이미지 데이터셋에 심층 신경망을 훈련해보세요.

#The CIFAR-10 dataset consists of 60000 32x32 colour images in 10 classes, with 6000 images per class. There are 50000 training images and 10000 test images.
import keras

(X_train_full, y_train_full), (X_test, y_test) = cifar10.load_data()

X_train = X_train_full[5000:]
y_train = y_train_full[5000:]
X_valid = X_train_full[:5000]
y_valid = y_train_full[:5000]

# plt.subplot(141)
# plt.imshow(X_train[0], interpolation="bicubic")
# plt.grid(False)
# plt.subplot(142)
# plt.imshow(X_train[4], interpolation="bicubic")
# plt.grid(False)
# plt.subplot(143)
# plt.imshow(X_train[8], interpolation="bicubic")
# plt.grid(False)
# plt.subplot(144)
# plt.imshow(X_train[12], interpolation="bicubic")
# plt.grid(False)
# plt.show()


#a. 100개의 뉴런을 가진 은닉층 20개로 심층 신경망을 만들어보세요. He 초기화와 ELU 활성화 함수를 사용하세요.
model=keras.models.Sequential()
model.add(keras.layers.Flatten(input_shape=[32,32,3]))
model.add(keras.layers.Dense(100,activation='elu',kernel_initializer='he_normal'))
model.add(keras.layers.Dense(100,activation='elu',kernel_initializer='he_normal'))
model.add(keras.layers.Dense(100,activation='elu',kernel_initializer='he_normal'))
model.add(keras.layers.Dense(100,activation='elu',kernel_initializer='he_normal'))
model.add(keras.layers.Dense(100,activation='elu',kernel_initializer='he_normal'))
model.add(keras.layers.Dense(100,activation='elu',kernel_initializer='he_normal'))
model.add(keras.layers.Dense(100,activation='elu',kernel_initializer='he_normal'))
model.add(keras.layers.Dense(100,activation='elu',kernel_initializer='he_normal'))
model.add(keras.layers.Dense(100,activation='elu',kernel_initializer='he_normal'))
model.add(keras.layers.Dense(100,activation='elu',kernel_initializer='he_normal'))
model.add(keras.layers.Dense(100,activation='elu',kernel_initializer='he_normal'))
model.add(keras.layers.Dense(100,activation='elu',kernel_initializer='he_normal'))
model.add(keras.layers.Dense(100,activation='elu',kernel_initializer='he_normal'))
model.add(keras.layers.Dense(100,activation='elu',kernel_initializer='he_normal'))
model.add(keras.layers.Dense(100,activation='elu',kernel_initializer='he_normal'))
model.add(keras.layers.Dense(100,activation='elu',kernel_initializer='he_normal'))
model.add(keras.layers.Dense(100,activation='elu',kernel_initializer='he_normal'))
model.add(keras.layers.Dense(100,activation='elu',kernel_initializer='he_normal'))
model.add(keras.layers.Dense(100,activation='elu',kernel_initializer='he_normal'))
model.add(keras.layers.Dense(100,activation='elu',kernel_initializer='he_normal'))

#b문제: Nadam 옵티마이저와 조기 종료를 사용하여 CIFAR10 데이터셋에 이 네트워크를 훈련하세요.
# keras.datasets.cifar10.load_ data()를 사용하여 데이터를 적재할 수 있습니다.
# 이 데이터셋은 10개의 클래스와 32×32 크기의 컬러 이미지 60,000개로 구성됩니다(50,000개는 훈련, 10,000개는 테스트).
# 따라서 10개의 뉴런과 소프트맥스 활성화 함수를 사용하는 출력층이 필요합니다.
# 모델 구조와 하이퍼파라미터를 바꿀 때마다 적절한 학습률을 찾아야 한다는 것을 기억하세요.

model.add(keras.layers.Dense(10, activation="softmax"))

#학습률 튜닝
# [1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2]: 이중에 accuracy와 학습곡선이 제일 나은걸 찾아볼 것
##궁금한 점: 컴파일과 피팅을 반복하면 모델 객체에 계속 쌓이는 거 아님.,..? 어떻게 초기화를 해야 하나
for i,x in enumerate([1e-5, 3e-5, 1e-4, 3e-4, 1e-3, 3e-3, 1e-2]):
    globals()['model_{}'.format(i)]=model
    globals()['model_{}'.format(i)].compile(loss='sparse_categorical_crossentropy',
                  optimizer=tf.keras.optimizers.Nadam(learning_rate=x),
                  metrics=['accuracy'])

    history=globals()['model_{}'.format(i)].fit(X_train,y_train,epochs=10,validation_data=(X_valid, y_valid))
    # Plotting
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    plt.figure(figsize=(8, 8))
    plt.subplot(2, 1, 1)
    plt.plot(acc, label='Training Accuracy')
    plt.plot(val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.ylabel('Accuracy')
    plt.ylim([min(plt.ylim()),0.5])#잘 안보여서 고침
    plt.title('Training and Validation Accuracy_{}'.format(i))

    plt.subplot(2, 1, 2)
    plt.plot(loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.ylabel('Cross Entropy')
    # plt.ylim([0,1.0])
    plt.title('Training and Validation Loss_{}'.format(i))
    plt.xlabel('epoch')
    plt.show()

#3e-4로 했다고 하자
model.compile(loss='sparse_categorical_crossentropy',
              optimizer=tf.keras.optimizers.Nadam(learning_rate=0.0003),
              metrics=['accuracy'])

checkpoint_cb=keras.callbacks.ModelCheckpoint(
    r"C:\Users\윤유진\OneDrive - 데이터마케팅코리아 (Datamarketingkorea)\바탕 화면\공부\model\my11modelb.h5",
    save_best_only=True)
early_stopping_cb=keras.callbacks.EarlyStopping(patience=10,restore_best_weights=True)
callbacks=[checkpoint_cb,early_stopping_cb]

history=model.fit(X_train,y_train,epochs=100,validation_data=(X_valid, y_valid),callbacks=callbacks)

model = keras.models.load_model(
    r"C:\Users\윤유진\OneDrive - 데이터마케팅코리아 (Datamarketingkorea)\바탕 화면\공부\model\my11modelb.h5")
model.evaluate(X_valid, y_valid)
#시간도 재야 함

#c.문제: 배치 정규화를 추가하고 학습 곡선을 비교해보세요.
# 이전보다 빠르게 수렴하나요? 더 좋은 모델이 만들어지나요? 훈련 속도에는 어떤 영향을 미치나요?
#keras.backend.clear_session() #뭔가 clear하는것 같아서 해봄-> 안된대

modelc=keras.models.Sequential()
modelc.add(keras.layers.Flatten(input_shape=[32,32,3]))
modelc.add(keras.layers.BatchNormalization())
for _ in range(20):
    modelc.add(keras.layers.Dense(100, kernel_initializer="he_normal"))
    modelc.add(keras.layers.BatchNormalization())
    modelc.add(keras.layers.Activation("elu"))
modelc.add(keras.layers.Dense(10, activation="softmax"))

modelc.compile(loss='sparse_categorical_crossentropy',
              optimizer=tf.keras.optimizers.Nadam(learning_rate=0.0003),
              metrics=['accuracy'])

checkpoint_cb=keras.callbacks.ModelCheckpoint(
    r"C:\Users\윤유진\OneDrive - 데이터마케팅코리아 (Datamarketingkorea)\바탕 화면\공부\model\my11modelc.h5",
    save_best_only=True)
early_stopping_cb=keras.callbacks.EarlyStopping(patience=10,restore_best_weights=True)
callbacks=[checkpoint_cb,early_stopping_cb]

history=modelc.fit(X_train,y_train,epochs=100,validation_data=(X_valid, y_valid),callbacks=callbacks)

model = keras.models.load_model(
    r"C:\Users\윤유진\OneDrive - 데이터마케팅코리아 (Datamarketingkorea)\바탕 화면\공부\model\my11modelc.h5")
model.evaluate(X_valid, y_valid)


#d. 문제: 배치 정규화를 SELU로 바꾸어보세요. 네트워크가 자기 정규화하기 위해 필요한 변경 사항을 적용해보세요
# (즉, 입력 특성 표준화, 르쿤 정규분포 초기화, 완전 연결 층만 순차적으로 쌓은 심층 신경망 등).

#학습률 체크
# 1e-5, 3e-5, 5e-5, 1e-4, 3e-4, 5e-4, 1e-3, 3e-3
keras.backend.clear_session()

modeld=keras.models.Sequential()
modeld.add(keras.layers.Flatten(input_shape=[32,32,3]))
for _ in range(20):
    modeld.add(keras.layers.Dense(100, kernel_initializer="lecun_normal"))
    modeld.add(keras.layers.Activation("selu"))
modeld.add(keras.layers.Dense(10, activation="softmax"))

modeld.compile(loss='sparse_categorical_crossentropy',
              optimizer=tf.keras.optimizers.Nadam(learning_rate=0.0003),####
              metrics=['accuracy'])

X_means = X_train.mean(axis=0)
X_stds = X_train.std(axis=0)
X_train_scaled = (X_train - X_means) / X_stds
X_valid_scaled = (X_valid - X_means) / X_stds
X_test_scaled = (X_test - X_means) / X_stds

checkpoint_cb=keras.callbacks.ModelCheckpoint(
    r"C:\Users\윤유진\OneDrive - 데이터마케팅코리아 (Datamarketingkorea)\바탕 화면\공부\model\my11modeld.h5",
    save_best_only=True)
early_stopping_cb=keras.callbacks.EarlyStopping(patience=10,restore_best_weights=True)
callbacks=[checkpoint_cb,early_stopping_cb]

history=modeld.fit(X_train,y_train,epochs=100,validation_data=(X_valid, y_valid),callbacks=callbacks)

## 정확도 , 시간 check

#e. 문제: 알파 드롭아웃으로 모델에 규제를 적용해보세요.
# 그다음 모델을 다시 훈련하지 않고 MC 드롭아웃으로 더 높은 정확도를 얻을 수 있는지 확인해보세요.
keras.backend.clear_session()

modele=keras.models.Sequential()
modele.add(keras.layers.Flatten(input_shape=[32,32,3]))
for _ in range(20):
    modele.add(keras.layers.Dense(100, kernel_initializer="lecun_normal"))
    modele.add(keras.layers.Activation("selu"))
modele.add(keras.layers.AlphaDropout(rate=0.1))
#많은 신경망 구조는 마지막 은닉층 뒤에만 드롭아웃 지정
#드롭아웃을 전체에 적용하는 것이 너무 강하다면 이렇게 시도해보자,

# 드롭아웃 비율 [0.05,0.1,0.2,0.4] 학습률 [1e-4, 3e-4, 5e-4, 1e-3] 그리드 서치 필요
modele.add(keras.layers.Dense(10, activation="softmax"))

modele.compile(loss='sparse_categorical_crossentropy',
              optimizer=tf.keras.optimizers.Nadam(learning_rate=3e-5),####
              metrics=['accuracy'])

X_means = X_train.mean(axis=0)
X_stds = X_train.std(axis=0)
X_train_scaled = (X_train - X_means) / X_stds
X_valid_scaled = (X_valid - X_means) / X_stds
X_test_scaled = (X_test - X_means) / X_stds

checkpoint_cb=keras.callbacks.ModelCheckpoint(
    r"C:\Users\윤유진\OneDrive - 데이터마케팅코리아 (Datamarketingkorea)\바탕 화면\공부\model\my11modele.h5",
    save_best_only=True)
early_stopping_cb=keras.callbacks.EarlyStopping(patience=10,restore_best_weights=True)
callbacks=[checkpoint_cb,early_stopping_cb]

history=modele.fit(X_train,y_train,epochs=100,validation_data=(X_valid, y_valid),callbacks=callbacks)

#MC드롭아웃
class MCAlphaDropout(keras.layers.AlphaDropout):
    def call(self, inputs):
        return super().call(inputs, training=True)

## 여기를 잘 모르겠다
mc_model = keras.models.Sequential([
    MCAlphaDropout(layer.rate) if isinstance(layer, keras.layers.AlphaDropout) else layer
    for layer in model.layers
])

#여기도....
def mc_dropout_predict_probas(mc_model, X, n_samples=10):
    Y_probas = [mc_model.predict(X) for sample in range(n_samples)]
    return np.mean(Y_probas, axis=0)

def mc_dropout_predict_classes(mc_model, X, n_samples=10):
    Y_probas = mc_dropout_predict_probas(mc_model, X, n_samples)
    return np.argmax(Y_probas, axis=1)

y_pred = mc_dropout_predict_classes(mc_model, X_valid_scaled)
accuracy = np.mean(y_pred == y_valid[:, 0])
accuracy

#f. 문제: 1사이클 스케줄링으로 모델을 다시 훈련하고 훈련 속도와 모델 정확도가 향상되는지 확인해보세요.
keras.backend.clear_session()

#1사이클 스케줄링 코드
K = keras.backend

class ExponentialLearningRate(keras.callbacks.Callback):
    def __init__(self, factor):
        self.factor = factor
        self.rates = []
        self.losses = []
    def on_batch_end(self, batch, logs):
        self.rates.append(K.get_value(self.model.optimizer.lr))
        self.losses.append(logs["loss"])
        K.set_value(self.model.optimizer.lr, self.model.optimizer.lr * self.factor)

def find_learning_rate(model, X, y, epochs=1, batch_size=32, min_rate=10**-5, max_rate=10):
    init_weights = model.get_weights()
    iterations = math.ceil(len(X) / batch_size) * epochs
    factor = np.exp(np.log(max_rate / min_rate) / iterations)
    init_lr = K.get_value(model.optimizer.lr)
    K.set_value(model.optimizer.lr, min_rate)
    exp_lr = ExponentialLearningRate(factor)
    history = model.fit(X, y, epochs=epochs, batch_size=batch_size,
                        callbacks=[exp_lr])
    K.set_value(model.optimizer.lr, init_lr)
    model.set_weights(init_weights)
    return exp_lr.rates, exp_lr.losses

def plot_lr_vs_loss(rates, losses):
    plt.plot(rates, losses)
    plt.gca().set_xscale('log')
    plt.hlines(min(losses), min(rates), max(rates))
    plt.axis([min(rates), max(rates), min(losses), (losses[0] + min(losses)) / 2])
    plt.xlabel("Learning rate")
    plt.ylabel("Loss")

class OneCycleScheduler(keras.callbacks.Callback):
    def __init__(self, iterations, max_rate, start_rate=None,
                 last_iterations=None, last_rate=None):
        self.iterations = iterations
        self.max_rate = max_rate
        self.start_rate = start_rate or max_rate / 10
        self.last_iterations = last_iterations or iterations // 10 + 1
        self.half_iteration = (iterations - self.last_iterations) // 2
        self.last_rate = last_rate or self.start_rate / 1000
        self.iteration = 0
    def _interpolate(self, iter1, iter2, rate1, rate2):
        return ((rate2 - rate1) * (self.iteration - iter1)
                / (iter2 - iter1) + rate1)
    def on_batch_begin(self, batch, logs):
        if self.iteration < self.half_iteration:
            rate = self._interpolate(0, self.half_iteration, self.start_rate, self.max_rate)
        elif self.iteration < 2 * self.half_iteration:
            rate = self._interpolate(self.half_iteration, 2 * self.half_iteration,
                                     self.max_rate, self.start_rate)
        else:
            rate = self._interpolate(2 * self.half_iteration, self.iterations,
                                     self.start_rate, self.last_rate)
        self.iteration += 1
        K.set_value(self.model.optimizer.lr, rate)


batch_size = 128
rates, losses = find_learning_rate(model, X_train_scaled, y_train, epochs=1, batch_size=batch_size)
plot_lr_vs_loss(rates, losses)
plt.axis([min(rates), max(rates), min(losses), (losses[0] + min(losses)) / 1.4])


#test끝
keras.backend.clear_session()


model = keras.models.Sequential()
model.add(keras.layers.Flatten(input_shape=[32, 32, 3]))
for _ in range(20):
    model.add(keras.layers.Dense(100,
                                 kernel_initializer="lecun_normal",
                                 activation="selu"))

model.add(keras.layers.AlphaDropout(rate=0.1))
model.add(keras.layers.Dense(10, activation="softmax"))

optimizer = keras.optimizers.SGD(learning_rate=1e-2)
model.compile(loss="sparse_categorical_crossentropy",
              optimizer=optimizer,
              metrics=["accuracy"])

n_epochs = 20
onecycle = OneCycleScheduler(len(X_train_scaled) // batch_size * n_epochs, max_rate=0.05)
history = model.fit(X_train_scaled, y_train, epochs=n_epochs, batch_size=batch_size,
                    validation_data=(X_valid_scaled, y_valid),
                    callbacks=[onecycle])

