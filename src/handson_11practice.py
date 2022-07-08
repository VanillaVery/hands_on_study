import tensorflow as tf
from tensorflow import keras
#%%
keras.layers.Dense(10,acrtivation="relu",kernel_initializer="he_normal")
#%%
he_avg_init=keras.initializers.VarianceScaling(scale=2.,mode='fan_avg',distribution='uniform')
keras.layers.Dense(10,activation='sigmoid',kernel_initializer=he_avg_init)
#%%
model=keras.models.Sequential([
    keras.layers.Flatten(input_shape=[28,28]),
    keras.layers.BatchNormalization(),
    keras.layers.Dense(300,activation='elu',kernel_initializer="he_normal"),
    keras.layers.BatchNormalization(),
    keras.layers.Dense(100,activation='elu',kernel_initializer="he_normal"),
    keras.layers.BatchNormalization(),
    keras.layers.Dense(10,activation='softmax')
])

#%%
[(var.name,var.trainable) for var in model.layers[1].variables]
#%%
model=keras.models.Sequential([
    keras.layers.Flatten(input_shape=[28,28]),
    keras.layers.BatchNormalization(),
    keras.layers.Dense(300,kernel_initializer="he_normal",use_bias=False),
    keras.layers.BatchNormalization(),
    keras.layers.Activation("elu"),
    keras.layers.Dense(100,kernel_initializer="he_normal",use_bias=False),
    keras.layers.BatchNormalization(),
    keras.layers.Activation("elu"),
    keras.layers.Dense(10,activation="softmax")
])
#%%
optimizer=keras.optimizers.SGD(clipvalue=1.0)
model.compile(loss="mse",optimizer=optimizer)
#%%전이학습
model_A=keras.models.load_model("my_model_A.h5")
model_B_on_A=keras.models.Sequential(model_A.layers[:-1])
#출력층 제외하고 모든 층 재사용
model_B_on_A.add(keras.layers.Dense(1,activation="sigmoid"))
#%%
model_A_clone=keras.models.clone_model(model_A)
model_A_clone.set_weights(model_A.get_weights())
#%%
for layer in model_B_on_A.layers[:-1]:
    layer.trainable = False

model_B_on_A.compile(loss="binary_crossentropy",optimizer="sgd",
                    metrics=["accuracy"])
#%%
history=model_B_on_A.fit(X_train_B, y_train_B,epochs=4,
                        validation_data=(X_valid_B,y_valid_B))

for layer in model_B_on_A.layers[:-1]:
    layer.trainable = True #동결 해제

optimizer=keras.optimizers.SGD(lr=1e-4) #기본 학습률은 1e-2
model_B_on_A.compile(loss="binary_crossentropy",optimizer=optimizer,
                    metrics=["accuracy"]) #다시 컴파일 
history=model_B_on_A.fit(X_train_B,y_train_B,epochs=16,
                        validation_data=(X_valid_B,y_valid_B)) 
#%%
#규제
layer=keras.layers.Dense(100, activation="elu",
                        kernel_initializer="he_normal",
                        kernel_regularizer=keras.regularizers.l2(0.01))
#%%
from functools import partial 

RegularizedDense = partial(keras.layers.Dense,
                            activation="elu",
                            kernel_initializer="he_normal",
                            kenel_regularizer=keras.regularizers.l2(0.01))

model=keras.models.Sequential([
    keras.layers.Flatten(input_shape=[28,28]),
    RegularizedDense(300),
    RegularizedDense(100),
    RegularizedDense(10,activation="softmax",
                    kernel_initializer="glorot_uniform")
])
#%%
#드롭아웃
model=keras.models.Sequential([
    keras.layers.Flatten(input_shape=[28,28]),
    keras.layers.Dropout(rate=0.2),
    keras.layers.Dense(300,activation="elu",kernel_initializer="he_normal"),
    keras.layers.Dropout(rate=0.2),
    keras.layers.Dense(100,activation="elu",kernel_initializer="he_normal"),
    keras.layers.Dropout(rate=0.2),
    keras.layers.Dense(10,activation="softmax")
])