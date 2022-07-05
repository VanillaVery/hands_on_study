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