import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import keras
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

dataset=pd.read_csv("santander-customer-satisfaction/train.csv")

# independent variables (matrix of features)
x=dataset.drop(labels=['ID', 'TARGET'], axis=1)
# axis=1 for selecting columns
y=dataset['TARGET']

x_train, x_test, y_train, y_test= train_test_split(x,y,test_size=0.2,
                                                   random_state=0)
# Removing constant, quasi constant and duplicate features
# constant column when all values in a column are same so no effect on output
# quasi constant column when most of the values in a column are same except few
# so no effect on output
# duplicate when values of 2 columns are identical

from sklearn.feature_selection import VarianceThreshold
re_f= VarianceThreshold(threshold=0.01)
# removes features with variance less than 1%
x_train=re_f.fit_transform(x_train)
x_test=re_f.transform(x_test)
# earlier columns in x= 369, now=266 after removing constant and quasi constant

# remove duplicate features
x_train_t= x_train.T # applying transpose method
x_test_t=x_test.T
x_train_t=pd.DataFrame(x_train_t)
x_test_t=pd.DataFrame(x_test_t)

# number of duplicate features
duplicated_features=x_train_t.duplicated()
# print(duplicated_features) false means not duplicated
features_to_keep=[not index for index in duplicated_features]

x_train=x_train_t[features_to_keep].T
x_test=x_test_t[features_to_keep].T
# earlier columns in x= 266, now=250 after removing duplicate columns

sc=StandardScaler()
x_train=sc.fit_transform(x_train)
x_test=sc.transform(x_test)

# x_train.shape= (60816, 250), x_test.shape= (15204, 250)
# reshaping the dataset
x_train=x_train.reshape(60816, 250, 1)
x_test=x_test.reshape(15204, 250,1)
# x_train.shape= (60816, 250,1), x_test.shape= (15204, 250,1)

y_train=y_train.to_numpy()
y_test=y_test.to_numpy()

model=tf.keras.models.Sequential()
model.add(tf.keras.layers.Conv1D(filters=32, kernel_size=3, activation='relu',
                                 input_shape= (250,1)))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.MaxPool1D(pool_size=2))
model.add(tf.keras.layers.Dropout(0.3))

model.add(tf.keras.layers.Conv1D(filters=64, kernel_size=3, activation='relu'))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.MaxPool1D(pool_size=2))
model.add(tf.keras.layers.Dropout(0.5))

model.add(tf.keras.layers.Conv1D(filters=128,kernel_size=3, activation='relu'))
model.add(tf.keras.layers.BatchNormalization())
model.add(tf.keras.layers.MaxPool1D(pool_size=2))
model.add(tf.keras.layers.Dropout(0.5))

model.add(tf.keras.layers.Flatten())

model.add(tf.keras.layers.Dense(units=256, activation='relu'))
model.add(tf.keras.layers.Dropout(0.5))
model.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))

opt= tf.keras.optimizers.Adam(learning_rate=0.00005)
model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['accuracy'])

history= model.fit(x_train,y_train,epochs=10,validation_data=(x_test, y_test))

y_pred=(model.predict(x_test) > 0.5).astype ('int32')
print(y_pred[0], y_test[0])
print(y_pred[10], y_test[10])
print(y_pred[-2], y_test[-2])

cm=confusion_matrix(y_test, y_pred)
print(cm)
acc_cm=accuracy_score(y_test, y_pred)
print(acc_cm)

epoch_range=range(1,11)
plt.plot(epoch_range, history.history['accuracy'])
plt.plot(epoch_range, history.history['val_accuracy'])
plt.title("Model Accuracy")
plt.ylabel("Accuracy")
plt.xlabel("Epoch")
plt.legend(["Train", 'val'], loc='upper left')
plt.show()

plt.plot(epoch_range, history.history['loss'])
plt.plot(epoch_range, history.history['val_loss'])
plt.title("Model Loss")
plt.ylabel("Loss")
plt.xlabel("Epoch")
plt.legend(["Train", 'val'], loc='upper left')
plt.show()
