import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense,Flatten
from tensorflow.keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt

#using mnist dataset -> large collection of handwritten digits 0-9
(x_train,y_train),(x_test,y_test) = mnist.load_data()

# X train shape (60000,28,28) Y train (60000,)
# X test shape (10000,28,28) Y test (10000,)
x_train = x_train /255.0
y_train = y_train / 255.0


model= Sequential()
model.add(Flatten(input_shape=(28,28)))
model.add(Dense(128, activation='relu'))
model.add(Dense(32,activation='relu'))
model.add(Dense(11,activation='softmax'))
model.summary()

model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])

history = model.fit(x_train,y_train,epochs=25,validation_split=0.2)

y_prob = model.predict(x_test)
y_pred = y_prob.argmax(axis=1)

from sklearn.metrics import accuracy_score
accuracy_score(y_test,y_pred)

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.show()

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.show()
# neurosymbolic part
# checking even number

def is_even(n):
    return n%2 ==0

def is_prime(n):
    if n<2:
        return False
    for i in range(2,int(n**0.5)+1):
        if n% i==0:
            return False
    return True


# picking image from test set

img_idx = 247
img = np.expand_dims(x_test[img_idx],axis=0)


# predict digit

pred_probs = model.predict(img)
pred_digit = np.argmax(pred_probs)

print(f"Neural network prediction: {pred_digit}")
print(f"Ground truth: {y_test[img_idx]}")

print(f"Is it even? {is_even(pred_digit)}")
print(f"Is it prime? {is_prime(pred_digit)}")