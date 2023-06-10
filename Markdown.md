```python
import tensorflow as tf
from tensorflow import keras
```


```python
fashion_mnist = keras.datasets.fashion_mnist
(X_train_full, y_train_full), (X_test, y_test) = fashion_mnist.load_data()
```

    Downloading data from https://storage.googleapis.com/tensorflow/tf-keras-datasets/train-labels-idx1-ubyte.gz
    29515/29515 [==============================] - 0s 1us/step
    Downloading data from https://storage.googleapis.com/tensorflow/tf-keras-datasets/train-images-idx3-ubyte.gz
    26421880/26421880 [==============================] - 1s 0us/step
    Downloading data from https://storage.googleapis.com/tensorflow/tf-keras-datasets/t10k-labels-idx1-ubyte.gz
    5148/5148 [==============================] - 0s 0us/step
    Downloading data from https://storage.googleapis.com/tensorflow/tf-keras-datasets/t10k-images-idx3-ubyte.gz
    4422102/4422102 [==============================] - 0s 0us/step



```python
X_train_full.shape
```




    (60000, 28, 28)




```python
X_train_full.dtype
```




    dtype('uint8')




```python
X_valid, X_train = X_train_full[:5000] / 255.0, X_train_full[5000:] / 255.0
y_valid, y_train = y_train_full[:5000], y_train_full[5000:]
```

When the label is equivalent to 5, it signifies the digit 5!


```python
class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
                   "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
```


```python
class_names[y_train[0]]
```




    'Coat'



## Creating a model using the Sequential API; Classification MLP


```python
model = keras.models.Sequential()
model.add(keras.layers.Flatten(input_shape=[28, 28]))
model.add(keras.layers.Dense(300, activation="relu"))
model.add(keras.layers.Dense(100, activation="relu"))
model.add(keras.layers.Dense(10, activation="softmax"))
```


```python
model.summary()
```

    Model: "sequential_1"
    _________________________________________________________________
     Layer (type)                Output Shape              Param #   
    =================================================================
     flatten_1 (Flatten)         (None, 784)               0         
                                                                     
     dense_3 (Dense)             (None, 300)               235500    
                                                                     
     dense_4 (Dense)             (None, 100)               30100     
                                                                     
     dense_5 (Dense)             (None, 10)                1010      
                                                                     
    =================================================================
    Total params: 266,610
    Trainable params: 266,610
    Non-trainable params: 0
    _________________________________________________________________



```python
model.layers[1].name
```




    'dense_3'




```python
model.compile(loss="sparse_categorical_crossentropy",
              optimizer="sgd",
              metrics=["accuracy"])
```


```python
history = model.fit(X_train, y_train, epochs=30,
                   validation_data=(X_valid, y_valid))
```

    Epoch 1/30
    1719/1719 [==============================] - 4s 2ms/step - loss: 0.7052 - accuracy: 0.7705 - val_loss: 0.5100 - val_accuracy: 0.8216
    Epoch 2/30
    1719/1719 [==============================] - 3s 2ms/step - loss: 0.4838 - accuracy: 0.8314 - val_loss: 0.4622 - val_accuracy: 0.8416
    Epoch 3/30
    1719/1719 [==============================] - 3s 2ms/step - loss: 0.4385 - accuracy: 0.8473 - val_loss: 0.4141 - val_accuracy: 0.8610
    Epoch 4/30
    1719/1719 [==============================] - 3s 2ms/step - loss: 0.4133 - accuracy: 0.8540 - val_loss: 0.3966 - val_accuracy: 0.8626
    Epoch 5/30
    1719/1719 [==============================] - 3s 2ms/step - loss: 0.3924 - accuracy: 0.8609 - val_loss: 0.3851 - val_accuracy: 0.8692
    Epoch 6/30
    1719/1719 [==============================] - 3s 2ms/step - loss: 0.3773 - accuracy: 0.8665 - val_loss: 0.3682 - val_accuracy: 0.8764
    Epoch 7/30
    1719/1719 [==============================] - 3s 2ms/step - loss: 0.3639 - accuracy: 0.8706 - val_loss: 0.3926 - val_accuracy: 0.8556
    Epoch 8/30
    1719/1719 [==============================] - 3s 2ms/step - loss: 0.3521 - accuracy: 0.8755 - val_loss: 0.3692 - val_accuracy: 0.8696
    Epoch 9/30
    1719/1719 [==============================] - 3s 2ms/step - loss: 0.3431 - accuracy: 0.8772 - val_loss: 0.3497 - val_accuracy: 0.8758
    Epoch 10/30
    1719/1719 [==============================] - 3s 2ms/step - loss: 0.3331 - accuracy: 0.8821 - val_loss: 0.3384 - val_accuracy: 0.8804
    Epoch 11/30
    1719/1719 [==============================] - 3s 2ms/step - loss: 0.3257 - accuracy: 0.8832 - val_loss: 0.3328 - val_accuracy: 0.8810
    Epoch 12/30
    1719/1719 [==============================] - 3s 2ms/step - loss: 0.3171 - accuracy: 0.8850 - val_loss: 0.3337 - val_accuracy: 0.8790
    Epoch 13/30
    1719/1719 [==============================] - 3s 2ms/step - loss: 0.3095 - accuracy: 0.8886 - val_loss: 0.3244 - val_accuracy: 0.8882
    Epoch 14/30
    1719/1719 [==============================] - 3s 2ms/step - loss: 0.3022 - accuracy: 0.8916 - val_loss: 0.3223 - val_accuracy: 0.8880
    Epoch 15/30
    1719/1719 [==============================] - 3s 2ms/step - loss: 0.2965 - accuracy: 0.8923 - val_loss: 0.3227 - val_accuracy: 0.8868
    Epoch 16/30
    1719/1719 [==============================] - 3s 2ms/step - loss: 0.2898 - accuracy: 0.8959 - val_loss: 0.3590 - val_accuracy: 0.8678
    Epoch 17/30
    1719/1719 [==============================] - 3s 2ms/step - loss: 0.2842 - accuracy: 0.8962 - val_loss: 0.3245 - val_accuracy: 0.8812
    Epoch 18/30
    1719/1719 [==============================] - 3s 2ms/step - loss: 0.2787 - accuracy: 0.8993 - val_loss: 0.3168 - val_accuracy: 0.8874
    Epoch 19/30
    1719/1719 [==============================] - 3s 2ms/step - loss: 0.2737 - accuracy: 0.8999 - val_loss: 0.3352 - val_accuracy: 0.8752
    Epoch 20/30
    1719/1719 [==============================] - 3s 2ms/step - loss: 0.2679 - accuracy: 0.9033 - val_loss: 0.3090 - val_accuracy: 0.8908
    Epoch 21/30
    1719/1719 [==============================] - 3s 2ms/step - loss: 0.2635 - accuracy: 0.9043 - val_loss: 0.3278 - val_accuracy: 0.8774
    Epoch 22/30
    1719/1719 [==============================] - 3s 2ms/step - loss: 0.2583 - accuracy: 0.9060 - val_loss: 0.3203 - val_accuracy: 0.8856
    Epoch 23/30
    1719/1719 [==============================] - 3s 2ms/step - loss: 0.2543 - accuracy: 0.9074 - val_loss: 0.2966 - val_accuracy: 0.8940
    Epoch 24/30
    1719/1719 [==============================] - 3s 2ms/step - loss: 0.2495 - accuracy: 0.9096 - val_loss: 0.3083 - val_accuracy: 0.8892
    Epoch 25/30
    1719/1719 [==============================] - 3s 2ms/step - loss: 0.2460 - accuracy: 0.9107 - val_loss: 0.3341 - val_accuracy: 0.8798
    Epoch 26/30
    1719/1719 [==============================] - 3s 2ms/step - loss: 0.2411 - accuracy: 0.9126 - val_loss: 0.2976 - val_accuracy: 0.8936
    Epoch 27/30
    1719/1719 [==============================] - 3s 2ms/step - loss: 0.2377 - accuracy: 0.9142 - val_loss: 0.3016 - val_accuracy: 0.8892
    Epoch 28/30
    1719/1719 [==============================] - 3s 2ms/step - loss: 0.2332 - accuracy: 0.9148 - val_loss: 0.3026 - val_accuracy: 0.8918
    Epoch 29/30
    1719/1719 [==============================] - 3s 2ms/step - loss: 0.2300 - accuracy: 0.9169 - val_loss: 0.2990 - val_accuracy: 0.8912
    Epoch 30/30
    1719/1719 [==============================] - 3s 2ms/step - loss: 0.2263 - accuracy: 0.9185 - val_loss: 0.3009 - val_accuracy: 0.8958



```python
import pandas as pd
import matplotlib.pyplot as plt

pd.DataFrame(history.history).plot(figsize=(8, 5))
plt.grid(True)
plt.gca().set_ylim(0, 1)
plt.show()
```


    
![png](output_14_0.png)
    



```python
X_new = X
```
