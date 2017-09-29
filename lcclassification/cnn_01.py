from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D

from scipy import misc
from sklearn.model_selection import train_test_split
from sklearn import preprocessing

import keras
import matplotlib.pyplot as plt
import numpy as np
import os.path
import pandas as pd
import scikitplot as skplt

def main():
    input_shape = (22, 24, 1)

    X, y = get_data()

    encoder = preprocessing.LabelEncoder()
    binarizer = preprocessing.LabelBinarizer()
    y = binarizer.fit_transform(encoder.fit_transform(y))

    num_classes = len(encoder.classes_)

    print(y)

    print("X.shape: ", X.shape)
    print("y.shape: ", y.shape)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.33, random_state=42)

    print("X_train.shape: ", X_train.shape)
    print("y_train.shape: ", y_train.shape)

    print("X_test.shape: ", X_test.shape)
    print("y_test.shape: ", y_test.shape)

    model = Sequential([
        Conv2D(32, (3, 3), activation="relu", input_shape=input_shape),
        Conv2D(64, (3, 3), activation="relu"),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(0.25),
        Flatten(),
        Dense(128, activation="relu"),
        Dropout(0.5),
        Dense(num_classes, activation="softmax")
    ])

    model.compile(optimizer="rmsprop",
              loss="categorical_crossentropy",
              metrics=["accuracy"])

    model.fit(X_train, y_train, epochs=3, batch_size=32,
            validation_data=(X_test, y_test))

    score = model.evaluate(X_test, y_test, verbose=0)
    print("Test loss:", score[0])
    print("Test accuracy:", score[1])

    y_pred = model.predict(X_test)
    y_pred = encoder.inverse_transform(binarizer.inverse_transform(y_pred))

    y_test = encoder.inverse_transform(binarizer.inverse_transform(y_test))

    skplt.plotters.plot_confusion_matrix(y_test, y_pred, labels=encoder.classes_)
    plt.show()

def get_data():
    id_col = "Numerical_ID"
    has_curve_col = "has_curve"
    type_col = "Var_Type"
    data_file = "~/Datasets/CRTS_Release2/CRTS_Varcat.csv"
    images_dir = "curves"
    images_ending = ".png"

    data = pd.read_csv(data_file)
    data = data[data[has_curve_col]]

    ids = data[id_col]
    y = data[type_col]

    def get_img(i):
        img_file = os.path.join(images_dir, str(i) + images_ending)

        return misc.imread(img_file)

    X = np.array([np.expand_dims(get_img(i), axis=2) for i in ids])
    return X, y

if __name__ == "__main__":
    main()
