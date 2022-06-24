import time
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_datasets as tfds
from keras.datasets import cifar10
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers import Convolution2D, MaxPooling2D
from keras.models import Sequential
from keras.layers import Dense, Dropout, Conv2D, MaxPool2D, Flatten, ZeroPadding2D
from keras.utils import np_utils


def bens_model(X_train, y_train, X_test, y_test):
    # building a linear stack of layers with the sequential model
    cnn_classifier = Sequential()

    # convolutional layer
    cnn_classifier.add(
        Conv2D(50, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', input_shape=(32, 32, 3)))

    # convolutional layer
    cnn_classifier.add(Conv2D(75, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'))
    cnn_classifier.add(MaxPool2D(pool_size=(2, 2)))
    cnn_classifier.add(Dropout(0.25))

    cnn_classifier.add(Conv2D(125, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'))
    cnn_classifier.add(MaxPool2D(pool_size=(2, 2)))
    cnn_classifier.add(Dropout(0.25))

    # flatten output of conv
    cnn_classifier.add(Flatten())

    # hidden layer
    cnn_classifier.add(Dense(500, activation='relu'))
    cnn_classifier.add(Dropout(0.4))
    cnn_classifier.add(Dense(250, activation='relu'))
    cnn_classifier.add(Dropout(0.3))
    # output layer
    cnn_classifier.add(Dense(10, activation='softmax'))

    # compiling the sequential model
    cnn_classifier.compile(loss='categorical_crossentropy', metrics=['accuracy'], optimizer='adam')

    # training the model for 10 epochs
    cnn_classifier.fit(X_train, y_train, batch_size=32, epochs=75, validation_data=(X_test, y_test))
    # cnn_classifier.summary()


def create_cnn_model(num_classes=10, input_shape=(32, 32, 3)):
    model = Sequential(
        [
            Convolution2D(32, (3, 3), activation='relu', input_shape=input_shape),
            MaxPooling2D(pool_size=2),

            Convolution2D(filters=64, kernel_size=4, padding='same', activation='relu'),
            MaxPooling2D(pool_size=2),

            Convolution2D(filters=128, kernel_size=3, padding='same', activation='relu'),
            MaxPooling2D(pool_size=2),

            Convolution2D(filters=128, kernel_size=2, padding='same', activation='relu'),
            MaxPooling2D(pool_size=2),

            Dropout(0.25),
            Flatten(),

            Dense(units=128, activation='relu'),
            Dense(units=64, activation='relu'),
            Dense(units=32, activation='relu'),
            Dense(units=num_classes, activation='softmax')
        ]
    )
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model


def load_noisy_dataset():
    noise_file = np.load('cifar-10-100n-main/data/CIFAR-10_human.npy', allow_pickle=True)
    clean_label = noise_file.item().get('clean_label')
    worst_label = noise_file.item().get('worse_label')
    aggre_label = noise_file.item().get('aggre_label')
    random_label1 = noise_file.item().get('random_label1')
    random_label2 = noise_file.item().get('random_label2')
    random_label3 = noise_file.item().get('random_label3')

    classes_name = {0: 'airplane',
                    1: 'automobile',
                    2: 'bird',
                    3: 'cat',
                    4: 'deer',
                    5: 'dog',
                    6: 'frog',
                    7: 'horse',
                    8: 'ship',
                    9: 'truck'}
    return noise_file


def load_cifar():
    # The noisy label matches with following tensorflow dataloader
    # train_ds, test_ds = tfds.load('cifar10', split=['train', 'test'], as_supervised=True, batch_size=-1)
    # train_images, train_labels = tfds.as_numpy(train_ds)
    # test_images, test_labels = tfds.as_numpy(test_ds)

    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')

    # normalizing the data to help with the training
    X_train /= 255
    X_test /= 255

    # one-hot encoding using keras' numpy-related utilities
    n_classes = 10
    y_train = np_utils.to_categorical(y_train, n_classes)
    y_test = np_utils.to_categorical(y_test, n_classes)

    return X_train, y_train, X_test, y_test


def train_model(model, X_train, y_train, X_test, y_test):
    early_stop = EarlyStopping(
        monitor='val_accuracy',
        patience=8,
        mode='auto',
    )
    cache = "saved_models/best_conf.ckpt"
    checkpoint = ModelCheckpoint(cache,
                                 monitor='val_accuracy',
                                 verbose=True,
                                 save_best_only=True,
                                 save_weights_only=True,
                                 mode='max')

    history = model.fit(X_train, y_train, batch_size=8,
                        epochs=50,
                        callbacks=[checkpoint, early_stop],
                        validation_data=(X_test, y_test),
                        verbose=True)
    model.load_weights("saved_models/best_conf.ckpt")
    return model, history


def plot_conf_mat(conf_mat):
    fig, ax = plt.subplots(figsize=(4, 4))
    ax = sns.heatmap(conf_mat,
                     annot=True,
                     cbar=False,
                     fmt='d')
    plt.xlabel("Predicted label")
    plt.ylabel("Actual label")
    plt.show()


def plot_model_results(model, history, X_test, y_test):
    model.evaluate(X_test, y_test)
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()

    plt.plot(history.history['accuracy'])
    plt.show()

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()


def main():
    # noisy_dataset = load_noisy_dataset()
    cnn_model = create_cnn_model()
    X_train, y_train, X_test, y_test = load_cifar()
    model, history = train_model(cnn_model, X_train, y_train, X_test, y_test)

    y_predicted = model.predict(X_test)
    # Get back from catagorical to lables
    y_actual = np.argmax(y_test, axis=1)
    print(classification_report(y_actual, y_predicted))
    conf_mat = confusion_matrix(y_actual, y_predicted)
    plot_conf_mat(conf_mat)


if __name__ == '__main__':
    start = time.time()
    main()
    end = time.time()
    print(f"total time took is: {end - start}")
