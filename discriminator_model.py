from keras.models import Sequential
from keras.layers.merge import concatenate
from keras.models import Model
from keras.layers import Dense, Dropout, BatchNormalization, Conv2D, MaxPool2D, Flatten, Input
import time
import os
import pickle
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.datasets import cifar10
from sklearn.metrics import classification_report, confusion_matrix
from keras.utils import np_utils

def tag_train_images_by_cnn(cnn, train_set):
    proba_preds_cnn = [cnn.predict(img.reshape(1, img.shape[0], img.shape[1], img.shape[2])) for img in train_set]
    final_labels_cnn = [list(pred[0]).index(max(pred[0])) for pred in proba_preds_cnn]
    with open('cnn_predictions.pickle', 'wb') as handle:
        pickle.dump(proba_preds_cnn, handle)
    with open('cnn_final_labels.pickle', 'wb') as handle:
        pickle.dump(final_labels_cnn, handle)

def discriminator_model():
    # building a linear stack of layers with the sequential model
    discriminator2 = Sequential()
    categorical_input = Input(shape=(1), name='categorical_lbl')
    dense01 = Dense(8, activation='relu')(categorical_input)

    inputL = Input(shape=(32, 32, 3), name='image')
    conv1 = Conv2D(32, (3, 3), strides=(1, 1), padding='same', activation='relu')(inputL)
    maxP1 = MaxPool2D(pool_size=2)(conv1)
    conv2 = Conv2D(filters=64, kernel_size=4, padding='same', activation='relu')(maxP1)
    maxP2 = MaxPool2D(pool_size=2)(conv2)
    conv3 = Conv2D(filters=64, kernel_size=4, padding='same', activation='relu')(maxP2)
    maxP3 = MaxPool2D(pool_size=2)(conv3)
    conv4 = Conv2D(filters=128, kernel_size=3, padding='same', activation='relu')(maxP3)
    maxP4 = MaxPool2D(pool_size=2)(conv4)
    conv5 = Conv2D(filters=128, kernel_size=2, padding='same', activation='relu')(maxP4)
    maxP5 = MaxPool2D(pool_size=2)(conv5)
    do1 = Dropout(0.25)(maxP5)
    flt = Flatten()(do1)
    dense1 = Dense(128, activation='relu')(flt)
    flatten2 = Flatten()(dense1)
    merge = concatenate([dense01, flatten2])

    dense2 = Dense(64, activation='relu')(merge)
    dense3 = Dense(32, activation='relu')(dense2)
    output = Dense(1, activation='sigmoid')(dense3)

    discriminator2 = Model(inputs=[inputL, categorical_input], outputs=output)
    # compiling the sequential model
    discriminator2.compile(loss='binary_crossentropy', metrics=['accuracy'], optimizer='adam')
    discriminator2.summary()
    return discriminator2

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

    if not os.path.exists("saved_models"):
        os.mkdir("saved_models")
    cache = "saved_models/best_dicriminator.ckpt"

    checkpoint = ModelCheckpoint(cache,
                                 monitor='val_accuracy',
                                 verbose=True,
                                 save_best_only=True,
                                 save_weights_only=True,
                                 mode='max')

    history = model.fit(X_train, y_train, batch_size=16,
                        epochs=50,
                        callbacks=[checkpoint, early_stop],
                        validation_data=(X_test, y_test),
                        verbose=True)
    model.load_weights("saved_models/best_dicriminator.ckpt")
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


def create_discrimination_dataset(X_train, Y_train, X_test, Y_test):
    New_Train_Data_i = []
    New_Train_Data_l = []
    New_Train_Label = []

    noise_file = load_noisy_dataset()

    with open('cnn_final_labels.pickle', 'rb') as handle:
        cnn_labels = pickle.load(handle)

    for i, data in enumerate(X_train):
        img_data = X_train[i]
        x_org = tf.stack(img_data)
        y_clean = cnn_labels[i] # noise_file.item().get('clean_label')[i]#     y_clean = list(Y_train[i]).index(1)
        # New_Train_Label.append(1)
        # New_Train_Data_i.append(x)
        # New_Train_Data_l.append(y_clean)
        past_noises = []
        for noise in ['worse_label', 'aggre_label', 'random_label1', 'random_label2', 'random_label3']:
            noise_label = noise_file.item().get(noise)[i]
            if y_clean != noise_label and noise_label not in past_noises:
                x = tf.stack(img_data)
                y = noise_label
                # balance with-:
                New_Train_Label.append(0)
                New_Train_Data_i.append(x_org)
                New_Train_Data_l.append(y_clean)
                past_noises.append(noise_label)
                New_Train_Label.append(1)
                New_Train_Data_i.append(x)
                New_Train_Data_l.append(y)

    print("Balance of labels : ", sum(New_Train_Label) / len(New_Train_Label))
    TEST_SIZE = int(len(New_Train_Label) / 10)
    print("TEST_SIZE: ", TEST_SIZE)
    New_Train_Data_i = New_Train_Data_i[0:len(New_Train_Data_i) - TEST_SIZE]
    New_Train_Data_l = New_Train_Data_l[0:len(New_Train_Data_l) - TEST_SIZE]
    New_Train_Label = New_Train_Label[0:len(New_Train_Label) - TEST_SIZE]

    New_Test_Data_i = New_Train_Data_i[(len(New_Train_Data_i) - TEST_SIZE):]
    New_Test_Data_l = New_Train_Data_l[(len(New_Train_Data_l) - TEST_SIZE):]
    New_Test_Label = New_Train_Label[(len(New_Train_Label) - TEST_SIZE):]

    x_TrI = np.stack(New_Train_Data_i)
    x_TrI = np.array(x_TrI)
    x_TeI = np.stack(New_Test_Data_i)
    x_TeI = np.array(x_TeI)

    x_TrL = np.stack(New_Train_Data_l).reshape(len(New_Train_Data_l), 1)
    x_TrL = np.array(x_TrL)
    x_TeL = np.stack(New_Test_Data_l).reshape(len(New_Test_Data_l), 1)
    x_TeL = np.array(x_TeL)


    New_Train_Label = np.array(New_Train_Label)
    New_Test_Label = np.array(New_Test_Label)
    print("DONE!")

    return [x_TrI, x_TrL], New_Train_Label, [x_TeI, x_TeL], New_Test_Label

def main():
    # noisy_dataset = load_noisy_dataset()
    discriminator = discriminator_model()
    X_train, y_train, X_test, y_test = load_cifar()
    X_train, y_train, X_test, y_test = create_discrimination_dataset(X_train, y_train, X_test, y_test)

    model, history = train_model(discriminator, X_train, y_train, X_test, y_test)

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
