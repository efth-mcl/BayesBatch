from zca import ZCA
from sklearn.preprocessing import StandardScaler

from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import utils

from tensorflow.keras.layers import Dense, Flatten, Dropout, BatchNormalization, Conv2D, MaxPooling2D, PReLU
import tensorflow as tf

import matplotlib.pyplot as plt

from tensorflow.keras.datasets import fashion_mnist
import numpy as np
from sklearn.model_selection import train_test_split

from skimage import exposure

from matplotlib.legend_handler import HandlerTuple

global BATCH_SIZE, OPTIMIZER, EPOCHS
BATCH_SIZE = 128
OPTIMIZER = Adam(learning_rate=1e-4, decay=1e-6, amsgrad=True)
EPOCHS = 60


# ---------------------- #
# load data & -----------#
# preprocessing ---------#
# ---------------------- #
class FashionMnistDataset(object):
    def __init__(self, img_preprocessing: str = None):
        self.x_train, self.y_train, self.x_val, self.y_val, self.x_test, self.y_test = self.__load_data(
            img_preprocessing)
        self.set = ["T-shirt", "Trouser", "Pullover", "Dress", "Coat",
                    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]

    def __load_data(self, img_preprocessing: str = None):
        (x_train, y_train), (x_val_test, y_val_test) = fashion_mnist.load_data()

        x_train = x_train.astype('float32') / 255.0
        x_val_test = x_val_test.astype('float32') / 255.0

        if img_preprocessing == "std_normal":
            x_train_flat = x_train.reshape(-1, 28 * 28)
            x_val_test_flat = x_val_test.reshape(-1, 28 * 28)

            std = StandardScaler().fit(x_train_flat)
            x_train = std.transform(x_train_flat).reshape(-1, 28, 28)
            x_val_test = std.transform(x_val_test_flat).reshape(-1, 28, 28)

        elif img_preprocessing == "eq_hist":
            x_train = exposure.equalize_hist(x_train)
            x_val_test = exposure.equalize_hist(x_val_test)

        elif img_preprocessing == "zca_whiting":
            x_train_flat = x_train.reshape(-1, 28 * 28)
            x_val_test_flat = x_val_test.reshape(-1, 28 * 28)
            zca = ZCA().fit(x_train_flat)
            x_train = zca.transform(x_train_flat).reshape(-1, 28, 28)
            x_val_test = zca.transform(x_val_test_flat).reshape(-1, 28, 28)

        x_train = x_train.reshape(-1, 28, 28, 1)
        x_val_test = x_val_test.reshape(-1, 28, 28, 1)

        x_test, x_val, y_test, y_val = train_test_split(x_val_test, y_val_test, train_size=0.5, random_state=42)
        y_train = utils.to_categorical(y_train, 10)
        y_val = utils.to_categorical(y_val, 10)
        y_test = utils.to_categorical(y_test, 10)

        return x_train, y_train, x_val, y_val, x_test, y_test

    def plot_data(self):

        i = 1
        plt.figure(1)
        plt.imshow(self.x_train[i].reshape(28, 28), cmap='gray')
        plt.show()

        fig = plt.figure(figsize=(8, 8))
        fig.subplots_adjust(hspace=0.4, wspace=0.4)
        for i in range(1, 11):
            j = 0
            while True:
                label = self.y_test[j].argmax()
                if label == i - 1:
                    image = self.x_test[j]
                    break
                j += 1
            ax = fig.add_subplot(5, 5, i)
            ax.imshow(image.reshape(28, 28), cmap='gray')
            ax.axis('off')
            ax.set_title(self.set[i - 1])
        plt.show()


# ---------------------- #
# model -----------------#
# ---------------------- #
def create_model(**kwargs):
    print(kwargs)
    dropout1 = kwargs['dropout1']
    dropout2 = kwargs['dropout2']
    dropout3 = kwargs['dropout3']
    dropout4 = kwargs['dropout4']
    dropout5 = kwargs['dropout5']
    l2_1 = kwargs['l2_1']
    l2_2 = kwargs['l2_2']
    l2_3 = kwargs['l2_3']
    l2_4 = kwargs['l2_4']
    layers_123_dist = kwargs['layers_123_dist']
    kernel_size = (3, 3)

    l2_1 = tf.keras.regularizers.l2(l2_1)
    l2_2 = tf.keras.regularizers.l2(l2_2)
    l2_3 = tf.keras.regularizers.l2(l2_3)
    l2_4 = tf.keras.regularizers.l2(l2_4)
    model = Sequential([
        Conv2D(64, kernel_size=kernel_size, padding='same', kernel_regularizer=l2_1, input_shape=(28, 28, 1)),
        BatchNormalization(),
        PReLU(),
        Conv2D(64, kernel_size=kernel_size, kernel_regularizer=l2_1),
        BatchNormalization(),
        PReLU(),
        MaxPooling2D(pool_size=(2, 2)),
        Dropout(dropout1)
    ])
    if layers_123_dist > 1 / 3:
        model.add(Conv2D(128, kernel_size=kernel_size, padding='same', kernel_regularizer=l2_2))
        model.add(BatchNormalization())
        model.add(PReLU())
        model.add(Conv2D(128, kernel_size=kernel_size, kernel_regularizer=l2_1))
        model.add(BatchNormalization())
        model.add(PReLU())
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(dropout2))

        if layers_123_dist > 2 / 3:
            model.add(Conv2D(256, kernel_size=kernel_size, padding='same', kernel_regularizer=l2_3))
            model.add(BatchNormalization())
            model.add(PReLU())
            model.add(Conv2D(256, kernel_size=kernel_size, kernel_regularizer=l2_1))
            model.add(BatchNormalization())
            model.add(PReLU())
            model.add(MaxPooling2D(pool_size=(2, 2)))
            model.add(Dropout(dropout3))

    model.add(Flatten())

    model.add(Dense(1024, kernel_regularizer=l2_4))
    model.add(PReLU())
    model.add(Dropout(dropout4))

    model.add(Dense(512, kernel_regularizer=l2_4))
    model.add(PReLU())
    model.add(Dropout(dropout5))
    model.add(Dense(10, activation='softmax'))

    model.compile(
        optimizer=OPTIMIZER,
        loss='categorical_crossentropy',
        metrics=['acc'])
    return model


# ---------------------- #
# train ---------------- #
# ---------------------- #
def simple_train(dataset: FashionMnistDataset, epochs, bayes=False, **kwargs):
    model = create_model(**kwargs)
    callbacks = create_callbacks()
    history = model.fit(dataset.x_train, dataset.y_train, batch_size=BATCH_SIZE, epochs=epochs,
                        validation_data=(dataset.x_val, dataset.y_val), callbacks=callbacks)
    _, test_accuracy = model.evaluate(dataset.x_test, dataset.y_test)
    if bayes:
        return test_accuracy
    history = history.history
    return test_accuracy, history, model.predict(dataset.x_test)


def increase_batch_size_train(dataset: FashionMnistDataset, epochs, bayes=False, **kwargs):
    model = create_model(**kwargs)
    callbacks = create_callbacks()
    b_s = BATCH_SIZE
    history = []
    print('batchsize: {0}'.format(b_s))

    h = model.fit(dataset.x_train, dataset.y_train, batch_size=b_s, epochs=1,
                  validation_data=(dataset.x_val, dataset.y_val),
                  callbacks=callbacks)
    history.append(h.history)
    _, val_accuracy = model.evaluate(dataset.x_val, dataset.y_val)
    _, train_accuracy = model.evaluate(dataset.x_train, dataset.y_train)

    while True:
        if abs(val_accuracy - train_accuracy) > 0.01:
            # memoey drop so we use up to 1000
            if b_s < 1000:
                b_s = round(b_s * 1.25)
        print('batchsize: {0}'.format(b_s))
        h = model.fit(dataset.x_train, dataset.y_train, batch_size=b_s,
                      epochs=epochs, validation_data=(dataset.x_val, dataset.y_val),
                      callbacks=callbacks)
        history.append(h.history)
        _, val_accuracy = model.evaluate(dataset.x_val, dataset.y_val)
        _, train_accuracy = model.evaluate(dataset.x_train, dataset.y_train)

        if train_accuracy > 0.99:
            break

    _, test_accuracy = model.evaluate(dataset.x_test, dataset.y_test)
    if bayes:
        return test_accuracy
    else:
        history_merge = {'loss': [], 'acc' : [], 'val_loss': [], 'val_acc': []}
        for h in history:
            for k, _ in h.items():
                history_merge[k] += h[k]
        return test_accuracy, history_merge, model.predict(dataset.x_test)


def create_callbacks():
    callbacks = []
    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        verbose=1,
        save_weights_only=True,
        filepath='.')
    callbacks.append(cp_callback)

    early_call = tf.keras.callbacks.EarlyStopping(monitor='val_acc', patience=7, min_delta=0.02)
    callbacks.append(early_call)
    return callbacks


# ---------------------- #
# Results -------------- #
# ---------------------- #
def preprocessing_per_train_method_bar_figure(dataf, figsize=(16, 8), ylim_max=1.5, legend_fontsize=20,
                                             legend_loc='upper left', axes_label_fontsize=20, ticks_fontsize=20):
    plt.figure(figsize=figsize)
    index = np.arange(4)
    bar_width = 0.15
    for i, key in enumerate(dataf.keys()):
        plt.bar(index + bar_width * i, dataf[key], bar_width, label=key)

    plt.legend(fontsize=legend_fontsize, loc=legend_loc)
    plt.ylabel(r'$Accuracy$',
               fontsize=axes_label_fontsize)

    plt.xticks(index + bar_width, [r'${}$'.format(idx) for idx in dataf.index],
               fontsize=ticks_fontsize)
    plt.yticks(fontsize=ticks_fontsize)
    plt.tight_layout()

    # the upper lim is hard coded based in max of metrics
    # maybe in future fix this all metrics ranges
    plt.ylim(0, ylim_max)
    plt.show()


def history_plot_figures(histrory_dict):
    def plot_figure(tran_metric, val_metric, metric_type='Loss'):
        c = ['red', 'brown', 'orange', 'darkmagenta', 'gold', 'coral', 'lightgreen', 'slategray', 'pink', 'tan',
             'darkkhaki', 'greenyellow', 'gray', 'lime', 'blue', 'indigo']
        models = ['NoPr/simple', 'NoPr/batch', 'NoPr/bayes', 'NoPr/bayes+batch', 'STD/simple', 'STD/batch', 'STD/bayes',
                  'STD/bayes+batch', 'ZCA/simple', 'ZCA/batch', 'ZCA/bayes', 'ZCA/bayes+bacth', 'HE/simple', 'HE/batch',
                  'HE/bayes', 'HE/bayes+batch']
        plt.figure(figsize=(30, 20))
        plot_train = []
        plot_val = []
        for i in range(16):
            plot_train.append(
                plt.plot(tran_metric[i], '-^', color=c[i], markersize=12)
            )
            plot_val.append(
                plt.plot(val_metric[i], '--*', color=c[i], markersize=12)
            )
        plt.legend([(plot_train[i][0], plot_val[i][0]) for i in range(16)], models, numpoints=1,
                   handler_map={tuple: HandlerTuple(ndivide=None)}, fontsize=24)
        plt.xlabel(r'$\# \, of \, epochs$', fontsize=24)
        plt.ylabel(r'${}$'.format(metric_type), fontsize=24)
        plt.show()

    train_loss = []
    val_loss = []
    train_score = []
    val_score = []
    for k, v in histrory_dict.items():
        # v = v.history
        train_loss.append(v['loss'])
        val_loss.append(v['val_loss'])
        train_score.append(v['acc'])
        val_score.append(v['val_acc'])

    train_loss = np.array(train_loss)
    val_loss = np.array(val_loss)
    train_score = np.array(train_score)
    val_score = np.array(val_score)

    metric_type = "Loss"
    plot_figure(train_loss, val_loss, metric_type)
    metric_type = "Accuracy_{Score}"
    plot_figure(train_score, val_score, metric_type)


def confution_matrix_figure(y_true, y_pred, labels: list = None):
    un = np.unique(y_true)
    n = len(un)
    confmatrix = np.zeros((n, n)).astype("int")
    for i in range(len(y_true)):
        confmatrix[y_true[i], y_pred[i]] += 1

    plot = plt.imshow(confmatrix, cmap="coolwarm")
    for i in range(n):
        for j in range(n):
            plt.text(i, j, confmatrix[j, i], ha="center", va="center", color="w")
    if labels is not None:
        plt.xticks([0, 1, 2, 3, 4, 5, 6, 7], fontsize=15, labels=labels, rotation=45)
        plt.yticks([0, 1, 2, 3, 4, 5, 6, 7], fontsize=15, labels=labels)
    plt.colorbar(ticks=np.unique(confmatrix[:]))
    plt.show()