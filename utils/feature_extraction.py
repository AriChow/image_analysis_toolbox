import numpy as np
from mahotas.features import haralick
import cv2


def haralick_features(names, distance=1):
    """
    :param names: list of full file names of the images
    :param distance: distance parameter (set to default as 1)
    :return: f (M x 13) matrix where M is the number of image files in name
    """
    f = []
    for i in range(len(names)):
        I = cv2.imread(names[i])
        if I is None or I.size == 0 or np.sum(I[:]) == 0 or I.shape[0] == 0 or I.shape[1] == 0:
            h = np.zeros((1, 13))
        else:
            I = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)
            h = haralick(I, distance=distance, return_mean=True, ignore_zeros=False)
            h = np.expand_dims(h, 0)
        if i == 0:
            f = h
        else:
            f = np.vstack((f, h))
    return f


def CNN(names, cnn):
    """
    :param names: list of full file names of the images
    :param cnn: type (VGG or inception)
    :return: f (M x 1000 for inception or M x 4096 for VGG)
    """
    from keras.applications.vgg19 import VGG19
    from keras.applications.inception_v3 import InceptionV3
    from keras.applications.vgg19 import preprocess_input
    f = []
    if cnn == 'VGG':
        model = VGG19(weights='imagenet')
        dsize = (224, 224)
    else:
        model = InceptionV3(weights='imagenet')
        dsize = (299, 299)
    for i in range(len(names)):
        img = cv2.imread(names[i])
        img = cv2.resize(img, dsize=dsize)
        img = img.astype('float32')
        x = np.expand_dims(img, axis=0)
        x = preprocess_input(x)
        features = model.predict(x)
        if i == 0:
            f = features
        else:
            f = np.vstack((f, features))
    return f


def VGG(names):
    """
    :param names: list of full file names of the images
    :return: f (M x 4096)
    """
    f = CNN(names, 'VGG')
    return f


def inception(names):
    """
    :param names: list of full file names of the images
    :return: f (M x 1000)
    """
    f = CNN(names, 'inception')
    return f
