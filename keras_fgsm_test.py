import foolbox
from foolbox.models import KerasModel
from foolbox.attacks import LBFGSAttack
from foolbox.criteria import TargetClassProbability, Misclassification
import numpy as np
import keras
from keras.models import load_model
import matplotlib.pyplot as plt


kmodel = load_model('./LeNet.h5')
preprocessing = (np.array([104, 116, 123]), 1)
fmodel = KerasModel(kmodel, bounds=(0, 255))
attack = LBFGSAttack(model=fmodel, criterion=Misclassification())


adversarial_imgs = []
adversarial_labels =[]
# adversarial_imgs = np.asarray(adversarial_imgs)
# adversarial_labels = np.asarray(adversarial_labels)
# print(type(adversarial_imgs))
img_temp = np.load('./mnist_pure/x_test.npy')
# print(img_temp.shape)
img_temp = np.asarray(img_temp, dtype=np.float32)
# print(img_temp[0].shape)
label_temp = np.load('./mnist_pure/y_test.npy')
label_temp= np.asarray(label_temp, dtype=np.float32)

for i in range(0,10000):
    adversarial = attack(img_temp[i], label_temp[i])
    adversarial_imgs.append(adversarial)
    adv_labels = np.argmax(fmodel.predictions(adversarial))
    adversarial_labels.append(adv_labels)
    print(np.array(adversarial_imgs).shape, np.array(adversarial_labels).shape, 'Actual Label: {}, Adversarial Label: {}'.format(label_temp[i], adv_labels))


adversarial_imgs = np.asarray(adversarial_imgs)
adversarial_labels = np.asarray(adversarial_labels)

np.save('Temp_adv_imgs_test.npy', adversarial_imgs)
np.save('Temp_adv_labels_test.npy',adversarial_labels)
