import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pickle

history = pickle.load(open("./data/logs/trainHistoryDict_dice_2", "rb"))
weights = pickle.load(open("./data/logs/trainHistoryDict_lowWeightsRandom", "rb"))
plt.plot(history['binary_crossentropy'])
plt.plot(history['val_binary_crossentropy'])
plt.title('weights initialized as zero: cross-entropy')
plt.ylabel('binary_crossentropy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')

plt.show()
# summarize history for loss
plt.plot(history['loss'])
plt.plot(history['val_loss'])
plt.title('Model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

plt.plot(weights['binary_crossentropy'])
plt.plot(weights['val_binary_crossentropy'])
plt.title('weights randomly initialized: cross-entropy')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()



plt.plot(history['dice_coefficient'])
plt.plot(history['val_dice_coefficient'])
plt.title('Dice coefficient')
plt.ylabel('dice coefficient')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()