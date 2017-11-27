import os,glob
import numpy as np
import pickle
from PIL import Image
from sklearn.model_selection import train_test_split

#download notMNIST


path_to_notmnist = '/Volumes/HDD 1/Downloads/notMNIST_small/'

letters = sorted([name for name in os.listdir(path_to_notmnist) if os.path.isdir(os.path.join(path_to_notmnist, name))])

fileCounter = []
for actid,act in enumerate(letters):
    fileCounter.append(len(glob.glob1(os.path.join(path_to_notmnist,act),"*.png")))

fileCounter=np.array(fileCounter)
countall = np.sum(fileCounter)

images=np.zeros((countall,1,32,32),dtype=np.uint8)
labels=np.zeros(countall,dtype=np.uint8)
k = 0
i = 0
for actid,act in enumerate(letters):
    for img in glob.glob1(os.path.join(path_to_notmnist,act),"*.png"):
        try:
            im = Image.open(os.path.join(path_to_notmnist,act,img))
            im = im.resize((32,32), Image.ANTIALIAS)
            images[k] = np.array(im).reshape((1,32,32))
            labels[k] = i
            k = k + 1
        except Exception as e:
            print(e)

    i = i + 1

images_train, images_test, labels_train, labels_test = train_test_split(images, labels, test_size=0.1, random_state=42)

train={}
train['features'] = images_train
train['labels'] = labels_train
test={}
test['features'] = images_test
test['labels'] = labels_test

with open(os.path.join(path_to_notmnist,'notmnist_train.pkl'), 'wb') as output:
    pickle.dump(train, output, pickle.HIGHEST_PROTOCOL)

with open(os.path.join(path_to_notmnist,'notmnist_test.pkl'), 'wb') as output:
    pickle.dump(test, output, pickle.HIGHEST_PROTOCOL)
