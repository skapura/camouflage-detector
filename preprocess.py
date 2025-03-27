from PIL import Image, ImageOps, ImageDraw
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from itertools import chain
import numpy as np
import shutil
import random


seed = 1
np.random.seed(seed)

train1 = Image.open('img/pencott_train_saturation.jpg')
train2 = Image.open('img/cadpat_marpat_saturate.jpg')
w, h = train2.size


### Pencott ###

# Negative class (natural scenery)
nat_a = 0
for x in range(0, 100):
    for y in range(0, 100):
        img = train1.crop((x, y, x + 50, y + 50))
        img.save('img/stage/nature_a_' + str(nat_a) + '.jpg')
        nat_a += 1

idx = list(range(0, nat_a))
random.shuffle(idx)
for i in range(0, 1000):
    shutil.copyfile('img/stage/nature_a_' + str(idx[i]) + '.jpg', 'img/train/nature_a_' + str(i) + '.jpg')


# Positive class (camouflaged object)
pen_count = 0
for x in range(380, 380+100):
    for y in range(150, 150+100):
        img = train1.crop((x, y, x + 50, y + 50))
        img.save('img/stage/pencott_' + str(pen_count) + '.jpg')
        pen_count += 1

idx = list(range(0, pen_count))
random.shuffle(idx)
for i in range(0, 1000):
    shutil.copyfile('img/stage/pencott_' + str(idx[i]) + '.jpg', 'img/train/pencott_' + str(i) + '.jpg')

#i = 0
#for x in range(0, w, 50):
#    for y in range(0, h, 50):
#        img = train2.crop((x, y, x + 50, y + 50))
#        img.save('img/train/train_' + str(x) + '_' + str(y) + '.jpg')
#        i += 1

### CadPat, MarPat ###

# Negative class (natural scenery)
nat_b = 0
for x in range(0, 100):
    for y in range(0, 100):
        img = train2.crop((x, y, x + 50, y + 50))
        img.save('img/stage/nature_b_' + str(nat_b) + '.jpg')
        nat_b += 1

idx = list(range(0, nat_b))
random.shuffle(idx)
for i in range(0, 1000):
    shutil.copyfile('img/stage/nature_b_' + str(idx[i]) + '.jpg', 'img/train/nature_b_' + str(i) + '.jpg')

# Positive class (MarPat object)
mar_count = 0
for x in range(960, 960+100):
    for y in range(500, 500+100):
        img = train2.crop((x, y, x + 50, y + 50))
        img.save('img/stage/marpat_' + str(mar_count) + '.jpg')
        mar_count += 1

idx = list(range(0, mar_count))
random.shuffle(idx)
for i in range(0, 1000):
    shutil.copyfile('img/stage/marpat_' + str(idx[i]) + '.jpg', 'img/train/marpat_' + str(i) + '.jpg')

# Positive class (CadPat object)
cad_count = 0
for x in range(320, 320+100):
    for y in range(430, 430+100):
        img = train2.crop((x, y, x + 50, y + 50))
        img.save('img/stage/cadpat_' + str(cad_count) + '.jpg')
        cad_count += 1

idx = list(range(0, cad_count))
random.shuffle(idx)
for i in range(0, 1000):
    shutil.copyfile('img/stage/cadpat_' + str(idx[i]) + '.jpg', 'img/train/cadpat_' + str(i) + '.jpg')
