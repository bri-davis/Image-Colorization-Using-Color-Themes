import numpy as np
import pickle
from PIL import Image

def imsave(img, path):
    im = Image.fromarray(np.array(img).astype(np.uint8).squeeze()).convert('L')
    im.save(path)

data = []
filename = '{}/test_batch'.format('/home/tug75084/Colorizing-with-GANs/dataset/cifar10')
with open(filename, 'rb') as fo:
    batch_data = pickle.load(fo, encoding='bytes')
data = batch_data[b'data']
w = 32
h = 32
s = w * h
data = np.array(data)
data = np.dstack((data[:, :s], data[:, s:2 * s], data[:, 2 * s:]))
data = data.reshape((-1, w, h, 3))
print(data)
length = len(data)
for i, image in enumerate(data):
    print(i)
    path = 'image{0:05d}.png'.format(i)
    imsave(image, path)
