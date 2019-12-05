import cv2
import numpy as np

means = []
for i in range(10000):
	img = cv2.imread('checkpoints/pretrained-cifar/output/{}.jpg'.format(i))
	try:
		means.append(np.mean(img, axis=(0, 1)))
	except Exception:
		break

print(np.mean(means, axis=0))
