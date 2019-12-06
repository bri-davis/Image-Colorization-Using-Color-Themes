import cv2
import numpy as np

means = []
for i in range(10000):
	img = cv2.imread('checkpoints/pretrained-cifar-blue2/output/{}.jpg'.format(i))
	try:
		means.append(np.mean(abs(np.array([209, 172, 48]) - img)))
	except Exception:
		break

baseline_means = []
for i in range(10000):
	img = cv2.imread('checkpoints/pretrained-cifar/output/{}.jpg'.format(i))
	try:
		baseline_means.append(np.mean(abs(np.array([209, 172, 48]) - img)))
	except Exception:
		break

print(np.mean(means))
print(np.mean(baseline_means))
