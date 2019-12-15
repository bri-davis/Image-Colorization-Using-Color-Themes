import cv2
import numpy as np

models = ['r10c1', 'r10c10', 'r10c100', 'r50c1', 'r50c10', 'r50c100', 'r100c1', 'r100c100']
for model in models:
    means = []
    for i in range(10000):
	    img = cv2.imread('checkpoints/{}/output/{}.jpg'.format(model, i))
	    try:
		    means.append(np.mean(img, axis=(0, 1)))
	    except Exception:
		    break
    print(model)
    print(np.mean(means, axis=0))
