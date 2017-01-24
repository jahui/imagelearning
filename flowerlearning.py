
import os
from PIL import Image
import numpy as np
from sklearn import svm
import time
#from sklearn import datasets

TARGET_SIZE = (480, 480)

def main():

	start = time.time()
	features, labels = getImage("daisy")
	print("Images for daisy loaded.")
	ftemp, ltemp = getImage("roses")
	print("Images for roses loaded.")
	features += ftemp
	labels += ltemp
	ftemp, ltemp = getImage("tulips")
	print("Images for tulips loaded.")
	features += ftemp
	labels += ltemp
	end = time.time()

	h, m, s = getTime(start, end)
	print("Time elapsed preparing features: {0:.0f}:{1:.0f}:{2:.2f}".format(h, m, s))
	
	#'''

	start = time.time()
	clf = svm.SVC(gamma = 0.001, C = 100.)
	clf.fit(features[:-1], labels[:-1])
	end = time.time()

	h, m, s = getTime(start, end)
	print("Time elapsed fitting the model: {0:.0f}:{1:.0f}:{2:.2f}".format(h, m, s))

	start = time.time()
	print(clf.predict(features[-1:]))
	end = time.time()

	h, m, s = getTime(start, end)
	print("Time elapsed making prediction: {0:.0f}:{1:.0f}:{2:.2f}".format(h, m, s))
	#'''

def getImage(imageClass):
	baseImagePath = os.path.join(os.getcwd(), "flower_photos", imageClass)
	features = []
	labels = []
	
	for root, directory, files in os.walk(baseImagePath):
		for f in files:
			if f.endswith("jpg"):
				im = Image.open(os.path.join(root, f))
				im = im.resize(TARGET_SIZE)

				im = list(im.getdata())
				# list of sequences
				im = list(map(list, im))
				# list of lists
				im = np.array(im)
				#print(im)
				matSize = im.shape[0] * im.shape[1]
				im = im.reshape(1, matSize)
				#list with 1 row list
				
				features.append(im[0])
				labels.append(imageClass)
	return features, labels

def getTime(start, end):
	m, s = divmod(end - start, 60)
	h, m = divmod(m, 60)
	return (h, m, s)

if __name__ == '__main__':
	main()