
import os
from PIL import Image
import numpy as np
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.decomposition import PCA
import time
#from sklearn import datasets

TARGET_SIZE = (480, 480)

def main():

	#start = time.time()
	ftemp, ltemp = getImage("daisy")
	print("Images for daisy loaded.")
	features = ftemp[:-5]
	labels = ltemp[:-5]
	daisyTest = ftemp[-5:]
	daisyLabel = ltemp[-5:]


	ftemp, ltemp = getImage("roses")
	print("Images for roses loaded.")
	features += ftemp[:-5]
	labels += ltemp[:-5]
	roseTest = ftemp[-5:]
	roseLabel = ltemp[-5:]


	ftemp, ltemp = getImage("tulips")
	print("Images for tulips loaded.")
	features += ftemp[:-5]
	labels += ltemp[:-5]
	tulipTest = ftemp[-5:]
	tulipLabel = ltemp[-5:]
	#end = time.time()

	#printTime(start, end, "preparing features")
	start = time.time()
	pca = PCA(n_components=20)
	features = pca.fit_transform(features)
	end = time.time()
	printTime(start, end, "transforming feature set")

	
	clfS = trainSVM(features, labels)
	clfN = trainKNN(features, labels)
	clfD = trainDTC(features, labels)
	clfB = trainGNB(features, labels)
	
	daisyTest = pca.transform(daisyTest)
	print("SVM")
	print(clfS.score(daisyTest, daisyLabel))
	print("K Nearest Neighbors")
	print(clfN.score(daisyTest, daisyLabel))
	print("Decision Tree")
	print(clfD.score(daisyTest, daisyLabel))
	print("Naive Bayes")
	print(clfB.score(daisyTest, daisyLabel))
	print("-----------------------")

	
	roseTest = pca.transform(roseTest)
	print("SVM")
	print(clfS.score(daisyTest, daisyLabel))
	print("K Nearest Neighbors")
	print(clfN.score(daisyTest, daisyLabel))
	print("Decision Tree")
	print(clfD.score(daisyTest, daisyLabel))
	print("Naive Bayes")
	print(clfB.score(daisyTest, daisyLabel))
	print("-----------------------")

	
	tulipTest = pca.transform(tulipTest)
	print("SVM")
	print(clfS.score(daisyTest, daisyLabel))
	print("K Nearest Neighbors")
	print(clfN.score(daisyTest, daisyLabel))
	print("Decision Tree")
	print(clfD.score(daisyTest, daisyLabel))
	print("Naive Bayes")
	print(clfB.score(daisyTest, daisyLabel))
	#print("-----------------------")
	

def trainDTC(trainSet, trainLabels):
	start = time.time()
	clfD = DecisionTreeClassifier()
	clfD.fit(trainSet, trainLabels)
	end = time.time()
	printTime(start, end, "fitting the Decision Tree")
	return clfD

def trainGNB(trainSet, trainLabels):
	start = time.time()
	clfB = GaussianNB()
	clfB.fit(trainSet, trainLabels)
	end = time.time()
	printTime(start, end, "fitting the Naive Bayes")
	return clfB

def trainKNN(trainSet, trainLabels):
	start = time.time()
	clfN = KNeighborsClassifier()
	clfN.fit(trainSet, trainLabels)
	end = time.time()
	printTime(start, end, "fitting the KNeighbors")
	return clfN

def trainSVM(trainSet, trainLabels):
	start = time.time()
	clfS = svm.SVC(gamma = 0.001, C = 100.)
	# enabling probability takes a long time
	clfS.fit(trainSet, trainLabels)
	end = time.time()
	printTime(start, end, "fitting the SVM")
	return clfS

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

def printTime(start, end, action=""):
	m, s = divmod(end - start, 60)
	h, m = divmod(m, 60)
	
	print("Time elapsed " + action + ": {0:.0f}:{1:.0f}:{2:.2f}".format(h, m, s))

if __name__ == '__main__':
	main()