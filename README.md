# imagelearning
run the flowerlearning.py script to initiate an svm model to differentiate flower images
images from https://codelabs.developers.google.com/codelabs/tensorflow-for-poets/

goal:
explore machine learning via experimentation by attempting to train a flower image classifier

notes:
SVM does not work well for this flower image classifier

Naive Bayes
- features should be independant
- computes probability of classification using Bayes theorem
- intuitively not suitable for images

Decision Trees
- creates subgroups based on path of features in a tree
- intuitively not suitable for images
- seen as it takes 10x the length to fit the model and yields poor results

Applying PCA may help in performance. PCA reduces the dimensionality of the data to
"relevant" components
- PCA with no components specified gets killed
- post PCA (n=20) results:

Time elapsed transforming feature set: 0:7:45.06
Time elapsed fitting the SVM: 0:0:0.30
Time elapsed fitting the KNeighbors: 0:0:0.01
Time elapsed fitting the Decision Tree: 0:0:0.05
Time elapsed fitting the Naive Bayes: 0:0:0.00
Time elapsed fitting the MLP: 0:0:0.40
SVM
0.0
K Nearest Neighbors
1.0
Decision Tree
0.8
Naive Bayes
0.8
MLP
1.0
-----------------------
SVM
0.0
K Nearest Neighbors
0.2
Decision Tree
0.0
Naive Bayes
0.4
MLP
0.2
-----------------------
SVM
1.0
K Nearest Neighbors
0.6
Decision Tree
0.6
Naive Bayes
1.0
MLP
0.4
