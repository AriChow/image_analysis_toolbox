from utils import classification, feature_extraction, feature_preprocessing, dimensionality_reduction, regression
import os
import glob
from sklearn.model_selection import StratifiedKFold
import numpy as np

dir_path = os.path.dirname(os.path.realpath(__file__))
dataset = 'matsc_dataset1'

l = os.listdir(dir_path + '/' + dataset)

# Make datasets
haralick = []
VGG = []
inception = []
Y = []
cnt = 0
for i in range(len(l)):
	l1 = l[i]
	if l1[0] == '.':
		continue
	names = glob.glob(dir_path + '/' + dataset + '/' + l1 + '/*.jpg')
	haralick += feature_extraction.haralick_features(names)
	VGG += feature_extraction.VGG(names)
	inception += feature_extraction.inception(names)
	Y += [cnt] * len(names)
	cnt += 1
Y = np.asarray(Y)

# Divide into training and testing data
skf = StratifiedKFold(n_splits=5)
a = []
f = []
p = []
r = []
rr = []
m = []
for train_index, test_index in skf.split(np.zeros((len(Y), 1)), Y):
	train_labels = Y[train_index]
	test_labels = Y[test_index]
	train = haralick[train_index] # Do the same for VGG / inception
	test = haralick[test_index]

	# Standardization
	train, test, _ = feature_preprocessing.preprocessing(train, test)

	# PCA
	train, test = dimensionality_reduction.principal_components(train, test)

	# Classification
	acc, f1, prec, rec, conf, probs, pred = classification.random_forests(train, test, train_labels, test_labels) # Try others the same way
	a.append(acc)
	f.append(f1)
	p.append(prec)
	r.append(rec)

	# Regression
	reg_train_labels = np.random.rand(len(train_labels), 1) # random outputs for regression, enter own outputs
	reg_test_labels = np.random.rand(len(test_labels), 1) # random outputs for regression, enter own outputs
	r2, mse = regression.linear_regression(train, test, reg_train_labels, reg_test_labels)
	rr.append(r2)
	m.append(mse)

# Output classification metrics
print("Accuracy: " + str(np.mean(a)) + '\n')
print("F1 - score: " + str(np.mean(f)) + '\n')
print("Precision: " + str(np.mean(p)) + '\n')
print("Recall: " + str(np.mean(r)) + '\n')
# Save confusion matrix, probabilities and predictions in csv files

# Output regression metrics
print("R2-score " + str(np.mean(rr)) + '\n')
print("Mean square error " + str(np.mean(m)) + '\n')


