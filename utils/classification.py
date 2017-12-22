from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
from sklearn import svm, ensemble
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
import numpy as np

val_split = 3  # Number of validation splits


def logreg(train, test, train_labels, test_labels):
	"""
	:param train: training data (Matrix of size M x N), M is the number of training images and N is the number of test images.
	:param test: testing data (Matrix of size M x N)
	:param train_labels: training labels (Matrix of size M x 1)
	:param test_labels: testing labels (Matrix of size M x 1)
	:return: outputs from fit_classifier (accuracy, f1 score, precision, recall, confusion matrix, probability of prediction, prediction)
	"""
	C = [10*i for i in range(-3,2)]
	params={'C': C}
	clf = GridSearchCV(LogisticRegression(class_weight='balanced'),params,cv=val_split)
	return fit_classifier(clf, train, train_labels, test, test_labels)


def random_forests(train, test, train_labels, test_labels):
	"""
	:param train: training data (Matrix of size M x N), M is the number of training images and N is the number of test images.
	:param test: testing data (Matrix of size M x N)
	:param train_labels: training labels (Matrix of size M x 1)
	:param test_labels: testing labels (Matrix of size M x 1)
	:return: outputs from fit_classifier (accuracy, f1 score, precision, recall, confusion matrix, probability of prediction, prediction)
	"""
	estimators = [50*i for i in range(4, 6)]
	parameters = {'n_estimators': estimators}
	clf = GridSearchCV(ensemble.RandomForestClassifier(class_weight='balanced'), parameters, cv=val_split)
	return fit_classifier(clf, train, train_labels, test, test_labels)


def SVM(train, test, train_labels, test_labels, kern='linear'):
	"""
	:param train: training data (Matrix of size M x N), M is the number of training images and N is the number of test images.
	:param test: testing data (Matrix of size M x N)
	:param train_labels: training labels (Matrix of size M x 1)
	:param test_labels: testing labels (Matrix of size M x 1)
	:return: outputs from fit_classifier (accuracy, f1 score, precision, recall, confusion matrix, probability of prediction, prediction)
	"""
	gamma = [10**i for i in range(-5,0)]
	C = np.linspace(0.01, 100, 30)
	parameters = {'kernel': [kern], 'C': C.tolist(), 'gamma': gamma}
	clf = GridSearchCV(svm.SVC(probability=True), parameters, cv=val_split)
	return fit_classifier(clf, train, train_labels, test, test_labels)


def fit_classifier(clf, train, train_labels, test, test_labels):
	"""
	:param clf: classifier (from logistic regression, SVM or random forests
	:param train: training data (Matrix of size M x N), M is the number of training images and N is the number of test images.
	:param test: testing data (Matrix of size M x N)
	:param train_labels: training labels (Matrix of size M x 1)
	:param test_labels: testing labels (Matrix of size M x 1)
	:return: acc (accuracy), f1 (f1-score), prec (precision), rec (recall), conf (confusion matrix), probs (probability), pred (prediction)
	"""
	clf.fit(train, train_labels)
	pred = clf.predict(test)
	pred = pred.ravel()
	pred = pred.tolist()
	test_labels = test_labels.ravel()
	act_labels = test_labels.tolist()
	acc = accuracy_score(act_labels,pred)
	f1 = f1_score(act_labels,pred,average='weighted')
	prec = precision_score(act_labels,pred,average='weighted')
	rec = recall_score(act_labels,pred,average='weighted')
	conf = confusion_matrix(act_labels,pred)
	probs = clf.predict_proba(test)
	return acc, f1, prec, rec, conf, probs, pred
