from sklearn import linear_model
from sklearn.metrics import r2_score, mean_squared_error


def linear_regression(train, test, train_labels, test_labels):
	"""
	:param train:  training data (M x N)
	:param test: testing data (M1 x N)
	:param train_labels: training labels (real numbers, M x 1)
	:param test_labels: testing labels (real numbers, M1 x 1)
	:return: r2 (R2-score), mse (mean squared error)
	"""
	regr = linear_model.LinearRegression()
	regr.fit(train, train_labels)
	pred = regr.predict(test)
	r2 = r2_score(test_labels, pred)
	mse = mean_squared_error(test_labels, pred)
	return r2, mse