from sklearn.preprocessing import StandardScaler, MinMaxScaler


def preprocessing(train, test, normalize=False, standardize=True):
	"""
	:param train: training data (Matrix of size M x N), M is the number of training images and N is the number of test images.
	:param test: testing data (Matrix of size M x N)
	:param normalize: normalizing (boolean)
	:param standardize: standardizing (boolean, default)
	:return: train_data (normalized or standardized training data), test_data (normalized or standardized testing data), , object for performing normalization
	or standardization on new data.
	"""
	if standardize == True and normalize == False:
		stdSlr = StandardScaler().fit(train)
		train_data = stdSlr.transform(train)
		test_data = stdSlr.transform(test)
	else:
		stdSlr = MinMaxScaler().fit(train)
		train_data = stdSlr.transform(train)
		test_data = stdSlr.transform(test)
	return train_data, test_data, stdSlr
