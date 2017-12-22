from sklearn.decomposition import PCA


def principal_components(train_data, test_data, maxvar=0.95):
	"""
	:param train_data: training data (Matrix of size M x N), M is the number of training images and N is the number of test images.
	:param test_data: testing data (Matrix of size M x N)
	:param maxvar: Maximum variance explained by principal components.
	:return: train_pca(principal components of training data), test_pca(principle components of testing data)
	"""
	pca = PCA()
	train = train_data
	train_data = pca.fit(train_data)
	var = pca.explained_variance_ratio_
	sum = 0
	for i in range(len(var)):
		sum += var[i]
		if sum > maxvar:
			break
	pca = PCA(n_components=i)
	train_data = train
	train_pca = pca.fit_transform(train_data)
	test_pca = pca.transform(test_data)
	return train_pca, test_pca
