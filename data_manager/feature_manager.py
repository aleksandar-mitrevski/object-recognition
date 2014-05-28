import numpy
from sklearn.decomposition import PCA

class FeatureManager(object):
    def reduce_dimensions(self, data, dimension_number):
        pca_manager = PCA(n_components=dimension_number)
        new_data = []

        for _,point_cloud in enumerate(data):
            projected_cloud = pca_manager.fit_transform(point_cloud)
            new_data.append(projected_cloud)

        new_data = numpy.array(new_data)
        return new_data

    def generate_feature_vectors(self, data):
        feature_vectors = numpy.empty((0,0))
        for _,point_cloud in enumerate(data):
            vector = self.__generate_vector(point_cloud)
            if len(feature_vectors) == 0:
                feature_vectors = vector
            else:
                feature_vectors = numpy.vstack((feature_vectors, vector))
        return feature_vectors

    def __generate_vector(self, point_cloud):
        features = numpy.var(point_cloud, axis=0)
        features = features[numpy.newaxis]
        return features

    # def generate_feature_vectors(self, data, number_of_features):
        # feature_vectors = numpy.empty((0,0))
        # for _,point_cloud in enumerate(data):
            # if point_cloud.shape[0] > number_of_features:
                # vector = self.__generate_vector(point_cloud, number_of_features)
                # if len(feature_vectors) == 0:
                    # feature_vectors = vector
                # else:
                    # feature_vectors = numpy.vstack((feature_vectors, vector))
        # return feature_vectors

    # def __generate_vector(self, point_cloud, number_of_features):
        # number_of_points = point_cloud.shape[0]
        # mean_point = numpy.mean(point_cloud, axis=0)
        # covariance = numpy.cov(point_cloud.T)

        # features = []
        # selected_points = []
        # for i in xrange(number_of_features):
            # new_point_found = False
            # point_index = -1
            # while not new_point_found:
                # point_index = numpy.random.randint(0,number_of_points)
                # if point_index not in selected_points:
                    # selected_points.append(point_index)
                    # new_point_found = True
            # point = point_cloud[point_index,:]
            # new_feature = self.__multivariate_normal(point, mean_point, covariance)
            # features.append(new_feature)

        # features = numpy.array(features)
        # features = features[numpy.newaxis]
        # return features

    def __multivariate_normal(self, x, mean, cov):
        diff = x - mean
        diff = diff[numpy.newaxis].T
        alpha = 1. / numpy.sqrt(2 * numpy.pi * numpy.linalg.det(cov))
        result = numpy.exp(-0.5 * diff.T.dot(numpy.linalg.inv(cov).dot(diff)))
        return numpy.double(result)