import numpy
from data_manager.file_manager import FileManager
from data_manager.feature_manager import FeatureManager

from sklearn import svm

number_of_directories = 10
number_of_features = 100

file_handle = FileManager()
feature_handle = FeatureManager()
clf = svm.SVC()

training_data = numpy.empty((0,0))
class_labels = []
for i in xrange(number_of_directories):
    data = file_handle.read_all_files('object_data/object_daylight/' + str(i+1) + '/', 'pcd')
    data = feature_handle.reduce_dimensions(data, 2)
    feature_vectors = feature_handle.generate_feature_vectors(data)

    for j in xrange(feature_vectors.shape[0]):
        class_labels.append(i+1)

    if len(training_data) == 0:
        training_data = feature_vectors
    else:
        training_data = numpy.vstack((training_data, feature_vectors))

class_labels = numpy.array(class_labels)
clf = clf.fit(training_data, class_labels)


test_data = numpy.empty((0,0))
test_class_labels = []
for i in xrange(number_of_directories):
    data = file_handle.read_all_files('object_data/object_lowlight/' + str(i+1) + '/', 'pcd')
    data = feature_handle.reduce_dimensions(data, 2)
    feature_vectors = feature_handle.generate_feature_vectors(data)

    for j in xrange(feature_vectors.shape[0]):
        test_class_labels.append(i+1)

    if len(test_data) == 0:
        test_data = feature_vectors
    else:
        test_data = numpy.vstack((test_data, feature_vectors))

test_class_labels = numpy.array(test_class_labels)

accuracy = 0.0
for i,label in enumerate(test_class_labels):
    prediction = clf.predict(test_data[i,:])
    if prediction == label:
        accuracy += 1.0
accuracy = accuracy / len(test_class_labels)