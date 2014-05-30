import numpy
from data_manager.file_manager import FileManager
from data_manager.feature_manager import FeatureManager

from sklearn import svm

from sklearn.naive_bayes import GaussianNB

from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.tools.shortcuts import buildNetwork

def label_mappings(label, real_class):
    if (real_class == 1 and label == 2) or (real_class == 2 and label == 1):
        return True
    if (real_class == 8 and label == 9) or (real_class == 9 and label == 8):
        return True
    return False

number_of_directories = 10
number_of_features = 100

file_handle = FileManager()
feature_handle = FeatureManager()

#loading training data
training_data = numpy.empty((0,0))
class_labels = []
file_columns_to_read = 3
for i in xrange(number_of_directories):
    data = file_handle.read_all_files('object_data/object_daylight/' + str(i+1) + '/', 'pcd', file_columns_to_read)
    data = feature_handle.reduce_dimensions(data, 2)
    feature_vectors = feature_handle.generate_feature_vectors(data)

    for j in xrange(feature_vectors.shape[0]):
        class_labels.append(i+1)

    if len(training_data) == 0:
        training_data = feature_vectors
    else:
        training_data = numpy.vstack((training_data, feature_vectors))

class_labels = numpy.array(class_labels)

#SVM training
svm_classifier = svm.SVC()
svm_classifier = svm_classifier.fit(training_data, class_labels)

#neural network training
number_of_input_units = 2
number_of_output_units = 10
number_of_hidden_units = 25#2 * (number_of_input_units + number_of_output_units) / 3
dataset = SupervisedDataSet(number_of_input_units, number_of_output_units)
for i in xrange(training_data.shape[0]):
    dataset.addSample(training_data[i,:], class_labels[i])
network = buildNetwork(number_of_input_units, number_of_hidden_units, number_of_output_units, bias=True)
trainer = BackpropTrainer(network, dataset)
trainer.trainUntilConvergence(maxEpochs=1000)
network = trainer.module

#naive Bayes training
naive_bayes_classifier = GaussianNB()
naive_bayes_classifier = naive_bayes_classifier.fit(training_data, class_labels)

#loading test data
test_data = numpy.empty((0,0))
test_class_labels = []
file_columns_to_read = 3
for i in xrange(number_of_directories):
    data = file_handle.read_all_files('object_data/object_lowlight/' + str(i+1) + '/', 'pcd', file_columns_to_read)
    data = feature_handle.reduce_dimensions(data, 2)
    feature_vectors = feature_handle.generate_feature_vectors(data)

    for j in xrange(feature_vectors.shape[0]):
        test_class_labels.append(i+1)

    if len(test_data) == 0:
        test_data = feature_vectors
    else:
        test_data = numpy.vstack((test_data, feature_vectors))

test_class_labels = numpy.array(test_class_labels)

#calculating classification accuracy
svm_accuracy = 0.0
network_accuracy = 0.0
naive_bayes_accuracy = 0.0
for i,label in enumerate(test_class_labels):
    svm_prediction = svm_classifier.predict(test_data[i,:])
    network_prediction = numpy.argmax(network.activate(test_data[i,:]))
    naive_bayes_prediction = naive_bayes_classifier.predict(test_data[i,:])
    if svm_prediction == label or label_mappings(svm_prediction, label):
        svm_accuracy += 1.0
    if network_prediction == label or label_mappings(network_prediction, label):
        network_accuracy += 1.0
    if naive_bayes_prediction == label or label_mappings(naive_bayes_prediction, label):
        naive_bayes_accuracy += 1.0
svm_accuracy = svm_accuracy / len(test_class_labels)
network_accuracy = network_accuracy / len(test_class_labels)
naive_bayes_accuracy = naive_bayes_accuracy / len(test_class_labels)