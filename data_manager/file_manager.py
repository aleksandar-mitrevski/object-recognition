import numpy
import glob
import os

class FileManager(object):
    def read_all_files(self, directory_name, extension):
        data = []
        for file_name in glob.glob(directory_name + '*.' + extension):
            temp_data = numpy.genfromtxt(file_name)
            data.append(temp_data[:,0:3])

        data = numpy.array(data)
        return data

# data = numpy.empty((0,0))
# for file_name in glob.glob('directory_name/*.' + extension):
    # if len(data) == 0:
        # data = numpy.genfromtxt(file_name)
    # else:
        # temp_data = numpy.genfromtxt(file_name)
        # data.vstack((data,temp_data))