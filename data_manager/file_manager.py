import numpy
import glob
import os

class FileManager(object):
    def read_all_files(self, directory_name, extension, columns_to_take):
        data = []
        for file_name in glob.glob(directory_name + '*.' + extension):
            temp_data = numpy.genfromtxt(file_name)
            data.append(temp_data[:,0:columns_to_take])

        data = numpy.array(data)
        return data