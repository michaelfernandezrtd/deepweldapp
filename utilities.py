import numpy as np


class minmax:

    def __init__(self, parameter=None):
        self.parameters=None

    def copy(self):
        return minmax()

    def map_data(self, data):
        max = np.amax(data,axis=0)
        min = np.amin(data,axis=0)
        #print "Inside maxmin",max,min
    #for idx in range(data.shape[1]):
        #    datanormalize[:,idx]=2*(data[:,idx]-min[idx])/(max[idx]-min[idx]) - 1
        datanormalize = 2*(data-min)/(max-min) - 1
        self.parameters = [max,min]
        return datanormalize

    def remap_data(self,data):

        '''Check if normalizing only label (1-D array) or all data matrix features and labels '''

        if len(data.shape)==1 :
            max=self.parameters[0][-1]
            min=self.parameters[1][-1]
            #print "MaxMin",max,min
        else:
            max=self.parameters[0]
            min=self.parameters[1]
            #print "MaxMin",max,min
            #print data
        databacknormalize=((data+1)*(max-min))/2 + min
        #print databacknormalize
        return  databacknormalize

    def transmap_data(self,data):
        #print np.asmatrix(data).shape
    #print self.parameters
        if len(data.shape)==1 :
            max=self.parameters[0][-1]
            min=self.parameters[1][-1]
        else:
            max=self.parameters[0]
            min=self.parameters[1]

        #print "MaxMin",max,min
        #for idx in range(data.shape[1]):
        #    datanormalize[:,idx]=2*(data[:,idx]-min[idx])/(max[idx]-min[idx]) - 1
        datanormalize=2*(data-min)/(max-min) - 1
        return  datanormalize

    def copy(self):
        return  minmax()

    def get_parameters(self):
        return self.parameters
