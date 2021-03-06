from abc import ABCMeta, abstractmethod
from skmultiflow.drift_detection.base_drift_detector import BaseDriftDetector
import numpy as np
from scipy import stats
from scipy.stats import f_oneway

class KSVEC(BaseDriftDetector):
    def __init__(self,vec_size=[1,1],alpha=0.9,p_alpha = 0.0001):
        self.alpha = alpha
        self.p_alpha = p_alpha
        self.mean_size = 50
        self.mean = np.zeros(vec_size)
        self.change_detected = False;
        self.p_value = None
        self.n = 0
        self.vec_size = vec_size
        pass

    def add_element(self, value):
        self.n +=value.shape[0]
        self.p_value = stats.ks_2samp(self.mean, np.mean(value,axis=0)).pvalue
        if self.n >= self.mean_size and self.p_value <= self.p_alpha:
            self.change_detected = True
            self.mean = self.expandingMean(value)
            self.n = 0
        else:
            self.change_detected = False
            self.mean = self.expandingMean(value)
        pass

    def expandingMean(self,value):
        for v in value:
            self.mean =  self.alpha*self.mean + (1-self.alpha)* v
        return self.mean
    # def incrementalMean(self,value):
    #     for v in value:
    #      self.mean =  self.mean + (1/self.n)*(v-self.mean)
    #     return self.mean

    def detected_change(self):
        return self.change_detected

    def reset(self):
        self.alpha = alpha
        self.mean = np.zeros(self.vec_size)
        self.change_detected = False;
        self.p_value = None
        self.n = 0
        pass
    
    def get_info(self):
         return "KSVEC Change: Current P-Value "+str(self.p_value)



if __name__ == '__main__':
    import numpy as np


    means = [0,5,0]
    dim = 10
    len(means)
    stream_size = 3000
    round(stream_size/len(means))
    data = np.concatenate([np.random.normal(elem,1,[round(stream_size/len(means)),dim]) for elem in means])

    ksvec = KSVEC(vec_size=data[1].size,alpha=0.6,p_alpha=0.0001)

    for i in range(len(data)):
        ksvec.add_element(data[i])
        if ksvec.get_change():
            print("\rIteration {}".format(i+1))
            print(ksvec.get_info())