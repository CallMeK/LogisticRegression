#logistic regression 

#There will be three mode
#SGD
#Batch with GD
#Batch with several different implemented optimizer

import numpy as np
import scipy
import sys
#basic class 
class RegLog(object):
	"""
	Basic class
	Training data load
	Abstract training
	Blind Test
	Test score
	"""

	def __init__(self,inputs,response,intercept=True,epoch=100,tol=0.01, mode='fix_epoch',alpha=0.1,C=1.0):
		#inputs is a matrix of (N,d), N is the number of the data and d is the dimension of features
		self.N = inputs.shape[0]
		self.d = inputs.shape[1]
		if(intercept):
			self.inputs = np.hstack([inputs,np.ones(self.N)[:,np.newaxis]]);
			self.d+=1
		self.epoch = epoch
		self.tol = tol
		self.mode = mode
		self.alpha = alpha
		self.C = C
		self.response = response
		self.class_list = sorted(list(set(self.response))) #the layout of the output layer
		self.class_index = [self.class_list.index(x) for x in self.response] #which node should be 1
		self.nclass = len(self.class_list)
		self.class_response_like = np.zeros([self.N,self.nclass]) #The supposed output of all nodes
		for row, i in zip(self.class_response_like, self.class_index):
			row[i] = 1 #The supposed output of all nodes
		print self.class_response_like

	def batch_forward(self, inputs = None):
		if(inputs is None):
			return self.trans_fcn(np.dot(self.inputs,self.weights))
		else:
			#for test set
			return self.trans_fcn(np.dot(inputs,self.weights))

	def single_forward(self,i):
		return self.trans_fcn(np.dot(self.inputs[i],self.weights))

	@staticmethod
	def get_predicted_class_index(probs):
		#it only gives out the index of the result in self.class_list
		#The prediction should be of N,nclass
		return np.argmax(probs,axis=1)

	@staticmethod
	def trans_fcn(x):
		return 1/(1+np.exp(-x))

	@staticmethod
	def trans_fcn_dev(y):
		return y*(1-y)

	def compute_CEloss(self, inputs=None, response=None):
		# -sum(prob of the actual class)
		# use the training class list in case some classes are absent
		if(inputs is None and response is None):
			#train
			probs = self.batch_forward()
			return -sum(np.log([probs[i,self.class_index[i]] for i in xrange(self.N)]))
		else:
			#test
			probs = self.batch_forward(inputs)
			class_index = [self.class_list.index(x) for x in response]
			return -sum(np.log([probs[i,class_index[i]] for i in xrange(inputs.shape[0])]))

	def compute_error(self,result=None,response=None):
		import operator as op
		if(result is None and response is None):
			#train
			return sum(map(op.__eq__,self.predict(),self.response))/float(self.N)
		else:
			N = response.shape[0]
			assert result.shape[0] == N
			return sum(map(op.__eq__,result,response))/float(N)


	def predict(self, inputs=None):
		if(inputs == None):
			probs = self.batch_forward()
			result = [self.class_list[i] for i in self.get_predicted_class_index(probs)]
		else:
			#test
			probs = self.batch_forward(inputs)
			result = [self.class_list[i] for i in self.get_predicted_class_index(probs)]
		return np.array(result)


	def train_report(self):
		return (result,
			self.compute_CEloss(),
			self.compute_error())



class Batch_RegLog(RegLog):
	def CEloss_grad(self):
		#only valid for logistic function
		out = self.batch_forward() #y
		diff_mat = out - self.class_response_like #1 or 0
		return sum(map(np.outer,self.inputs, diff_mat))

	def train(self):
		self.weights = np.random.rand(self.d,self.nclass)-0.5
		i=0
		while(i<=self.epoch):
			step = -self.alpha*self.CEloss_grad()
			self.weights+= step/self.N
			i+=1
			print i, self.compute_CEloss()/self.N
		print self.compute_error()


def ExtractData(filename,QuickTest=True):
		import csv
		f = open(filename,'Urb')
		spamreader = csv.reader(f,delimiter=" ")
		datalist = list(spamreader)
		if(QuickTest):
			datalist = datalist[:100]
		digit_data_raw = np.array(datalist) #numpy array
		digit_data_response = np.array([item.split('.')[0] for item in digit_data_raw[:,0]])
		digit_data = digit_data_raw[:,1:-1].astype(float)
		return (digit_data_response,digit_data)




if __name__== "__main__":
	response, inputs = ExtractData('ZipDigits.test.csv',False)
	model = Batch_RegLog(inputs,response,alpha=1.0,epoch=500)
	model.train();



