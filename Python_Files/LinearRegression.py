import numpy as np

class LinearRegression(object):
    
    def __init__(self, al=0.3,max_epoch=10):
        self.al = al
        self.max_epoch = max_epoch
    
    def fit(self, X, y):
        self.th0 = np.random.rand((1))[0]
        self.th = np.random.rand((len(X[0])))
        
        epoch = 1
        while epoch != self.max_epoch+1:
            self.GradientDescent(X, y)
            print 'Epoch : {0}\tloss : {1}'.format(epoch, self.loss(X,y,self.th0,self.th))
            epoch += 1
    
    def predict(self, X):
        self.pred = []
        for i in range(len(X)):
            self.pred.append(self.hypo(X[i]))
        return self.pred
    
    def hypo(self, x):
        return self.th0 + np.dot(self.th,x)
    
    def diffterm(self, X, y, th0, th, num):
        tot = 0
        m,n = X.shape
        if num == 1:
            for i in range(m):
                h = self.hypo(X[i])
                tot += (h-y[i])
        else:
            for i in range(m):
                xi = X[i][num]
                h = self.hypo(X[i])
                tot += (h-y[i])*xi
        
        return tot
    
    def loss(self, X, y, th0, th):
        tot = 0
        for i in range(len(X)):
            tot += (self.hypo(X[i]) - y[i])**2
        return tot
        
       
    def GradientDescent(self, X, y):
        self.temp = np.zeros((len(X[0])))
        self.temp0 = 0
        
        m, n = X.shape
        
        self.temp0 = self.th0 - (self.al/m) * self.diffterm(X,y,self.th0,self.th,1)
        
        for i in range(n):
            self.temp[i] = self.th[i] - (self.al/m)*self.diffterm(X,y,self.th0,self.th,i)
        
        self.th0 = self.temp0
        
        for i in range(n):
            self.th[i] = self.temp[i]