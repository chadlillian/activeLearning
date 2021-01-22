import numpy as np

class cfd:
    def __init__(self,dx):
        self.x,self.y = np.meshgrid(np.arange(-1,1,dx),np.arange(-1,1,dx))
        self.x = np.reshape(self.x,[-1])
        self.y = np.reshape(self.y,[-1])
        self.points = []
    
    def run(self,params):
        xi,yi = params
        ret = np.exp(.2*(-(self.x-xi)**2-(self.y-yi)**2))
        return ret

    def getMeshSize(self):
        return np.prod(self.x.shape)
