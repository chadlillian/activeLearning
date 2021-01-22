import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" 
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import numpy as np

import matplotlib.pyplot as plt
import matplotlib.tri as tri
from scipy.spatial import Delaunay

plt.rcParams["figure.figsize"] = [12, 12]
plt.rcParams["figure.dpi"] = 100

class activelearning:
    def __init__(self, cfdinst, anninst):
        self.cfdinst = cfdinst
        self.anninst = anninst

        self.x = []
        
        dp = 0.01
        self.params = np.meshgrid(np.arange(-1,1,dp),np.arange(-1,1,dp))
        self.params = np.vstack([np.reshape(p,[-1]) for p in self.params]).T
    
    def randomParams(self,n):
        self.params = np.random.rand(n,2)*2-1
    
    # [[mins, maxs]] 
    def setParameterEnvelope(self,minmax):
        # take an envelope and make all of the possible corners by taking all
        #  combinations of these coordinates
        p = np.meshgrid(*minmax)    # makes the cartesian product of all points in each set
        p = np.stack(p).reshape([minmax.shape[0],-1]).T   # reshapes this into a 2d list instead of an n dimensional matrix
        self.envelope = p

    def initialize(self):      
        self.x = self.envelope #np.array([[-1,-1],[-1,1],[1,-1],[1,1],[0,0]])
        self.y = []
        for xy in self.x:
            yy = self.cfdinst.run(xy)#[0],xy[1])
            self.y.append(yy)
        
        self.y = np.stack(self.y,0)
        self.anninst.train(self.x,self.y)
        self.anninst.var(self.params)

    #   selects the maximum variance center point of simplices constructed from self.params
    #       this should be more robust than just the max variance of the points in self.params
    #       because it will identify larger regions (an entire simplex) with higher average variance.
    #   zz list of variances at each point in self.params
    #   n number of new points to return
    def selectMaxTriangles(self,zz,n):

        # make the triangulation of the points in self.params
        triang = Delaunay(self.params)
        tris = triang.simplices
        # use the simplices (list of node indices for each simplex) to calculate the average value
        #   of the variance on each simplex
        #   zz[tris] returns a matrix with the variance of each node. 
        #   axis 1 is a list of all nodes in the simplex
        zt = np.mean(zz[tris],axis=1)
        
        iis = np.argsort(zt)[-n:] # get the triangles with highest variance
        mtris = tris[iis,:] # each row has the indices of a simplex
        xnext = np.mean(self.params[mtris],axis=1) # caluclate the center point of the simplices
        
        return xnext
    
    #   selects the maximum variance points from self.params
    def selectMaxPoints(self,zz,n):
        iis = np.argsort(zz)[-n:]
        xnext = self.params[iis,:]
        
        return xnext
        
    #   calculate variance at locations in the envelope from the BANN and then select the locations with the highest variance
    def findnextpoints(self,n):
        self.randomParams(1000) # TODO should include all points currently in the training set, make the number of randomparams set by the user and perhaps dynamically changed
        self.anninst.train(self.x,self.y)
        zz = self.anninst.var(self.params)

        xnext = self.selectMaxTriangles(zz,n)

        y = []
        for i in range(n):
            ynext = self.cfdinst.run(xnext[i,:])#0],xnext[i,1])
            y.append(ynext)

        y = np.stack(y,0)
        self.y = np.concatenate([self.y,y],axis=0)
        
        self.x = np.concatenate([self.x,xnext],axis=0)
    
    # TODO should make niters and nperiter more dynamic
    #   Main loop to run the BANN, select new points and rerun BANN
    #       niters how many loops to do
    #       nperiter, how many points to select per iteration
    def iterate(self,niters,nperiter=1):
        for i in range(niters):
            self.findnextpoints(nperiter)

    #   plot out the results
    def showresults(self):
        X,Y = np.meshgrid(np.arange(-1,1+.1,.1),np.arange(-1,1+.1,.1))
        xx = np.reshape(X,[-1])
        yy = np.reshape(Y,[-1])
        xy = np.vstack([xx,yy]).T
        
        triang = tri.Triangulation(self.params[:,0],self.params[:,1])

        zz = self.anninst.var(self.params)
        cf = plt.tricontourf(self.params[:,0],self.params[:,1],zz)
        plt.scatter(self.x[:,0],self.x[:,1],s=49,c='w')
        plt.scatter(self.x[:,0],self.x[:,1],s=16,c='k')
        plt.triplot(triang, lw=0.5, color='white')
        plt.colorbar(cf)
        plt.show()
