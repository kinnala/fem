import numpy as np
import matplotlib.pyplot as plt
import matplotlib.delaunay as triang
from mpl_toolkits.mplot3d import axes3d

class Trimesh:
    """Inverted trimesh structure. TODO: Add edge processing."""

    p = np.array([[]])
    t = np.array([[]])

    def __init__(self,p,t):
        self.p = p
        self.t = t

    def visualize(self):
        plt.figure()
        for tri in self.t:
            xpts = np.array([self.p[tri[k],0] for k in [0,1,2,0]])
            ypts = np.array([self.p[tri[k],1] for k in [0,1,2,0]])
            plt.plot(xpts,ypts,'k')
        plt.show()
            
def square_mesh(N):
    """Build an uniform triangular mesh on unit square."""
    xs,ys = np.meshgrid(np.linspace(0,1,N),np.linspace(0,1,N))
    xs = xs.flatten(1)
    ys = ys.flatten(1)
    _,_,t,_ = triang.delaunay(xs,ys)
    p = np.vstack((xs,ys)).T

    return Trimesh(p,t)

def trisurf(t,x,y,u):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    X,Y = np.meshgrid(x,y)
