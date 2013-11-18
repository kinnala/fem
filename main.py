from meshtools import *
from assembly import *

mesh = square_mesh(20)
#mesh.visualize()

dudv = lambda u,v,ux,uy,vx,vy,x,y: ux*vx+uy*vy
K = bilin_asm_vec(dudv, mesh)
#Ko = bilin_asm(dudv, mesh)

#   fv = lambda v,vx,vy,x,y: 1.*v
#   f = lin_asm(fv, mesh)

#   N = mesh.p.shape[0]
#   A = np.arange(0,N)
#   D1 = np.nonzero(mesh.p[:,0]==0)[0]
#   D2 = np.nonzero(mesh.p[:,0]==1)[0]
#   D3 = np.nonzero(mesh.p[:,1]==0)[0]
#   D4 = np.nonzero(mesh.p[:,1]==1)[0]
#   D = np.union1d(np.union1d(np.union1d(D1,D2),D3),D4)
#   F = np.setdiff1d(A,D)

#   x = np.zeros(N)
#   x[F] = np.linalg.solve(K[np.ix_(F,F)],f[F])

#   if 0.0731 < np.max(x) and 0.0732 > np.max(x):
#       print "OK!"
#   else:
#       print "ERROR IN CODE!"
#       print np.max(x)

#mesh.visualize_data(x)

#draw_all()
