import numpy as np
from meshtools import *

dunavant1 = np.array([
    3.333333333333333333333e-01,
    3.333333333333333333333e-01,
    5.000000000000000000000e-01])

dunavant2 = np.array([
    1.666666666666666666666e-01,
    6.666666666666666666666e-01,
    1.666666666666666666666e-01,
    6.666666666666666666666e-01,
    1.666666666666666666666e-01,
    1.666666666666666666666e-01,
    1.666666666666666666666e-01,
    1.666666666666666666666e-01,
    1.666666666666666666666e-01])

def triquad(d):
    """Return Dunavant quadrature rules."""
    n = d.size/3
    return np.vstack((d[0:n],d[n:2*n])).T, d[2*n:3*n]

def affine_map(tri, mesh):
    """Build the affine mapping for 'tri'."""
    n = mesh.p[tri]
    A = np.vstack((n[1]-n[0],n[2]-n[0])).T
    b = n[0]

    return A,b

def lin_basis(qp):
    """Return the values and gradients of linear
    reference triangle basis functions at local
    quadrature points (qp)."""
    phi = {}

    phi[0] = 1.-qp[:,0]-qp[:,1]
    phi[1] = qp[:,0]
    phi[2] = qp[:,1]

    gradphi = {}

    gradphi[0] = np.array([-1.,-1.])
    gradphi[1] = np.array([1.,0.])
    gradphi[2] = np.array([0.,1.])

    return phi, gradphi

def bilin_asm(bilin, mesh):
    """Assembly the stiffness matrix."""
    N = mesh.p.size/2
    K = np.zeros((N,N))

    qp,qw = triquad(dunavant1)
    phi,gradphi = lin_basis(qp)

    for tri in mesh.t:
        # Local stiffness matrix
        Ke = np.zeros((3,3))
        # Affine mapping for 'tri'
        A,b = affine_map(tri, mesh)
        
        detA = np.linalg.det(A)
        invA = np.linalg.inv(A)

        for i in [0,1,2]:
            for j in [0,1,2]:
                for ix,w in enumerate(qw): 
                    u = phi[i][ix]
                    v = phi[j][ix]
                    ux,uy = np.dot(invA.T,gradphi[i])
                    vx,vy = np.dot(invA.T,gradphi[j])
                    Ke[i,j] += w*bilin(u,v,ux,uy,vx,vy,0,0)*np.abs(detA)

        K[np.ix_(tri,tri)] += Ke

    return K

def lin_asm(lin,mesh):
    """Assembly the load vector."""
    N = mesh.p.size/2
    f = np.zeros(N)

    qp,qw = triquad(dunavant1)
    phi,gradphi = lin_basis(qp)

    for tri in mesh.t:
        fe = np.zeros(3)

        A,b = affine_map(tri, mesh)
        
        detA = np.linalg.det(A)
        invA = np.linalg.inv(A)

        for i in [0,1,2]:
            for ix,w in enumerate(qw): 
                v = phi[i][ix]
                vx,vy = np.dot(invA.T,gradphi[i])

                fe[i] += w*lin(v,vx,vy,0,0)*np.abs(detA)

        f[tri] += fe

    return f
