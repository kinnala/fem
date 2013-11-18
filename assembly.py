import numpy as np
from meshtools import *
from scipy.sparse import coo_matrix

import ipdb

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

def affine_map_vec(mesh):
    """Build affine mappings for all triangles."""

    A = {0:{},1:{}}

    A[0][0] = mesh.p[mesh.t[:,1],0]-mesh.p[mesh.t[:,0],0]
    A[0][1] = mesh.p[mesh.t[:,2],0]-mesh.p[mesh.t[:,0],0]
    A[1][0] = mesh.p[mesh.t[:,1],1]-mesh.p[mesh.t[:,0],1]
    A[1][1] = mesh.p[mesh.t[:,2],1]-mesh.p[mesh.t[:,0],1]

    b = {}

    b[0] = mesh.p[mesh.t[:,0],0]
    b[1] = mesh.p[mesh.t[:,0],1]

    detA = A[0][0]*A[1][1]-A[0][1]*A[1][0]

    invA = {0:{},1:{}}

    invA[0][0] = A[1][1]/detA
    invA[0][1] = -A[0][1]/detA
    invA[1][0] = -A[1][0]/detA
    invA[1][1] = A[0][0]/detA

    return A,b,detA,invA

def affine_map(tri, mesh):
    """Build the affine mapping for 'tri'."""
    n = mesh.p[tri]
    A = np.vstack((n[1]-n[0],n[2]-n[0])).T
    b = n[0]

    return A,b

def lin_basis_vec(qp):
    """Return the values and gradients of linear
    reference triangle basis functions at local
    quadrature points (qp)."""
    phi = {}

    phi[0] = 1.-qp[:,0]-qp[:,1]
    phi[1] = qp[:,0]
    phi[2] = qp[:,1]

    gradphi = {}

    gradphi[0] = np.tile(np.array([-1.,-1.]),(qp.shape[0],1))
    gradphi[1] = np.tile(np.array([1.,0.]),(qp.shape[0],1))
    gradphi[2] = np.tile(np.array([0.,1.]),(qp.shape[0],1))

    return phi, gradphi

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

def bilin_asm_vec(bilin, mesh):
    """Assembly the stiffness matrix (vectorized)."""
    N = mesh.p.shape[0]
    T = mesh.t.shape[0]
    K = np.zeros((N,N))

    qp,qw = triquad(dunavant2)
    phi,gradphi = lin_basis_vec(qp)
    A,b,detA,invA = affine_map_vec(mesh)

    data = np.array([])
    rows = np.array([])
    cols = np.array([])

    for i in [0,1,2]:
        u = np.tile(phi[i],(T,1))
        ux = np.outer(invA[0][0],gradphi[i][:,0])+np.outer(invA[1][0],gradphi[i][:,1])
        uy = np.outer(invA[0][1],gradphi[i][:,0])+np.outer(invA[1][1],gradphi[i][:,1])
        for j in range(0,i+1):
            v = np.tile(phi[j],(T,1))
            vx = np.outer(invA[0][0],gradphi[j][:,0])+np.outer(invA[1][0],gradphi[j][:,1])
            vy = np.outer(invA[0][1],gradphi[j][:,0])+np.outer(invA[1][1],gradphi[j][:,1])
            
            #K[i,j] += np.dot(qw,bilin(u,v,ux,uy,vx,vy,0,0))*np.abs(detA)
            #ipdb.set_trace()
            x = np.dot(bilin(u,v,ux,uy,vx,vy,0,0),qw)*np.abs(detA)
            data = np.append(data,x)
            rows = np.append(rows,mesh.t[:,i])
            cols = np.append(cols,mesh.t[:,j])
            if i != j:
                data = np.append(data,x)
                rows = np.append(rows,mesh.t[:,j])
                cols = np.append(cols,mesh.t[:,i])
            #ipdb.set_trace()
                #ipdb.set_trace()

    #    K[np.ix_(tri,tri)] += Ke
    return coo_matrix((data,(rows,cols)), shape=(N,N)).todense()

def bilin_asm(bilin, mesh):
    """Assembly the stiffness matrix."""
    N = mesh.p.size/2
    K = np.zeros((N,N))

    qp,qw = triquad(dunavant2)
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

                    ipdb.set_trace()

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
