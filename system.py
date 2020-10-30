from utils import *

def diff_drive(x,u):
    xvel = [u[0]*np.cos(x[2]),
            u[0]*np.sin(x[2]),
            u[1]]
    return np.array(xvel).flatten()

def single_int(x,u):
    xvel = [u[0],
            u[1]]
    return np.array(xvel).flatten()

def double_int(x,u):
    xvel = [x[2],
            x[3],
            u[0],
            u[1]]
    return np.array(xvel).flatten()

def double_int_1D(x,u):
    xvel = [x[1],
            u[0]]
    return np.array(xvel).flatten()

def quadratic_objective(xvec,uvec,xdes=None,Q=None,R=None):
    if Q is None:
        Q = np.eye(xvec.shape[0])
    if R is None:
        R = np.eye(uvec.shape[0])
    if xdes is None:
        xd = np.zeros(xvec.shape)
    elif len(xdes.shape) == 1:
        xd = np.repeat(xdes.reshape(-1,1),xvec.shape[1],axis=1)

    c = 0
    for i in range(xvec.shape[1]):
        c+=(xvec[:,i]-xd[:,i]).dot(Q).dot((xvec[:,i]-xd[:,i]).T) + uvec[:,i].dot(R).dot(uvec[:,i].T)
    return c

def quadratic_rattling_objective(xvec, uvec, dt=0.05, w1=1, w2=1, coord_fun=None, w_sz=20, ov=1, xdes=None, Q=None, R=None):
    c = w1*quadratic_objective(xvec,uvec,xdes,Q,R)
    if coord_fun is None:
        r = rattling_windows(xvec.T, dt, w_sz, ov)[0]
        c += w2*np.mean(r)
    else:
        r = rattling_windows(coord_fun(xvec).T, dt, w_sz, ov)[0]
        c += w2*np.mean(r)
    return c

def gauss_pdf(x,mean,cov):
    return np.exp(-0.5*(x-mean).dot(np.linalg.inv(cov)).dot((x-mean).T))

def bimodal_objective(x,mean1,cov1,mean2,cov2):
    return -(gauss_pdf(x,mean1,cov1) + gauss_pdf(x,mean2,cov2)+1.0)

def bimodal_pdf(x,mean1,cov1,mean2,cov2):
    return (gauss_pdf(x,mean1,cov1) + gauss_pdf(x,mean2,cov2))/2.0

def bimodal_rattling_objective(xvec,uvec,mean1,cov1,mean2,cov2,dt=0.05, w1=1, w2=1, w_sz=20, ov=1,R=None):
    c1 = 0
    c2 = 0
    for i in range(xvec.shape[1]):
        c1 += -w1*(gauss_pdf(xvec[:,i],mean1,cov1) + gauss_pdf(xvec[:,i],mean2,cov2)+1.0)
        if R is not None:
            c1 += uvec[:,i].dot(R).dot(uvec[:,i].T)
    c2 = w2*np.mean(rattling_windows(xvec.T, dt, w_sz, ov)[0])
    return c1 + c2

def double_well_1D(x, a=1, b=1, xloc = 0.0):
    return a*(x[0]**4.0)-b*((x[0]-xloc)**2.0)

def double_well_objective_1D(xvec, uvec, R=None, a=1, b=1,xloc = 0.0):
    c = 0
    for i in range(xvec.shape[1]):
        c += double_well_1D(xvec[:,i],a,b,xloc)
        if R is not None:
            c += uvec[:,i].dot(R).dot(uvec[:,i].T)
    return c

def double_well_rattling_objective_1D(xvec, uvec, a=1, b=1, xloc = 0.0, dt=0.05, w=1, w_sz=20, ov=1,R=None):
    c = 0
    for i in range(xvec.shape[1]):
        c += double_well_1D(xvec[:,i],a,b,xloc)
        if R is not None:
            c += uvec[:,i].dot(R).dot(uvec[:,i].T)
    c += w*np.mean(rattling_windows(xvec.T, dt, w_sz, ov)[0])
    return c
