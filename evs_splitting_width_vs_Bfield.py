import numpy as np
from numpy import linalg as LA # used in eigenvalue function
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import datetime
from scipy.optimize import curve_fit # used in swth function

# This code simulates:
# (1) How the splitting of the 4 NV-center orientations depend on the direction of a 
# 150uT magnetic field (see plotsplittings)
# and
# (2) How the splitting and width of a two-lorentzian fit depend on the direction
# of a 200uT magnetic field (see swth)
# Note: the swth simulation can accomodate an array of magnetic field magnitudes
# in addition to theta and phi values, but it takes A LOT time

# Generate phase space array of theta, phi, and Bmag
def s2c(theta,phi,Bmag):
    thetaphi = np.array([[x0,y0] for x0 in theta \
                      for y0 in phi])
    theta,phi = thetaphi.transpose()
    xyz = (np.array([np.cos(phi)*np.sin(theta),\
               np.sin(phi)*np.sin(theta),\
               np.cos(theta)])).transpose()
    Bxyztmp = (Bmag*xyz).reshape(len(Bmag),len(thetaphi),1,3)
    return Bxyztmp


# Calculate the eigenvalues for phase space
def eigenvalues(Bxyz,zf):
    Bxyztmp = np.copy(Bxyz)
    base = np.ones((len(Bxyztmp)*len(Bxyztmp[0]),1)).\
           reshape(len(Bxyztmp),len(Bxyztmp[0]),1,1)
    zfs1 = zf[0]*(base*np.array([[0,.3333,.3333],\
                    [.3333,0,-.3333],\
                    [.3333,-.3333,0]]))
    strain1 = zf[1]*(base*np.array([[.3333,-.3333,.6667],\
                    [-.3333,-.6667,.3333],\
                    [.6667,.3333,.3333]]))
    zfs2 = zf[0]*(base*np.array([[0,.3333j,-.3333],\
                    [-.3333j,0,-.3333j],\
                    [-.3333,.3333j,0]]))
    strain2 = zf[1]*(base*np.array([[.3333,-.3333j,-.6667],\
                    [.3333j,-.6667,.3333j],\
                    [-.6667,-.3333j,.3333]]))
    zfs3 = zf[0]*(base*np.array([[0,-.3333,.3333],\
                    [-.3333,0,.3333],\
                    [.3333,.3333,0]]))
    strain3 = zf[1]*(base*np.array([[.3333,.3333,.6667],\
                    [.3333,-.6667,-.3333],\
                    [.6667,-.3333,.3333]]))
    zfs4 = zf[0]*(base*np.array([[0,-.3333j,-.3333],\
                    [.3333j,0,.3333j],\
                    [-.3333,-.3333j,0]]))
    strain4 = zf[1]*(base*np.array([[.3333,.3333j,-.6667],\
                    [-.3333j,-.6667,-.3333j],\
                    [-.6667,.3333j,.3333]]))
    sx = base*np.array([[0,.7071,0],[.7071,0,.7071],[0,.7071,0]])
    sy = base*np.array([[0,-.7071j,0],[-.7071j,0,-.7071j],[0,-.7071j,0]])
    sz = base*np.array([[1,0,0],[0,0,0],[0,0,-1]])
    zeeman = 28024951642 * ((Bxyztmp[:,:,:,0].reshape(len(Bxyztmp),\
                                        len(Bxyztmp[0]),1,1)*sx) + \
                            (Bxyztmp[:,:,:,1].reshape(len(Bxyztmp),\
                                        len(Bxyztmp[0]),1,1)*sy) + \
                            (Bxyztmp[:,:,:,2].reshape(len(Bxyztmp),\
                                        len(Bxyztmp[0]),1,1)*sz))
    # NOTE: The eigh function sorts the eigenvalues in increasing order
    # this leads to an error in the calculation of the splittings:
    # it is not currently possible to distinguish between B parallel to NV
    # and B antiparallel to NV
    w, v = LA.eigh(zfs1+strain1+zeeman)
    f1 = w[:,:,1] - w[:,:,0]
    f2 = w[:,:,2] - w[:,:,0]
    w, v = LA.eigh(zfs2+strain2+zeeman)
    f3 = (abs(w[:,:,0])+abs(w[:,:,1]))
    f4 = (abs(w[:,:,0])+abs(w[:,:,2]))
    w, v = LA.eigh(zfs3+strain3+zeeman)
    f5 = (abs(w[:,:,0])+abs(w[:,:,1]))
    f6 = (abs(w[:,:,0])+abs(w[:,:,2]))
    w, v = LA.eigh(zfs4+strain4+zeeman)
    f7 = (abs(w[:,:,0])+abs(w[:,:,1]))
    f8 = (abs(w[:,:,0])+abs(w[:,:,2]))
    evals = np.dstack([f1,f2,f3,f4,f5,f6,f7,f8]).reshape(len(Bxyztmp),\
                                len(Bxyztmp[0]),1,8)
##    evals = np.sort(evals,axis=2).reshape(len(Bxyztmp),\
##                                len(Bxyztmp[0]),1,8)
##    return np.concatenate((Bxyztmp,evals),axis=3)
    return evals

def splittings(evals):
    base = np.ones((len(evals),len(evals[0]),1,4))
    base[:,:,:,0] = evals[:,:,:,1] - evals[:,:,:,0]
    base[:,:,:,1] = evals[:,:,:,3] - evals[:,:,:,2]
    base[:,:,:,2] = evals[:,:,:,5] - evals[:,:,:,4]
    base[:,:,:,3] = evals[:,:,:,7] - evals[:,:,:,6]
    return base

# Generate spectra in phase space
def lor8(freq,zf,ev):
    evtmp = np.copy(ev)
    base = np.ones((len(evtmp)*len(evtmp[0]),1)).\
           reshape(len(evtmp),len(evtmp[0]),1,1)
    freq = freq*base
    ev1 = (ev[:,:,:,0].reshape(len(evtmp),len(evtmp[0]),1,1))\
            *base
    ev2 = (ev[:,:,:,1].reshape(len(evtmp),len(evtmp[0]),1,1))\
            *base
    ev3 = (ev[:,:,:,2].reshape(len(evtmp),len(evtmp[0]),1,1))\
            *base
    ev4 = (ev[:,:,:,3].reshape(len(evtmp),len(evtmp[0]),1,1))\
            *base
    ev5 = (ev[:,:,:,4].reshape(len(evtmp),len(evtmp[0]),1,1))\
            *base
    ev6 = (ev[:,:,:,5].reshape(len(evtmp),len(evtmp[0]),1,1))\
            *base
    ev7 = (ev[:,:,:,6].reshape(len(evtmp),len(evtmp[0]),1,1))\
            *base
    ev8 = (ev[:,:,:,7].reshape(len(evtmp),len(evtmp[0]),1,1))\
            *base
    lor1 = 1e7*((zf[2]**2)/\
            (np.pi*zf[2]*((freq-ev1)**2+ zf[2]**2)))
    lor2 = 1e7*((zf[2]**2)/\
            (np.pi*zf[2]*((freq-ev2)**2+ zf[2]**2)))
    lor3 = 1e7*((zf[2]**2)/\
            (np.pi*zf[2]*((freq-ev3)**2+ zf[2]**2)))
    lor4 = 1e7*((zf[2]**2)/\
            (np.pi*zf[2]*((freq-ev4)**2+ zf[2]**2)))
    lor5 = 1e7*((zf[2]**2)/\
            (np.pi*zf[2]*((freq-ev5)**2+ zf[2]**2)))
    lor6 = 1e7*((zf[2]**2)/\
            (np.pi*zf[2]*((freq-ev6)**2+ zf[2]**2)))
    lor7 = 1e7*((zf[2]**2)/\
            (np.pi*zf[2]*((freq-ev7)**2+ zf[2]**2)))
    lor8 = 1e7*((zf[2]**2)/\
            (np.pi*zf[2]*((freq-ev8)**2+ zf[2]**2)))
    return lor1+lor2+lor3+lor4+lor5+lor6+lor7+lor8





############################################
# Define theta, phi, Bmag arrays to calculate eigenvalues
############################################
##theta = np.linspace(0.00,np.pi,num=40)
##phi = np.linspace(0,2*np.pi,num=40)
####theta = np.array([0,np.pi/2])
####phi = np.array([0])
####Bmag = np.linspace(0,150e-6,2)
##Bmag = np.array([150e-6])
##Bmag = Bmag.reshape(len(Bmag),1,1)
##Bxyz = s2c(theta,phi,Bmag)
##ev = eigenvalues(Bxyz)

############################################
# Calculate and plot NV splittings as a function of theta and phi
# Requires single value for Bmag (i.e.: Bmag = np.array([150e-6]))
############################################
def plotsplittings(phi,theta,zf):
    Bmag = np.array([150e-6])
    Bmag = Bmag.reshape(len(Bmag),1,1)
    Bxyz = s2c(theta,phi,Bmag)
    ev = eigenvalues(Bxyz,zf)
    s = splittings(ev)

    fig = plt.figure(figsize=plt.figaspect(1.))

    ax = fig.add_subplot(221)
    X = np.degrees(phi)
    Y = np.degrees(theta)
    X, Y = np.meshgrid(X,Y)
    Z = s[:,:,:,0].reshape(len(theta),len(phi))
    plt.contourf(X,Y,Z,100)
    plt.colorbar()
    ax.set_title('NV-axis: (1,0,1)')
    ax.set_xlabel('phi (degrees)')
    ax.set_ylabel('theta (degrees)')

    ax = fig.add_subplot(222)
    X = np.degrees(phi)
    Y = np.degrees(theta)
    X, Y = np.meshgrid(X,Y)
    Z = s[:,:,:,1].reshape(len(theta),len(phi))
    plt.contourf(X,Y,Z,100)
    plt.colorbar()
    ax.set_title('NV-axis: (0,-1,-1)')
    ax.set_xlabel('phi (degrees)')
    ax.set_ylabel('theta (degrees)')

    ax = fig.add_subplot(223)
    X = np.degrees(phi)
    Y = np.degrees(theta)
    X, Y = np.meshgrid(X,Y)
    Z = s[:,:,:,2].reshape(len(theta),len(phi))
    plt.contourf(X,Y,Z,100)
    plt.colorbar()
    ax.set_title('NV-axis: (0,1,1)')
    ax.set_xlabel('phi (degrees)')
    ax.set_ylabel('theta (degrees)')

    ax = fig.add_subplot(224)
    X = np.degrees(phi)
    Y = np.degrees(theta)
    X, Y = np.meshgrid(X,Y)
    Z = s[:,:,:,3].reshape(len(theta),len(phi))
    plt.contourf(X,Y,Z,100)
    plt.colorbar()
    ax.set_title('NV-axis: (-1,0,-1)')
    ax.set_xlabel('phi (degrees)')
    ax.set_ylabel('theta (degrees)')

    plt.subplots_adjust(wspace=.3,hspace=.5)
    plt.draw
    plt.show()
##zf = np.array([2.87e9,3.6e6,2.6e6,0,0,0,0,0])
##theta = np.linspace(0.,np.pi,num=100)
##phi = np.linspace(0.001,2*np.pi,num=100)
##plotsplittings(phi,theta,zf)



############################################
# Current problem to solve:
# How to characterize small field spectra from splitting and width?
# hrm...
# the splitting places a range on the magnetic field magnitude
# if the width has changed, this Bmag is not Bz
# plug in B corresponding to fitting, rotate this Bmag, and fit to data?
############################################
##freq = np.linspace(2.77e9,2.97e9,1e6)
##zf = np.array([2.87e9,3.6e6,2.6e6,0,0,0,0,0])
##spectra = lor8(freq,zf,ev)
##print freq.shape
##print spectra.shape
##plt.plot(freq,spectra[0,0,0,:])
##plt.show()


############################################
# Calculate splitting and width as a function of theta and phi
############################################
def swth(phi,theta,zf,freq):
    print datetime.datetime.now()
    Bmag = np.array([200e-6])
    Bmag = Bmag.reshape(len(Bmag),1,1)
    Bxyz = s2c(theta,phi,Bmag)
    ev = eigenvalues(Bxyz,zf)
    spectra = lor8(freq,zf,ev)
    def lor(freq,a1,a2,D,E,width,offset):
        lor1 = a1*((width**2)/(np.pi*width*((freq-(D+E))**2 + width**2)))
        lor2 = a2*((width**2)/(np.pi*width*((freq-(D-E))**2 + width**2)))
        return lor1+lor2 + offset
    a = (np.amax(spectra,axis=3)*7e6).reshape((len(Bxyz)*len(Bxyz[0]),1))
    base = a/a
    zfcentral = base*zf[0]
    zfsplitting = base*zf[1]
    zfwidth = base*zf[2]
    zfoffset = base*zf[3]
    freq = (base*freq)
    spectra = spectra.reshape((len(Bxyz)*len(Bxyz[0]),len(freq[1])))
    p0 = np.concatenate((a,a,zfcentral,zfsplitting,zfwidth,zfoffset),axis=1)
    print freq.shape
    print spectra.shape
    print datetime.datetime.now()
##    print lor(freq,a,a,zfcentral,zfsplitting,zfwidth,zfoffset).shape
##    print a.shape
##    print zfcentral.shape
##    print zfsplitting.shape
##    print zfwidth.shape
##    print zfoffset.shape
##    print p0.shape
    coeffs = np.ones((len(Bxyz)*len(Bxyz[0]),len(p0[0])))
    print coeffs.shape
    thetaphi = np.array([[x0,y0] for x0 in theta \
                      for y0 in phi])
    for i in range(0,len(coeffs),1):
        print i
##        print thetaphi[i]
        coeffs[i], matcov = curve_fit(lor,freq[i],spectra[i],p0[i])
        yfit = lor(freq[i],coeffs[i,0],coeffs[i,1],coeffs[i,2],\
                   coeffs[i,3],coeffs[i,4],coeffs[i,5])
##        print coeffs[i]
##        plt.plot(freq[i],spectra[i,:],'r-',freq[i],yfit,'b--')
##        plt.show()
    np.savetxt('coeffs.txt',coeffs,delimiter=', ')

    
    
freq = np.linspace(2.77e9,2.97e9,1e6)
zf = np.array([2.87e9,3.6e6,2.6e6,0,0,0,0,0])
theta = np.linspace(0.001,np.pi,num=13)
phi = np.linspace(0.001,2*np.pi,num=17)
##swth(phi,theta,zf,freq)

a1,a2,zfcentral,zfsplitting,zfwidth,zfoffset =\
    np.loadtxt('coeffs.txt',delimiter=', ',unpack=True)

print zfsplitting.shape
print theta.shape
print phi.shape
fig = plt.figure(figsize=plt.figaspect(1.))

ax = fig.add_subplot(211)
X = np.degrees(phi)
Y = np.degrees(theta)
X, Y = np.meshgrid(X,Y)
Z = zfsplitting.reshape(len(theta),len(phi))
plt.contourf(X,Y,Z,100)
plt.colorbar()
ax.set_title('Splitting')
ax.set_xlabel('phi (degrees)')
ax.set_ylabel('theta (degrees)')

ax = fig.add_subplot(212)
X = np.degrees(phi)
Y = np.degrees(theta)
X, Y = np.meshgrid(X,Y)
Z = zfwidth.reshape(len(theta),len(phi))
plt.contourf(X,Y,Z,100)
plt.colorbar()
ax.set_title('Width')
ax.set_xlabel('phi (degrees)')
ax.set_ylabel('theta (degrees)')

plt.subplots_adjust(wspace=.3,hspace=.5)
plt.draw
plt.show()





