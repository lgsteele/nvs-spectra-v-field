import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.optimize import leastsq
from scipy.signal import medfilt
from scipy.stats import chisquare
import peakutils
import datetime

##from DEwidthZeroFieldFit import de
##from lorfit import lorfit
##from BtpSweep import BtpSweep
##from NVeigenvalues import eigenvalues
##from lor8 import lor8

###########################
###########################
# Determine D, E, and width from zero-field data
###########################
###########################
def de(filename,zfArray):
    def lor(f,a1,a2,D,E,width,offset):
        lor1 = a1*((width**2)/(np.pi*width*((f-(D+E))**2 + width**2)))
        lor2 = a2*((width**2)/(np.pi*width*((f-(D-E))**2 + width**2)))
        return lor1+lor2 + offset
    freq, signal = np.loadtxt('%s.txt' % filename,\
                              delimiter=', ',skiprows=1,unpack=True)
    signal = signal/np.amax(signal) # normalize the signal to 1
    p0=[1e7,1e7,2.87e9,4e6,4e6,0]
    coeffs, matcov = curve_fit(lor, freq, signal, p0)
    yStdErr = np.sqrt(np.diag(matcov)) # compute standard errors of fit params
    yfit = lor(freq,coeffs[0],coeffs[1],coeffs[2],\
               coeffs[3],coeffs[4],coeffs[5])
    zfArray[0] = coeffs[2] # D
    zfArray[1] = coeffs[3] # E
    zfArray[2] = coeffs[4] # width
    zfArray[3] = coeffs[5] # offset
##    plt.plot(freq,signal,'b-',freq,yfit,'r-')
##    plt.show()
    return zfArray



###########################
###########################
# Calculate eigenvalues based on Bmag, theta, and phi
###########################
###########################
def eigenvalues(zfArray,btpArray):
    zfs1 = zfArray[0]*np.array([[0,.3333,.3333],\
                              [.3333,0,-.3333],\
                              [.3333,-.3333,0]])
    strain1 = zfArray[1]*np.array([[.3333,-.3333,.6667],\
                              [-.3333,-.6667,.3333],\
                              [.6667,.3333,.3333]])
    zfs2 = zfArray[0]*np.array([[0,.3333j,-.3333],\
                              [-.3333j,0,-.3333j],\
                              [-.3333,.3333j,0]])
    strain2 = zfArray[1]*np.array([[.3333,-.3333j,-.6667],\
                              [.3333j,-.6667,.3333j],\
                              [-.6667,-.3333j,.3333]])
    zfs3 = zfArray[0]*np.array([[0,-.3333,.3333],\
                       [-.3333,0,.3333],\
                       [.3333,.3333,0]])
    strain3 = zfArray[1]*np.array([[.3333,.3333,.6667],\
                         [.3333,-.6667,-.3333],\
                         [.6667,-.3333,.3333]])
    zfs4 = zfArray[0]*np.array([[0,-.3333j,-.3333],\
                       [.3333j,0,.3333j],\
                       [-.3333,-.3333j,0]])
    strain4 = zfArray[1]*np.array([[.3333,.3333j,-.6667],\
                          [-.3333j,-.6667,-.3333j],\
                          [-.6667,.3333j,.3333]])
    sx = np.array([[0,.7071,0],[.7071,0,.7071],[0,.7071,0]])
    sy = np.array([[0,-.7071j,0],[-.7071j,0,-.7071j],[0,-.7071j,0]])
    sz = np.array([[1,0,0],[0,0,0],[0,0,-1]])
    zeeman = 28024951642*btpArray[0]*(np.sin(btpArray[1])*np.cos(btpArray[2])*sx+\
                            np.sin(btpArray[1])*np.sin(btpArray[2])*sy+\
                            np.cos(btpArray[2])*sz)
    w, v = LA.eigh(zfs1+strain1+zeeman)
    f1 = abs(w[0])+abs(w[1])
    f2 = abs(w[0])+abs(w[2])
    w, v = LA.eigh(zfs2+strain2+zeeman)
    f3 = abs(w[0])+abs(w[1])
    f4 = abs(w[0])+abs(w[2])
    w, v = LA.eigh(zfs3+strain3+zeeman)
    f5 = abs(w[0])+abs(w[1])
    f6 = abs(w[0])+abs(w[2])
    w, v = LA.eigh(zfs4+strain4+zeeman)
    f7 = abs(w[0])+abs(w[1])
    f8 = abs(w[0])+abs(w[2])
    evals = np.sort(np.array([f1,f2,f3,f4,f5,f6,f7,f8]))
##    evArray[0] = evals[0]
##    evArray[1] = evals[1]
##    evArray[2] = evals[2]
##    evArray[3] = evals[3]
##    evArray[4] = evals[4]
##    evArray[5] = evals[5]
##    evArray[6] = evals[6]
##    evArray[7] = evals[7]
    return evals



###########################
###########################
# Generate 8 peak lorentzian spectra using eigenvalues
###########################
###########################
def lor8(freq,zfArray,ampArray,evArray):
    width = zfArray[2]
    offset = zfArray[3]
    def func(freq,ampArray,evArray):
        lor1 = ampArray[0]*((width**2)/(np.pi*width*((freq-evArray[0])**2 + width**2)))
        lor2 = ampArray[1]*((width**2)/(np.pi*width*((freq-evArray[1])**2 + width**2)))
        lor3 = ampArray[2]*((width**2)/(np.pi*width*((freq-evArray[2])**2 + width**2)))
        lor4 = ampArray[3]*((width**2)/(np.pi*width*((freq-evArray[3])**2 + width**2)))
        lor5 = ampArray[4]*((width**2)/(np.pi*width*((freq-evArray[4])**2 + width**2)))
        lor6 = ampArray[5]*((width**2)/(np.pi*width*((freq-evArray[5])**2 + width**2)))
        lor7 = ampArray[6]*((width**2)/(np.pi*width*((freq-evArray[6])**2 + width**2)))
        lor8 = ampArray[7]*((width**2)/(np.pi*width*((freq-evArray[7])**2 + width**2)))
        return lor1 + lor2 + lor3 + lor4 + lor5 + lor6 + lor7 + lor8 + offset
    return func(freq,ampArray,evArray)
    
    



###########################
###########################
# Fit data to 8 lorentzians using eigenvalues
###########################
###########################
def lorfit(spectraToAnalyze,zfArray,ampArray,evArray):
    ###########################
    # Load data to fit
    ###########################
    freq, signal = np.loadtxt('%s.txt' % spectraToAnalyze,\
                              delimiter=', ',skiprows=1,unpack=True)
##    signal = signal/np.amax(signal) # normalize the signal to 1

    ###########################
    # define fitting function
    ###########################
    def lor8(freq,\
         a1,a2,a3,a4,a5,a6,a7,a8,\
         f1,f2,f3,f4,f5,f6,f7,f8,\
             width, offset):
        lor1 = a1*((width**2)/(np.pi*width*((freq-f1)**2 + width**2)))
        lor2 = a2*((width**2)/(np.pi*width*((freq-f2)**2 + width**2)))
        lor3 = a3*((width**2)/(np.pi*width*((freq-f3)**2 + width**2)))
        lor4 = a4*((width**2)/(np.pi*width*((freq-f4)**2 + width**2)))
        lor5 = a5*((width**2)/(np.pi*width*((freq-f5)**2 + width**2)))
        lor6 = a6*((width**2)/(np.pi*width*((freq-f6)**2 + width**2)))
        lor7 = a7*((width**2)/(np.pi*width*((freq-f7)**2 + width**2)))
        lor8 = a8*((width**2)/(np.pi*width*((freq-f8)**2 + width**2)))
        return lor1 + lor2 + lor3 + lor4 + lor5 + lor6 + lor7 + lor8 + offset
    p0 = [ampArray[0],ampArray[1],ampArray[2],ampArray[3],\
          ampArray[4],ampArray[5],ampArray[6],ampArray[7],\
          evArray[0],evArray[1],evArray[2],evArray[3],\
          evArray[4],evArray[5],evArray[6],evArray[7],\
          zfArray[2],zfArray[3]]
    coeffs, matcov = curve_fit(lor8, freq, signal, p0)
##    print coeffs
    return coeffs

###########################
###########################
# Fit data to 8 lorentzians using Bmag, theta, and phi
###########################
###########################
def lorfit_btp(spectraToAnalyze,zfArray,ampArray,btpArray):
    ###########################
    # Load data to fit
    ###########################
    freq, signal = np.loadtxt('%s.txt' % spectraToAnalyze,\
                              delimiter=', ',skiprows=1,unpack=True)
    Bmag = btpArray[0]
    theta = btpArray[1]
    phi = btpArray[2]
    def lor8_btp(freq,a1,a2,a3,a4,a5,a6,a7,a8,Bmag,theta,phi,width,offset):
        btp = np.array([Bmag,theta,phi])
        btp_ev = eigenvalues(zfArray,btp)
        f1 = btp_ev[0]
        f2 = btp_ev[1]
        f3 = btp_ev[2]
        f4 = btp_ev[3]
        f5 = btp_ev[4]
        f6 = btp_ev[5]
        f7 = btp_ev[6]
        f8 = btp_ev[7]
        lor1 = a1*((width**2)/(np.pi*width*((freq-f1)**2 + width**2)))
        lor2 = a2*((width**2)/(np.pi*width*((freq-f2)**2 + width**2)))
        lor3 = a3*((width**2)/(np.pi*width*((freq-f3)**2 + width**2)))
        lor4 = a4*((width**2)/(np.pi*width*((freq-f4)**2 + width**2)))
        lor5 = a5*((width**2)/(np.pi*width*((freq-f5)**2 + width**2)))
        lor6 = a6*((width**2)/(np.pi*width*((freq-f6)**2 + width**2)))
        lor7 = a7*((width**2)/(np.pi*width*((freq-f7)**2 + width**2)))
        lor8 = a8*((width**2)/(np.pi*width*((freq-f8)**2 + width**2)))
        return lor1 + lor2 + lor3 + lor4 + lor5 + lor6 + lor7 + lor8 + offset
    p0 = [ampArray[0],ampArray[1],ampArray[2],ampArray[3],\
          ampArray[4],ampArray[5],ampArray[6],ampArray[7],\
          Bmag,theta,phi,\
          zfArray[2],zfArray[3]]
    coeffs, matcov = curve_fit(lor8_btp,freq,signal,p0)
    return coeffs





def analyze(zeroFieldFile,spectraToAnalyze,Bmag,theta,phi):
    ###########################
    # Upload data sets
    ###########################
    zffreq, zfsignal = np.loadtxt('%s.txt'%zeroFieldFile,\
                              delimiter=', ',skiprows=1,unpack=True)
    freq, signal = np.loadtxt('%s.txt' % spectraToAnalyze,\
                              delimiter=', ',skiprows=1,unpack=True)

    ###########################
    # Define initial parameter arrays
    ###########################
    zfInit = np.array([2.87e6,4.e6,4.e6,0.])
##    btpInit = np.array([0.,0.,0.])
    x = np.amax(signal)*(1e7) # set peak amplitude
    ampMultiPeakArray = np.array([x,x,x,x,x,x,x,x])
##    evInit = np.array([0.,0.,0.,0.,0.,0.,0.,0.])
    
    ###########################
    # Get D, E, width from zero field data
    ###########################
    zfCalc = de('%s'%zeroFieldFile,zfInit)
##    print zfCalc

    ###########################
    # Calculate eigenvalues using Bmag, theta, phi, and zfCalc
    ###########################
    btpArray = np.array([Bmag,theta,phi])
    evals = eigenvalues(zfCalc,btpArray)
##    print evals

    ###########################
    # Generate guess from eigenvalues
    ###########################
    guess_spectra = lor8(freq,zfCalc,ampMultiPeakArray,evals)

    ###########################
    # Find data eigenvalues using curve fit
    ###########################    
##    coeffs = lorfit('%s'%spectraToAnalyze,zfCalc,ampMultiPeakArray,evals)
    coeffs = lorfit_btp('%s'%spectraToAnalyze,zfCalc,ampMultiPeakArray,btpArray)
    print coeffs
    btpCalc = np.array([coeffs[8],coeffs[9],coeffs[10]])
##    evCalc = np.array([coeffs[8],coeffs[9],coeffs[10],coeffs[11],\
##                        coeffs[12],coeffs[13],coeffs[14],coeffs[15]])
    ampCalc = np.array([coeffs[0],coeffs[1],coeffs[2],coeffs[3],\
                        coeffs[4],coeffs[5],coeffs[6],coeffs[7]])
    zfPost = np.array([zfCalc[0],zfCalc[1],coeffs[11],coeffs[12]])
    evCalc = eigenvalues(zfPost,btpCalc)
    fit_spectra = lor8(freq,zfPost,ampCalc,evCalc)
    plt.plot(freq,signal,'r',freq,guess_spectra,'g',freq,fit_spectra)
    plt.show()


analyze('esr-zerofield','esr-8peaks',.013,.5,1.2)















