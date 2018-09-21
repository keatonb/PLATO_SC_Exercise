#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Keaton Bell's code for PSM 128, detecting numax and deltanu
"""

import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from tqdm import tqdm_notebook as tqdm
from scipy.interpolate import interp1d
from scipy.optimize import root
import gatspy.periodic as gp

#Read in tabulated false alarm probabilities
faps = np.loadtxt('../2dof/APOKASC/FAPs.dat')
binsizes = faps[:,0]
fap99p9 = faps[:,3]
fap = interp1d(binsizes,fap99p9,bounds_error=False,assume_sorted=True)

def fillgaps(time,flux,maxfillcadences=93):
    #Interpolate across short gaps to make a nice spectral window
    #(often every 3 days in Kepler)
    
    #First, if epochs are missing, insert nans at new timestamps
    diffs = np.diff(time)
    #For short cadence, typical difference is 1 minute
    typicaldiff = np.median(diffs)
    #if gap <=~ 90 cadences, fill
    addtimes = np.where((diffs > 1.8*typicaldiff) & (diffs < maxfillcadences*typicaldiff))[0]
    for addt in addtimes:
        nmissed = int(np.round(diffs[addt]/typicaldiff))-1
        newstamps = np.linspace(time[addt],time[addt+1],nmissed+2)[1:-1]
        time = np.hstack((time,newstamps))
        flux = np.hstack((flux,np.zeros(nmissed)*np.nan))
    
    #Reorder
    flux = flux[np.argsort(time)]
    time = time[np.argsort(time)]
    
    missing = np.isnan(flux) #not finite
    interp = interp1d(time[~missing],flux[~missing],bounds_error=False)#nans for values outside range
    
    #Delete any timestamps associated with long gaps
    wm= np.where(missing)[0] #where missing
    #are the previous or next missing points adjacent?
    #pad ends with placeholders
    padwm = np.insert(wm,[0,wm.shape[0]],[-1,time.shape[0]])
    nearestind = np.min([padwm[1:-1]-padwm[:-2],padwm[2:]-padwm[1:-1]],axis=0)
    longgap = wm[nearestind < maxfillcadences]
    newtime = np.delete(time,longgap)
    return newtime,interp(newtime)

#Fast Fourier transform based on Lomb-Scargle (gatspy)
def ps(time,flux):
    time-=time[0]
    #if in days, convert to seconds
    if time[1]<1: 
        time=time*86400.
    
    c=np.median(np.diff(time))
    nyq=1./(2.*c)
    df=1./time[-1]
    f,p=gp.lomb_scargle_fast.lomb_scargle_fast(time,flux,f0=df,df=df,Nf=nyq/df)
    lhs=(1./len(time))*np.sum(flux**2.)
    rhs= np.sum(p)
    ratio=lhs/rhs
    p*=ratio/(df*1e6)#ppm^2/uHz
    f*=1e6
    return f,p

#For determining where to place CV bin edges:
def fun(x,a,b,c):
    return a*x**b-x+c

def getbinedges(alpha,beta,A,B,deltafract,fractinc=1.001):
    #We want contiguous bins between A and B
    #With widths that are some constant fraction of a power law
    #binwidth = deltafract*alpha*nu**beta
    #The given deltafract is a lower limit that will be increased until 
    #a bin edge falls at B.
    #Test new fractions, increasing by fractinc
    
    #This is not going to be smooth
    
    #First figure out how many bins there are going to be
    edges = [A]
    while edges[-1] < B:
        bincenter = root(fun,edges[-1],args=(deltafract*alpha/2.,beta,edges[-1])).x[0]
        edges.append(2.*bincenter-edges[-1])
    
    N = len(edges)-2 #number of bins to use
    
    #Now, increase deltafract until the last bin is pushed past B
    testfract = [deltafract]
    edgeend = [edges[-2]]
    
    while edgeend[-1] < B:
        testfract.append(testfract[-1]*fractinc)
        edges = [A]
        for i in range(N):
            bincenter = root(fun,edges[-1],args=(testfract[-1]*alpha/2.,beta,edges[-1])).x[0]
            edges.append(2.*bincenter-edges[-1])
        edgeend.append(edges[-1])
    
    #interpolate new deltafract
    newfract = interp1d(edgeend,testfract)(B)
    
    #use this bin size to calculate final bin edges
    edges = [A]
    for i in range(N):
        bincenter = root(fun,edges[-1],args=(newfract*alpha/2.,beta,edges[-1])).x[0]
        edges.append(2.*bincenter-edges[-1])
    
    return edges,newfract

#Function for computing CV spectra:
def CVspec(freq,power,lowedges,highedges):
    #Writes out central bin freq, N in bin, CV, FAP_0.1%
    nbins = len(lowedges)
    binfreq = (lowedges + highedges)/2.
    Ninbin = np.zeros(nbins)
    CV = np.zeros(nbins)
    for i in range(nbins):
        inbin = np.where((freq >= lowedges[i]) & (freq < highedges[i]))
        Ninbin[i] = inbin[0].shape[0]
        CV[i] = np.nanstd(power[inbin],ddof=1)/np.nanmean(power[inbin])
    FAP = fap(Ninbin)
    return binfreq,Ninbin,CV,FAP

def calculate(lcfile):
    #load data
    lcdat = np.loadtxt(lcfile)
    time = lcdat[:,0]*24.*3600
    flux = 1.+lcdat[:,1]/1e6
    #Cleanup
    #Remove non-finite
    isfinite = np.isfinite(flux)
    time = time[isfinite]
    flux=flux[isfinite]
    #5 sigma clip
    keep = np.where(np.abs(flux - np.mean(flux)) < 5.*np.std(flux))[0]
    flux = flux[keep]
    time = time[keep]
    time,flux = fillgaps(time,flux) 
    flux = 1e6*(flux/np.mean(flux) -1.)
    
    #Estimate numax from variance
    #Rough estimate of nu_max from variance
    #Using long cadence relation for giants from Hekker et al. 2012,A&A,544,A90
    #(Systematically underestimates numax)
    var = np.var(flux)
    numaxest = (var/2.4e7)**(1./-1.18)
    
    #Use different scaling relation if numaxest > 300 muHz
    #(Huber et al. 2011, ApJ, 743, 143)
    lowalpha = 0.267
    lowbeta = 0.76
    highalpha = 0.22
    highbeta = 0.8
    alpha = lowalpha
    beta = lowbeta
    if numaxest > 300:
        alpha = highalpha
        beta = highbeta
    dnufract = 2.02 #Bin over ~2 radial orders for CV spectra
    lowbinedges,lownewfract = getbinedges(lowalpha,lowbeta,1.,300,dnufract,1.0001)
    highbinedges,highnewfract = getbinedges(highalpha,highbeta,300,fnyq,dnufract,1.0001)
    binedges = np.hstack((lowbinedges[:-1],highbinedges))
    
    #Decide on bin edges
    binedges,newfract = getbinedges(alpha,beta,1.,fnyq,dnufract,1.0001)
    binedges = np.array(binedges)
    noversamples = 2000
    obincenters = np.logspace(np.log10((binedges[0]+binedges[1])/2.),np.log10((binedges[-2]+binedges[-1])/2.),noversamples)
    obinwidths = newfract*alpha*obincenters**beta #Adjusted to newbinfract
    olowedges = obincenters - obinwidths/2.
    ohighedges = obincenters + obinwidths/2.
    

    #compute power spectrum
    freq,power = ps(time,flux)
    #Mask bins nearest to instrumental artifacts:
    for noisefreq in np.arange(1,16)*lcsamp:
        power[np.argsort(np.abs(freq - noisefreq))[:10]]=np.nan
    #Compute contiguous and oversampled CV spectra
    binfreq,Ninbin,cv,FAP = CVspec(freq,power,binedges[:-1],binedges[1:])
    obinfreq,oNinbin,ocv,oFAP = CVspec(freq,power,olowedges,ohighedges)
    
    #Only consider signals above some conservative lower limit
    lownumaxlim = numaxest/4.
    
    #Set CV spectra =1 below this limit
    cv[binfreq < lownumaxlim] = 1.
    ocv[obinfreq < lownumaxlim] = 1.
    
    #Vet all candidate power excesses
    cvpeakheight = []
    cvbreadth = []
    cvweightedfreq = []
    
    fig,ax = plt.subplots(figsize=(10,3))
    cvorig = np.copy(cv)
    while(np.any(cv > FAP)):
        #Choose highest independent CV feature exceeding FAP
        freqguess = binfreq[np.where(cv > FAP)[0][np.argmax(cv[cv > FAP])]]
        #Measure breadth of oCV > 1 and compare to ~6deltanu threshold
        
        #If wide enough, record value
        
        #Mask all CV > 1 surrounding candidate
        cv[:] = 0
        
        
        ax.axvline(freqguess,c='black')
    
    
    
    
    ax.plot(binfreq,cv,lw=1)
    ax.plot(binfreq,FAP,ls='--')
    ax.plot(obinfreq,ocv,lw=1)
    ax.axhline(1,lw=0.5,ls='--')
    ax.axvline(numaxest,c='lightblue')
    ax.axvline(numaxest/4.,c='lightblue',ls='--')
    ax.set_xlabel('frequency')
    ax.set_ylabel('CV')
    ax.set_title(lcfile.split('/')[-1])
    ax.set_xlim(1,8800)
    ax.set_xscale('log')
    for be in binedges:
        ax.axvline(be,c='gray',lw=0.5)
    plt.show()

if __name__ == '__main__':
    #Find all light curve files
    lcfiles = glob('/home/bell/seismo/data/PLATO_Ex1_Data/*/*.dat')
    #Set a few needed values
    fnyq = 283.2*30.
    lcsamp = 0.566391*1e3 #long cadence
    
    for lcfile in lcfiles:
        calculate(lcfile)
    