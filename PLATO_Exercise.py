#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 20 19:36:01 2018

Keaton Bell's (Max Planck Institute for Solar System Research) code for the
PSM WP128 Short Cadence (SC) Community Data Analysis Exercise to ptepare for 
PLATO data.

measure(time,flux) function returns measures of (numax,deltanu) from input 
short-cadence time series relative flux measurements.

Can return approximate frequency range of oscillations if returnrange = True

Based on the coefficient of variation method of Bell, Hekker and Kuszlewicz
(2018, MNRAS, submitted).

@author: keatonb
"""

import sys
import numpy as np
from glob import glob
from scipy.interpolate import interp1d
from scipy.optimize import root
import gatspy.periodic as gp

def fillgaps(time,flux):
    #Fill snall gaps of missing values
    #(often every 3 days in Kepler)
    
    #First, if epochs are missing, insert nan at new timestamp
    diffs = np.diff(time)
    typicaldiff = np.median(diffs)
    #if gap <=~ 90 cadences, fill
    addtimes = np.where((diffs > 1.8*typicaldiff) & (diffs < 93*typicaldiff))[0]
    for addt in addtimes:
        nmissed = int(np.round(diffs[addt]/typicaldiff))-1
        newstamps = np.linspace(time[addt],time[addt+1],nmissed+2)[1:-1]
        time = np.hstack((time,newstamps))
        flux = np.hstack((flux,np.zeros(nmissed)*np.nan))
    
    #Reorder
    flux = flux[np.argsort(time)]
    time = time[np.argsort(time)]
    
    missing = np.invert(np.isfinite(flux)) #not finite
    interp = interp1d(time[~missing],flux[~missing],bounds_error=False)#nans for values outside range
    wm= np.where(missing)[0] #where missing
    #are the previous or next missing points adjacent?
    #pad ends with placeholders
    padwm = np.insert(wm,[0,wm.shape[0]],[-1,time.shape[0]])
    nearestind = np.min([padwm[1:-1]-padwm[:-2],padwm[2:]-padwm[1:-1]],axis=0)
    longgap = wm[nearestind < 49]#try filling gaps < 1 day
    newtime = np.delete(time,longgap)
    return newtime,interp(newtime)

#Fast Power Spectrum based on Lomb-Scargle (gatspy)
def ps(time,flux,minfreq=None):
    time-=time[0]
    #if in days, convert to seconds
    if time[1]<1: 
        time=time*86400.
    
    c=np.median(np.diff(time))
    nyq=1./(2.*c)
    df=1./time[-1]
    if minfreq is None:
        minfreq = df
    f,p=gp.lomb_scargle_fast.lomb_scargle_fast(time,flux,f0=minfreq,df=df,Nf=nyq/df)
    lhs=(1./len(time))*np.sum(flux**2.)
    rhs= np.sum(p)
    ratio=lhs/rhs
    p*=ratio/(df*1e6)#ppm^2/uHz
    f*=1e6
    return f,p

def fun(x,a,b,c):
    return a*x**b-x+c

def getbinedges(alpha,beta,A,B,deltafract,fractinc=1.001):
    #We want contiguous bins between A and B
    #With widths that are some constant fraction of a power law
    #binwidth = deltafract*alpha*nu**beta
    #The given deltafract is a lower limit that will be increased until 
    #a bin edge falls at B.
    #Test new fractions, increasing by fractinc
    
    #This is not going to be pretty
    
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


#Try scaling delta nu with two different power laws above and below 300 muHz

#Values needed for calculations
#From Huber et al. 2011, ApJ, 743, 143
minfreq = 1. #muHz
fnyq = 283.2*30. #muHz
boundary = 300. #muHz
lowalpha = 0.267 #Below 300 muHz
lowbeta = 0.76
highalpha = 0.22 #Above 300 muHz
highbeta = 0.8
dnufract = 2. #Min bin size relative to delta nu

def guessdeltanu(numax):
    if numax < boundary:
        return lowalpha*numax**lowbeta
    else:
        return highalpha*numax**highbeta
    
lowbinedges,lownewfract = getbinedges(lowalpha,lowbeta,minfreq,boundary,dnufract,1.0001)
nlowbins = len(lowbinedges) - 1
lowbincenters = (np.array(lowbinedges[:-1])+np.array(lowbinedges[1:]))/2.
highbinedges,highnewfract = getbinedges(highalpha,highbeta,boundary,fnyq,dnufract,1.0001)
nhighbins = len(highbinedges) - 1
highbincenters = (np.array(highbinedges[:-1])+np.array(highbinedges[1:]))/2.
binedges = np.hstack((lowbinedges[:-1],highbinedges))
bincenters = (np.array(binedges[:-1])+np.array(binedges[1:]))/2.

#Interpolate the overplotted bin centers so they scale too
osample = 50
nosamples = osample*(bincenters.shape[0]-1)+1
osamplebincenters = interp1d(np.arange(bincenters.shape[0]),bincenters)
obincenters = osamplebincenters(np.linspace(0,bincenters.shape[0]-1,nosamples))
obinwidths = lownewfract*lowalpha*obincenters**lowbeta
highscaling = np.where(obincenters > 300)
obinwidths[highscaling] = highnewfract*highalpha*obincenters[highscaling]**highbeta
olowedges = obincenters - obinwidths/2.
ohighedges = obincenters + obinwidths/2.

faps = np.loadtxt('FAPs.dat')
binsizes = faps[:,0]
fap99p9 = faps[:,3]
fap = interp1d(binsizes,fap99p9,bounds_error=False,assume_sorted=True)
def CVspec(freq,power,binedges):
    #Writes out central bin freq, N in bin, CV, FAP_0.1%
    nbins = len(binedges)-1
    binfreq = np.zeros(nbins)
    Ninbin = np.zeros(nbins)
    CV = np.zeros(nbins)
    for i in range(nbins):
        binfreq[i] = (binedges[i] + binedges[i+1])/2.
        inbin = np.where((freq >= binedges[i]) & (freq < binedges[i+1]))
        Ninbin[i] = inbin[0].shape[0]
        CV[i] = np.nanstd(power[inbin],ddof=1)/np.nanmean(power[inbin])
    FAP = fap(Ninbin)
    return binfreq,Ninbin,CV,FAP

def CVspec_noncontig(freq,power,lowedges,highedges):
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

def contiguous_regions(condition):
    """Finds contiguous True regions of the boolean array "condition". Returns
    a 2D array where the first column is the start index of the region and the
    second column is the end index.
    
    from https://stackoverflow.com/questions/4494404/find-large-number-of-consecutive-values-fulfilling-condition-in-a-numpy-array"""
    
    # Find the indicies of changes in "condition"
    d = np.diff(condition)
    idx, = d.nonzero() 

    # We need to start things after the change in "condition". Therefore, 
    # we'll shift the index by 1 to the right.
    idx += 1

    if condition[0]:
        # If the start of condition is True prepend a 0
        idx = np.r_[0, idx]

    if condition[-1]:
        # If the end of condition is True, append the length of the array
        idx = np.r_[idx, condition.size] # Edit

    # Reshape the result into two columns
    idx.shape = (-1,2)
    return idx

#LC sampling
lcsamp = 0.566391*1e3

pspsw=10. #width in approximate radial orders
momentbins = 4 #Number of independent bins to calculate moments over
minwidthorders = 4 #Minimum continuous oversampled CV > 1 in approx. radial orders


def measure(time,flux,returnrange = False):
    """Measure numax and deltanu from time series photometry.
    
    Input:
        time (in days)
        flux (relative, in ppm)
    
    Output:
        (numax,deltanu) tuple (in muHz)
    if returnrange:
        (numax,deltanu,lowfreq,highfreq),
        where the latter are approximate limits of the oscillation range
        
    """
    
    #Cleanup light curve
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
    var = np.var(flux)
    numaxest = (var/2.4e7)**(1./-1.18)
    
    #compute power spectrum
    minfreq = numaxest/4.
    freq,power = ps(time,flux,minfreq=minfreq/1e6)
    
    #Mask bins nearest to instrumental artifacts:
    #Only needed for Kepler short cadence
    masknearest = 50
    for noisefreq in np.arange(1,16)*lcsamp:
        power[np.argsort(np.abs(freq - noisefreq))[:masknearest]]=np.nan
    
    #Calculate CV spectra
    binfreq,Ninbin,cv,FAP = CVspec(freq,power,binedges)
    obinfreq,oNinbin,ocv,oFAP = CVspec_noncontig(freq,power,olowedges,ohighedges)
    
    #Calculate local moments
    #Calculate running first moment (will decide which to record later)
    indepsearchbins = momentbins*osample
    minsearchedind = int(indepsearchbins/2)
    maxsearchedind = -int(indepsearchbins/2)
    moment0 = np.zeros(ocv.shape) #integrated cv
    moment1 = np.zeros(ocv.shape) #weighted mean
    temp = np.nancumsum(ocv)
    localnorm = temp[indepsearchbins:] - temp[:-indepsearchbins]
    moment0[minsearchedind:maxsearchedind] = localnorm
    temp = np.nancumsum(np.log10(obincenters)*ocv)
    moment1[minsearchedind:maxsearchedind] = (temp[indepsearchbins:] - temp[:-indepsearchbins])/localnorm
    
    #Assess for each significant, indep. CV above the FAP threshold.
    #Are surrounding oversampled CVs continuously > 1 for > 4 independent bins?
    #Find regions where oversampled CV is contiguously > 1
    rangesabove1 = contiguous_regions(ocv > 1)
    widths = np.array([ra1[1] - ra1[0] for ra1 in rangesabove1])
    wideenough = np.where(widths > minwidthorders*osample)[0]
    
    numaxmeas = np.nan
    peakmoment = 0
    acceptedrange = (np.nan, np.nan)
    
    for i in wideenough:
        lowfreq = obincenters[rangesabove1[i][0]]
        highfreq = obincenters[rangesabove1[i][1]-1]
        contiginds = np.where((bincenters >= lowfreq) & (bincenters <= highfreq))
        if np.any(cv[contiginds] > FAP[contiginds]):
            localpeakmoment = rangesabove1[i][0] + np.argmax(moment0[rangesabove1[i][0]:rangesabove1[i][1]])
            if moment0[localpeakmoment] > peakmoment:
                peakmoment = moment0[localpeakmoment]
                numaxmeas = 10.**moment1[localpeakmoment]
                acceptedrange = (lowfreq,highfreq)
    
    #PSPS for deltanu
    deltanumeas = np.nan
    
    if np.isfinite(numaxmeas):
        numax = numaxmeas
        dnuguess = guessdeltanu(numax)
        norm1 = np.where((freq > numax-(pspsw/2.+1.)*dnuguess) & (freq < numax-(pspsw/2.)*dnuguess))
        norm2 = np.where((freq > numax+(pspsw/2.)*dnuguess) & (freq < numax+(pspsw/2.+1.)*dnuguess))   
        if norm2[0].size == 0: #near Nyquist?
            norm2 = np.where(freq > freq[-100])
        cand = np.where((freq > numax-(pspsw/2.)*dnuguess) & (freq < numax+(pspsw/2.)*dnuguess))
        bkg = interp1d([numax-(pspsw/2.+.5)*dnuguess,numax+(pspsw/2.+.5)*dnuguess],[np.nanmean(power[norm1]),np.nanmean(power[norm2])])
        normed = power[cand]/bkg(freq[cand])

        #Replace interpolate any missing values
        notmissing = np.where(np.isfinite(normed))
        normed = interp1d(freq[cand][notmissing],normed[notmissing],fill_value='extrapolate')(freq[cand])


        fftps = np.abs(np.fft.fft(normed-np.nanmean(normed), n=10*normed.size))**2
        fftfreqs = np.fft.fftfreq(10*normed.size, freq[1]-freq[0])
        #Search close to half Dnu
        search = np.where((fftfreqs > 1./(0.6*dnuguess)) * 
                          (fftfreqs < 1./(0.4*dnuguess)))
        search = search[0][np.argsort(fftfreqs[search])]
        fftps = fftps[search]
        fftfreqs = fftfreqs[search]
        deltanumeas = 2./fftfreqs[np.argmax(fftps)]

    if returnrange:
        return numaxmeas,deltanumeas,acceptedrange[0],acceptedrange[1]
    else:
        return numaxmeas,deltanumeas



if __name__== "__main__":
    
    #Can specify light curve filenames in command line
    lcfiles = []
    if len(sys.argv) > 1:
        lcfiles = sys.argv[1:]
    else:
        lcfiles = glob('Data/*/*.dat')
    
    for filenum,lcfile in enumerate(lcfiles):
        kic = int(lcfile.split('kic')[1].split('_')[0])
        
        lcdat = np.loadtxt(lcfile)
        time = lcdat[:,0]*24.*3600
        flux = 1.+lcdat[:,1]/1e6
        
        res = measure(time,flux,returnrange=True)
        
        print((kic,res))

















