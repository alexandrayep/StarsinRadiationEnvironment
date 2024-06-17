#from Mamajek:

#import Python stuff
import numpy as np
#from to_precision import to_precision
import matplotlib.pyplot as plt
#%matplotlib inline
plt.rcParams.update({'font.size': 14,'lines.linewidth':2})
#import cPickle as cP
#import pickle
import math
import matplotlib.cm as cm
import os
from os.path import exists
from astropy.io import fits
import random
from getpass import getuser
from astropy.time import Time
from PyAstronomy import pyasl
from PyAstronomy.pyaC import pyaErrors as PE
from PyAstronomy.pyasl import _ic
import scipy.interpolate as sci

print('Working on '+getuser()+'.')

#directories:
curr=os.getcwd().replace('PythonPlots','').replace('\\','/')
ddir=curr+r'PythonPlots/data2/' #where to access and save your data
pdir=curr+r'PythonPlots/plots2/' #where to save your plots
cdir=curr+r'chiron/' #reduced chiron data, straight from GSU server
downdir=r'/Users/'+getuser()+r'/Downloads/' #downloads folder, where Vizier will put the .tsv
#edir=curr+r'PaperCC_CG17/Figures/'

#Ultimate opendat:
def opendatt(dir,filename,spl=''): #dir,'filename'. For opening a data file. Can then send through roundtable.
    f=open(dir+filename,'r')
    dat=f.readlines()
    f.close()
    if spl=='':
        labels=dat[0][0:-1].split()
        dat2=[[a.strip('\n') for a in d.split()] for d in dat if d[0]!='#']
    else:
        labels=dat[0][0:-1].split(spl)
        dat2=[[a.strip('\n') for a in d.split(spl)] for d in dat if d[0]!='#']
    dat3=[['nan' if a.strip()=='' else a for a in d] for d in dat2]
    return [dat3,labels]

def opendat(dirr,filename,params,splitchar=''): #Use as var,var,var...=opendat(dir,'filename',['keys']).
    if splitchar=='':
        dat,label=opendatt(dirr,filename)
    else:
        dat,label=opendatt(dirr,filename,splitchar)  #Get keys by first leaving ['keys'] blank: opendat(dirr,filename,[])
    print(label)
    varrs=[]
    for i in range(len(params)):
        j=label.index(params[i])
        try:
            var=np.array([float(d[j]) for d in dat]) #works for float.
            varrs.append(var)
        except ValueError:
            var=[d[j].strip() for d in dat] #works for strings.
            varrs.append(var)
    if len(params)==1:
        varrs=varrs[0]
    return varrs

def writedat(dirr,filename,pars,label): #.dat auto included. pars as [name,ra,dec] etc.
    datp=[[str(a[i]) for a in pars] for i in range(len(pars[0]))]
    f=open(dirr+filename+'.dat','w')
    print('\t'.join(label),file=f)
    print(label)
    for d in datp:
        print('\t'.join(d),file=f)
    f.close()
    print('It is written: '+filename+'.dat')

def erm(val,err): #list,list
    v=np.array(val)
    e=np.array(err)
    w=1.0/e**2.0
    avg=np.nansum(w*v)/np.nansum(w)
    avgerr=1.0/np.sqrt(np.nansum(w))
    return avg,avgerr

def magl(mag): #for set of magnitudes with known mag errs. Err-weighted flux-average.
    m=np.array(mag)
    f=10.0**(-0.4*m) #*F0   flux
    fav=np.mean(f)
    mav=-2.5*np.log10(fav)
    return mav 
    
def errmagl(mag,magerr): #for set of magnitudes with known mag errs. Err-weighted flux-average.
    m=np.array(mag)
    merr=np.array(magerr)
    f=10.0**(-0.4*m) #*F0   flux
    ferr=abs(-0.4*np.log(10.)*f*merr) #propagated mag err to flux err
    fav,faverr=erm(f,ferr)
    mav=-2.5*np.log10(fav)
    maverr=abs(-2.5/np.log(10.)*faverr/fav)
    #return -2.5*np.log10(f),-2.5*np.log10(f-ferr)+2.5*np.log10(f),-2.5*np.log10(f)+2.5*np.log10(f+ferr)
    #PHa 41,15,51 show significant diff between upper and lower errors. Others do not.
    #settling for formal, even errors for now.
    return mav,maverr

def errmagst(mag,magerr): #for set of magnitudes with known mag errs. Err-weighted flux-average.
    m=np.array(mag)
    merr=np.array(magerr)
    f=10.0**(-0.4*m) #*F0   flux
    ferr=abs(-0.4*np.log(10.)*f*merr) #propagated mag err to flux err
    fav,faverr=erm(f,ferr)
    fstd=np.nanstd(f)
    mav=-2.5*np.log10(fav)
    mstd=abs(-2.5/np.log(10.)*fstd/fav)
    #return -2.5*np.log10(f),-2.5*np.log10(f-ferr)+2.5*np.log10(f),-2.5*np.log10(f)+2.5*np.log10(f+ferr)
    #PHa 41,15,51 show significant diff between upper and lower errors. Others do not.
    #settling for formal, even errors for now.
    return mav,mstd

def nodup(lis):
    return([lis[i] for i in range(len(lis)) if lis[i] not in lis[:i]])

#Calc V from BP, from Jao 2018 (Wei-Chun!)
def fm(m,merr):
    f=10.**(-0.4*np.array(m))
    ferr=abs(-0.4*np.log(10.)*f*np.array(merr))
    return f,ferr

def mf(f,ferr):
    m=-2.5*np.log10(f)
    merr=2.5/np.log(10.)*ferr/f
    return m,merr

def weiv(Bp,Bperr,Rp,Rperr): #Johnson V
    Fbp,Fbperr=fm(Bp,Bperr)
    Frp,Frperr=fm(Rp,Rperr)
    V=0.97511*Bp+0.02489*Rp-0.20220
    Verr=np.sqrt((2.437775/np.log(10.)*Fbperr/Fbp)**2.+(0.062225/np.log(10.)*Frperr/Frp)**2.+0.04035**2.)
    return V,Verr

def weii(Bp,Bperr,Rp,Rperr): #Cousin I
    Fbp,Fbperr=fm(Bp,Bperr)
    Frp,Frperr=fm(Rp,Rperr)
    I=Rp-0.75319+1.41105*(Bp-Rp)-1.00136*(Bp-Rp)**2.+0.27106*(Bp-Rp)**3.-0.02489*(Bp-Rp)**4.
    db=2.5/np.log(10.)*Fbperr/Fbp
    dr=2.5/np.log(10.)*Frperr/Frp
    dIdb=1.41105 - 2.00272*(Bp - Rp) + 0.81318*(Bp - Rp)**2. - 0.09956*(Bp - Rp)**3. #wolfram alpha
    dIdr=-0.41105 + 2.00272*(Bp - Rp) - 0.81318*(Bp - Rp)**2. + 0.09956*(Bp - Rp)**3. #wolfram alpha
    Ierr=np.sqrt((dIdb*db)**2.+(dIdr*dr)**2.+0.02770**2.)
    return I,Ierr

def bol(m,merr,BCv,VMint): #array, array, intrinsic band-V (mamajek). If m is V, then color=0.
    mbol=m+BCv+VMint
    mbolerr=merr
    return mbol,mbolerr

def Mabs(m,mbolerr,dist,derr): #apparent magnitude
    #m,dist=np.array(mm),np.array(distt)
    #mbolerr,D,derr=np.array(mbolerrr),np.array(distt),np.array(derrr)
    M = m-5.0*np.log10(dist/10.0)
    
    f,ferr=fm(m,mbolerr)
    Merr=2.5/np.log(10.)*np.sqrt((ferr/f)**2.+(2.*derr/dist)**2.)
    #Merr=np.sqrt(mbolerr**2.+(5.*derr/(np.log(10.)*dist))**2.)
    return M,Merr

#M,Me=Mabs(m,mea,361.8,0.0)

def Lum(Mb,Mberr): #bollometric Mag
    #Mb,Mberr=np.array(Mbb),np.array(Mbberr)
    Mbolsun,Mbolsunerr=4.7554,0.0004 #using Mamajek Mbolsun
    #Lsun=3.8270e33 #for ergs
    #Lerg=Lsun*10.0**((Mbolsun-Mb)/2.5)
    L=10.0**((Mbolsun-Mb)/2.5) #in units of Lsun
    
    f,ferr=fm(Mb,Mberr)
    fs,fserr=fm(Mbolsun,Mbolsunerr)
    Lerr=L*np.sqrt((ferr/f)**2.+(fserr/fs)**2.)
    #Lerr=np.log(10.)/2.5*L*np.sqrt(Mberr**2.+Mbolsunerr**2.)
    return list(L),list(Lerr)
    
def hmsdms(ra,dec): #input in degrees, output in 'HH MM SS.SSS'
    H=ra/15.
    h=int(H)
    M=(H-h)*60.
    m=int(M)
    s=(M-m)*60.
    
    DEC=abs(dec)
    sign=dec/DEC
    dd=int(DEC)
    d=int(sign*dd)
    AM=(DEC-dd)*60.
    am=int(AM)
    ass=(AM-am)*60.
    return str(h)+' '+str(m)+' '+str(round(s,3))+'\t'+str(d)+' '+str(am)+' '+str(round(ass,3))

def dmshms(ra,dec): #input 'HH:MM:S.SSS', output degrees
    splitchar=' '
    if ':' in ra:
        splitchar=':'
    hms=[a.split(splitchar) for a in ra]
    h=[float(a[0])*15. for a in hms]
    m=[float(a[1])*15./60. for a in hms]
    s=[float(a[2])*15./3600. for a in hms]
    H=[h[i]+m[i]+s[i] for i in range(len(ra))]
    
    dms=[a.split(splitchar) for a in dec]
    d=[float(a[0]) for a in dms]
    am=[float(a[1])/60. for a in dms]
    ass=[float(a[2])/3600. for a in dms]
    sign=[-1. if a[0]=='-' else 1. for a in dec]
    D=[sign[i]*(abs(d[i])+am[i]+ass[i]) for i in range(len(ra))]
    
    return H,D

#MESA Isochrones
#tmodp,Gmodp,Bmodp,Rmodp=opendat(ddir,'PadovaGaiaDR2Isochrones.dat',('Age','Gmag','G_BPmag','G_RPmag'))
# tmodm,Gmodm,Bmodm,Rmodm=opendat(ddir,'MESA_z00_vcrit0.dat',('isochrone_age_yr','Gaia_G_MAW','Gaia_BP_MAWf', 'Gaia_RP_MAW'))
# t1myrm=(np.array(tmodm)>1e6-10.)*(np.array(tmodm)<1e6+10.)
# t10myrm=(np.array(tmodm)>1e7-10.)*(np.array(tmodm)<1e7+10.)
# t100myrm=(np.array(tmodm)>1e8-10.)*(np.array(tmodm)<1e8+10.)
# t1000myrm=(np.array(tmodm)>1e9-10.)*(np.array(tmodm)<1e9+10.)

#dists=opendat(ddir,'CGdists.dat',['CG1', 'CG3', 'CG4', 'CG14', 'CG17', 'CG22', 'CG30', 'GDC1'])

#Import plts variable from ClusterFinder:
# with open(ddir+'CHIRONtargets.datbn', 'rb') as f:
#     plts = pickle.load(f)

#plts=[plt30,plt1,plt22,plt4,plt3,plt14,plt17,pltgdc1]
#for each CG: [padova1gyr,padova4myr,pri2,pri1.5,pri1]
#Each has [BR,B].

#from Blaze Use:

#names:achis setup
stds,sachi=opendat(cdir,'standards/CHIRON_standards_best.dat',('#name','achi'))
cgs,cachi=opendat(cdir,'CGS/CHIRON_CGS_best.lis',('#name','achi'))
names=stds+cgs
achis=sachi+cachi

#categories: standards vs. targets
cats=['standards/']*len(stds)+['CGS/']*len(cgs)

#open .fits files
def opfits(name,epoch=1): #'object name' from the log.  Setup:   dat,head=opfits('name') NEW since LogCruncher
    ind=[i for i, n in enumerate(names) if n == name][epoch-1] #some stars have more than one epoch. In such a case, you can choose which you want.
    
    achi=achis[ind] #references achis, generated from LogCruncher files above
    cat=cats[ind]
    
    hdulist=fits.open(cdir+cat+achi)
    #hdulist.info()
    dat=hdulist[0].data
    head=hdulist[0].header
    hdulist.close()
    return dat,head

#wavelength, flux
def wf(dat,o): #whichord,dat. Setup:   w,f=wf(#,dat)
    w=np.array([d[0] for d in dat[o]])
    f=np.array([d[1] for d in dat[o]])
    return w,f

#Blaze file:
bdat=opendat(cdir,'blaze.dat',['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '10', '11', '12', '13', '14', '15', '16', '17', '18', '19', '20', '21', '22', '23', '24', '25', '26', '27', '28', '29', '30', '31', '32', '33', '34', '35', '36', '37', '38', '39', '40', '41', '42', '43', '44', '45', '46', '47', '48', '49', '50', '51', '52', '53', '54', '55', '56', '57', '58', '59', '60', '61'])

#Orders of interest:
os=[0,1,2,3,5,6,7,10,11,12,13,14,17,18,19,20,21,22,23,24,25,28,31,32,33,34,36,37,40,52] #of interest; dodge pressure-broadened (wide wings) and telluric: skip any with sharp lines in B star! ex: 26,29,35,38,39(butHa),43-48,51(best!),53-7 have tel. lines.

#Special orders:
oss=[39,41] #Ha,Li

#Add 39 and 41 for ALL orders of interest and special orders. :) For plots.
osplus=[0,1,2,3,5,6,7,10,11,12,13,14,16,17,18,19,20,21,22,23,24,25,28,31,32,33,34,36,37,39,40,41,52] #of interest; dodge pressure-broadened (wide wings) and telluric: skip any with sharp lines in B star! ex: 26,29,35,38,39(butHa),43-48,51(best!),53-7 have tel. lines.

def useblaze(fraw,head,o): #flux of object, order of interest
    fb=bdat[o]
    #factor=np.max(fb)/(np.max(f[350:375])-np.std(f[350:375]))
    #fn=np.array(f)/(np.array(fb)*(np.max(f[350:375])-np.std(f[350:375]))/np.max(fb))
    
    gain=float(head['GAIN'])
    RN=float(head['RON'])
    K=2.5
    
    mean=np.mean(fraw[325:425])
    std=np.std(fraw[325:425])
    
    #kill cosmic rays, particularly in peak region
    f=[fraw[0],]+[fraw[i] if fraw[i]<mean+2.5*std else np.mean((fraw[i-1],fraw[i+1])) for i in range(1,len(fraw)-1)]+[fraw[-1],]
    
    SNR=np.array(f)*gain/np.sqrt(np.array(f)*gain+K*RN**2.)    
    SNRb=np.array(fb)*gain/np.sqrt(np.array(fb)*gain+K*RN**2.)
    #SNR=signal/noise --> noise=signal/SNR
    #f=signal+noise=signal+signal/SNR=signal(1+1/SNR)
    #signal=f/(1+1/SNR)
    #scale max signal of blaze to max signal of target.
    
    signalmax=sorted(f[325:425])[-6]/(1.+1./np.mean(SNR[325:425])) #cut the highest
    signalbmax=sorted(fb[325:425])[-6]/(1.+1./np.mean(SNRb[325:425])) #cut the highest
    scaleblaze=signalmax/signalbmax
    
    #fn=np.array(f)/(np.array(fb)*np.max(f[350:375])/np.max(fb)) #low-noise standards work better like this.
    fn=np.array(fraw)/(np.array(fb)*scaleblaze)
    return fn

def normalize(name,plotyn='y'): #'name','y' or 'n'
    #Normalize one spectrum for display or other, normalizes and nothing else
    ws=[[]]*len(osplus)
    fns=[[]]*len(osplus)
    snrs=['snr',]*801 #will hide in first column
    dat,head=opfits(name)
    for o in osplus:
        w,f=wf(dat,o)
        fn=useblaze(f,head,o)
        if plotyn=='y':
            plott(w,fn)
            plt.plot(w[325:425],fn[325:425]) #set up for fiber's center
            mean=np.mean(f[325:425])
            std=np.std(f[325:425])
            plt.plot((w[325],w[425]),(mean,mean))
            plt.plot((w[325],w[425]),(mean+2.5*std,mean+2.5*std))
        ws[osplus.index(o)]=w
        fns[osplus.index(o)]=fn
        snrs[osplus.index(o)]=snr
    return ws,fns,snrs
#check normalizations. If an order needs slightly different scaling, do so to fns.

def smooth(w,f,wc,fc,deg):
    #Fit polynomial to cut spectra.
    Ac = np.zeros((len(wc),2)) 
    Ac[:,0] = wc 
    Ac[:,1] = fc 

    xc = Ac[:,0] 
    yc = Ac[:,1] 
    zc = np.polyfit(xc,yc,deg)
    #print(zc)
    pc = np.poly1d(zc)
    
    D = np.zeros((len(w),2)) 
    D[:,0] = w
    D[:,1] = f
    x = D[:,0]
    y = D[:,1]
    return pc(x)

def polyfit(x,y,deg):
    #Fit polynomial to cut spectra.
    A = np.zeros((len(x),2)) 
    A[:,0] = x 
    A[:,1] = y 

    xx = A[:,0] 
    yy = A[:,1] 
    zz = np.polyfit(xx,yy,deg)
    #print(zz)
    p = np.poly1d(zz)
    
    return p(xx)

def useblaze_cnr(w,fraw,o,head): #flux of object, order of interest
    fb=bdat[o]
    #factor=np.max(fb)/(np.max(f[350:375])-np.std(f[350:375]))
    #fn=np.array(f)/(np.array(fb)*(np.max(f[350:375])-np.std(f[350:375]))/np.max(fb))
    
    gain=float(head['GAIN'])
    RN=float(head['RON'])
    K=2.5
    
    mean=np.mean(fraw[325:425])
    std=np.std(fraw[325:425])
    
    #kill cosmic rays, particularly in peak region
    f=[fraw[0],]+[fraw[i] if fraw[i]<mean+2.5*std else np.mean((fraw[i-1],fraw[i+1])) for i in range(1,len(fraw)-1)]+[fraw[-1],]
    
    SNR=np.array(f)*gain/np.sqrt(np.array(f)*gain+K*RN**2.)    
    SNRb=np.array(fb)*gain/np.sqrt(np.array(fb)*gain+K*RN**2.)
    #SNR=signal/noise --> noise=signal/SNR
    #f=signal+noise=signal+signal/SNR=signal(1+1/SNR)
    #signal=f/(1+1/SNR)
    #scale max signal of blaze to max signal of target.
    
    signalmax=sorted(f[325:425])[-6]/(1.+1./np.mean(SNR[325:425])) #cut the highest
    signalbmax=sorted(fb[325:425])[-6]/(1.+1./np.mean(SNRb[325:425])) #cut the highest
    scaleblaze=signalmax/signalbmax
    
    #fn=np.array(f)/(np.array(fb)*np.max(f[350:375])/np.max(fb)) #low-noise standards work better like this.
    fn=np.array(fraw)/(np.array(fb)*scaleblaze)
    
    #remove cnr and maybe some emission, fine for rv stuff:
    fnn=[fn[0],]+[fn[i] if fn[i]<1+2.5*np.std(fn) else np.mean((fn[i-1],fn[i+1])) for i in range(1,len(fn)-1)]+[fn[-1],]
    
    #flatten out any slope
    slope=smooth(w,fnn,w,fnn,1)
    fnnn=fnn/(slope+np.std(fnn))
    return fnnn

def normalize_rv(name,plotyn='y'): #'name','y' or 'n' #normalizes and removes cosmic rays and some emission for rv
    ws=[[]]*len(osplus)
    fns=[[]]*len(osplus)
    sigmavs=['sigmav',]*801 #will hide in first column; should be as long as w
    dat,head=opfits(name)
    for o in osplus:
        w,f=wf(dat,o)
        fn=useblaze_cnr(w,f,o,head)
        del f
        sig=sigmav(dat,head,fn,o)
        #resample to log-linear
        #w2,fn2=resample(w,fn)
        if plotyn=='y':
            plottt(o,w,fn)
        ws[osplus.index(o)]=w
        fns[osplus.index(o)]=fn
        sigmavs[osplus.index(o)]=sig
    return ws,fns,sigmavs
#check normalizations. If an order needs slightly different scaling, do so to fns.
        
#For weighting results:
#By chain rule: df/dv = df/dw * w/c. So can use df/dw derivative and multiply by w/c for deriv at each pixel.
def deriv(X,Y): #numerical
    #middle: 2 derivs on either side of pixel, then average
    x0=np.array(X[:-2])
    y0=np.array(Y[:-2])
    x1=np.array(X[1:-1])
    y1=np.array(Y[1:-1])
    x2=np.array(X[2:])
    y2=np.array(Y[2:])
    dydx_m=((y1-y0)/(x1-x0)+(y2-y1)/(x1-x0))/2.
    #print len(dydx_m)
    #ends: just do 1 deriv
    dydx_b=(Y[1]-Y[0])/(X[1]-X[0])
    dydx_e=(Y[-1]-Y[-2])/(X[-1]-X[-2])
    #combine:
    dydx=[dydx_b,]+list(dydx_m)+[dydx_e,]
    #print len(dydx)
    return dydx

def sigmav(dat,head,fn,o): #fits data, fits header, normalized flux,order
    w,f=wf(dat,o)
    gain=float(head['GAIN'])
    RN=float(head['RON'])
    if head['MODES'].split(',')[int(head['MODE'])].strip()=='fiber':
        K=2.5
    elif head['MODES'].split(',')[int(head['MODE'])].strip()=='slicer':
        K=9.
    SNR=np.array(f)*gain/np.sqrt(np.array(f)*gain+K*RN**2.)
    c=299792.458 #km/s
    dfdw=deriv(w,fn)
    sigmav=1./np.sqrt(np.nansum((np.array(dfdw)*np.array(w)*np.array(SNR)/c)**2.))
    return sigmav

stds,sachi,sspt=opendat(cdir,'standards/CHIRON_standards_best.dat',('#name','achi','spt'))
rvstds,srvs,vsis=opendat(cdir,'standards/standards_metadata.dat',['#name','rv','vsini'],splitchar='\t')
#cgs,cachi=opendat(cdir,'CGS/CHIRON_CGS_best.lis',('#name','achi'))
#cgsdone=opendat(cdir,'CGS/CGS_SpTGuess.dat',('#name',))[0]
#names=stds+cgs
achis=sachi+cachi
qname,qcg=opendat(cdir,'AllQueuedCGStars.lis',('#name','CG'),splitchar='\t')

def splot(name,o,mode='n'): #plot spectrum, n for normalized, r for raw
    plt.figure(figsize=(10,2))
    plt.title('Order '+str(o))
    dat,head=opfits(name)
    #for o in osplus:
    w,f=wf(dat,o)
    if mode=='r':
        fn=f
        plt.text(w[0]+1.,np.mean(fn[0:20])+0.05*np.mean(fn[200:222]),name+', raw',fontsize=16) #target
    else:
        if mode!='n':
            print('mode invalid, defaulting to normalized spectrum.')
        fn=useblaze_cnr(w,f,o,head)
        plt.plot((w[0],w[-1]),(1.,1),lw=1,ls='--',color='gray')
        plt.text(w[0]+1.,1.+0.05,name,fontsize=16) #target
    plt.plot(w,fn,lw=1,color='blue')
    
#crosscorrRV routine with normalization:
def bcv(head):
    #RA,Dec,expt,jd (jd of middle of exposure time)

    # Coordinates of CHIRON
    longitude = 360.-(70.+48./60.+24.44/3600.) #degrees 0 to 360
    latitude = -30.-10./60.-9.42/3600. #degrees
    altitude = 2207.3 #meters
    
    expt=head['EXPTIME'] #exposure time
    JD=Time(head['DATE-OBS']).jd+expt/2./60./24. #Julian day of middle of exposure, UT

    # Coordinates in degrees (J2000) for pyas1
    RA=[float(d)*15. for d in head['RA'].split(':')] #RA, degrees
    DEC=[float(d) for d in head['DEC'].split(':')] #Dec
    if '-' in head['DEC']:
        sign=-1.
    else:
        sign=1.
    ra2000 = RA[0]+RA[1]/60.+RA[2]/3600. #RA
    dec2000 = sign*(abs(DEC[0])+DEC[1]/60.+DEC[2]/3600.) #Dec

    # Barycentric velocity correction
    cor = pyasl.helcorr(longitude, latitude, altitude,ra2000, dec2000, JD)[0]
    return cor

def RVshift(w0, f0, tw, tf, rv, mode="normdop",skipedge0=0,edgeTapering=None):
    # Speed of light in km/s
    c = 299792.458
    
    if np.sign(rv)>0:
        DW=w0[1]-w0[0]
        fac=1.+rv/c
        skippy=math.ceil(abs(w0[0]*fac-w0[0])/DW)
    if np.sign(rv)<0:
        DW=w0[-2]-w0[-1]
        fac=1.+rv/c
        skippy=math.ceil(abs(w0[-1]*fac-w0[-1])/DW)
    if np.sign(rv)==0:
        skippy=0
    skipedge=skippy+50+skipedge0
    if not _ic.check["scipy"]:
        raise(PE.PyARequiredImport("This routine needs scipy (.interpolate.interp1d).", \
                                                             where="crosscorrRV", \
                                                             solution="Install scipy"))
    # Copy and cut wavelength and flux arrays
    w, f = w0.copy(), f0.copy()
    if skipedge > 0:
        w, f = w[skipedge:-skipedge], f[skipedge:-skipedge]
    
    if edgeTapering is not None:
        # Smooth the edges using a sine
        if isinstance(edgeTapering, float):
            edgeTapering = [edgeTapering, edgeTapering]
        if len(edgeTapering) != 2:
            raise(PE.PyAValError("'edgeTapering' must be a float or a list of two floats.", \
                                                     where="crosscorrRV"))
        if edgeTapering[0] < 0.0 or edgeTapering[1] < 0.0:
            raise(PE.PyAValError("'edgeTapering' must be (a) number(s) >= 0.0.", \
                                                     where="crosscorrRV"))
        # Carry out edge tapering (left edge)
        indi = np.where(w < w[0]+edgeTapering[0])[0]
        f[indi] *= np.sin((w[indi] - w[0])/edgeTapering[0]*np.pi/2.0)
        # Carry out edge tapering (right edge)
        indi = np.where(w > (w[-1]-edgeTapering[1]))[0]
        f[indi] *= np.sin((w[indi] - w[indi[0]])/edgeTapering[1]*np.pi/2.0 + np.pi/2.0)
    
    # Check order of rvmin and rvmax (fake)
    rvmax=rv+1.
    rvmin=rv+0.
    if rvmax <= rvmin:
        raise(PE.PyAValError("rvmin needs to be smaller than rvmax.",
                                                 where="crosscorrRV", \
                                                 solution="Change the order of the parameters."))
    # Check whether template is large enough
    if mode == "lin":
        meanWl = np.mean(w)
        dwlmax = meanWl * (rvmax/c)
        dwlmin = meanWl * (rvmin/c)
        if (tw[0] + dwlmax) > w[0]:
            raise(PE.PyAValError("The minimum wavelength is not covered by the template for all indicated RV shifts.", \
                                                     where="crosscorrRV", \
                                                     solution=["Provide a larger template", "Try to use skipedge"]))
        if (tw[-1] + dwlmin) < w[-1]:
            raise(PE.PyAValError("The maximum wavelength is not covered by the template for all indicated RV shifts.", \
                                                     where="crosscorrRV", \
                                                     solution=["Provide a larger template", "Try to use skipedge"]))
    elif mode == "doppler" or mode == 'normdop':
        # Ensure that the template covers the entire observation for all shifts
        maxwl = tw[-1] * (1.0+rvmin/c)
        minwl = tw[0] * (1.0+rvmax/c)
        if minwl > w[0]:
            raise(PE.PyAValError("The minimum wavelength is not covered by the template for all indicated RV shifts.", \
                                                     where="crosscorrRV", \
                                                     solution=["Provide a larger template", "Try to use skipedge"]))
        if maxwl < w[-1]:
            raise(PE.PyAValError("The maximum wavelength is not covered by the template for all indicated RV shifts.", \
                                                     where="crosscorrRV", \
                                                     solution=["Provide a larger template", "Try to use skipedge"]))
    else:
        raise(PE.PyAValError("Unknown mode: " + str(mode), \
                                                 where="crosscorrRV", \
                                                 solution="See documentation for available modes."))
    # Calculate the cross correlation
    # Apply the Doppler shift
    fi = sci.interp1d(tw*(1.0 + rv/c), tf)
    
    wi=list(w0[:skipedge])+list(w)+list(w0[-1*skipedge:])
    fio=list(np.array(f0[:skipedge])*np.max(tf)/np.max(f0))+list(fi(w))+list(np.array(f0[-1*skipedge:])*np.max(tf)/np.max(f0))
    return wi,fio

def sval(ss): #'spt'
    def vs(s): #v from s, for single value
        if s=='nan':
            v= np.float('nan')
            return np.float('nan')
        else:
            val=60-['O','B','A','F','G','K','M','L','T','Y'].index(s[0])*10+(10-float(s[1:].strip('V').strip('I')))
        if 'III' in s:
            v= val-100.
        elif 'I' in s and 'III' not in s:
            v= val-200.
        else:
            v= val
        return v
    if type(ss)==str:
        s=ss
        final=vs(s)
    if type(ss)==list or type(ss)==type(np.array([1])):
        final=[]
        for i in range(len(ss)):
            s=ss[i]
            v=vs(s)
            final.append(v)
    return final

def svaltotype(vv):
    def sv(v): #s from v
        if np.isnan(v):
            return 'nan'
        lets=['O','B','A','F','G','K','M']
        vals=[60.,50.,40.,30.,20.,10.,0.]
        if v<-130:
            lumn=200
            lum='I'
        elif v<-30:
            lumn=100
            lum='III'
        else:
            lumn=0
            lum='V'
        sv2=int((v+lumn)/10.)*10.
        LET=lets[vals.index(sv2)]
        Class=str(10-(v+lumn-sv2)).replace('.0','')
        if Class=='10':
            LET=lets[vals.index(sv2)+1]
            Class='0'
        final=LET+Class+lum
        return final
    if type(vv)==float or type(vv)==int or type(vv)==type(np.mean([1.,2.])):
        v=vv
        final=sv(v)
    if type(vv)==list or type(vv)==type(np.array([1])):
        final=[]
        for i in range(len(vv)):
            v=vv[i]
            s=sv(v)
            final.append(s)
    return final

def sval2(s): #'spt' Ignore luminosity class
    #print(s,type(s))
    try:
        if np.isnan(s):
            val=np.float('nan')
    except TypeError:
        if s=='nan':
            val=np.float('nan')
        else:
            val=60-['O','B','A','F','G','K','M','L','T','Y'].index(s[0])*10+(10-float(s[1:].strip('V').strip('I')))
    return val

#Mamajek!
sptm, Tm, logTm, logLm, Mbolm, BCvm, Mvm, BVm, BtVtm, GVm, BpRpm, GRpm, M_Gm, bym, UBm, VRcm, VIcm, VKsm, JHm, HKsm, KsW1m, W1W2m, W1W3m, W1W4m, M_Jm, M_Ksm, izm, zYm, R_Rsunm, Msunm=opendat(ddir,'Mamajek.dat',['#SpT', 'Teff', 'logT', 'logL', 'Mbol', 'BCv', 'Mv', 'B-V', 'Bt-Vt', 'G-V', 'Bp-Rp', 'G-Rp', 'M_G', 'b-y', 'U-B', 'V-Rc', 'V-Ic', 'V-Ks', 'J-H', 'H-Ks', 'Ks-W1', 'W1-W2', 'W1-W3', 'W1-W4', 'M_J', 'M_Ks', 'i-z', 'z-Y', 'R_Rsun', 'Msun'])
svalm=sval(sptm)
JKsm=JHm+HKsm
pars=[sptm, Tm, logTm, logLm, Mbolm, BCvm, Mvm, BVm, BtVtm, GVm, BpRpm, GRpm, M_Gm, bym, UBm, VRcm, VIcm, VKsm, JHm, HKsm, KsW1m, W1W2m, W1W3m, W1W4m, M_Jm, M_Ksm, izm, zYm, R_Rsunm, Msunm, svalm,JKsm]
labs=['spt', 'T', 'logT', 'logL', 'Mbol', 'BCv', 'Mv', 'B-V', 'Bt-Vt', 'G-V', 'Bp-Rp', 'G-Rp', 'M_G', 'b-y', 'U-B', 'V-Rc', 'V-Ic', 'V-Ks', 'J-H', 'H-Ks', 'Ks-W1', 'W1-W2', 'W1-W3', 'W1-W4', 'M_J', 'M_Ks', 'i-z', 'z-Y', 'R_Rsun', 'Msun', 'sval','J-Ks']
Mamajek={'Hi':'Cass'}
for i in range(len(labs)):
    Mamajek[labs[i]]=pars[i]

def svalinterp(sv,parm,i1,i2): #sval of target, mamajek param, nearest svalm index i1, 2nd nearest sval index i2. linear interp.
    x=sv
    x1,x2,y1,y2=svalm[i1],svalm[i2],parm[i1],parm[i2]
    y = ((x2-x)/(x2-x1))*y1 + ((x-x1)/(x2-x1))*y2
    return y

def mamajek(spt,parm): #target spectral type, mamajek parameter
    sv=sval2(spt)
    if sv in svalm:
        i=svalm.index(sv)
        parint=parm[i]
    else:
        #linear interp:
        #y = ((x2-x)/(x2-x1)) y1 + ((x-x1)/(x2-x1)) y2
        i1=np.argmin(np.array(svalm)-sv)
        i2=np.argmin(np.array(svalm[::-1])-sv)
        sv1=svalm[i1]
        sv2=svalm[i2]
        svr=sv2-sv1
        parint=svalinterp(sv,parm,i1,i2)
    return parint

def mamajeksv(sv,param): #target spectral type, mamajek parameter
    #sv=sval(spt)
    if np.isnan(sv):
        return np.float('nan')
    parm=Mamajek[param]
    if sv in svalm:
        i=svalm.index(sv)
        parint=parm[i]
    else:
        #linear interp:
        #y = ((x2-x)/(x2-x1)) y1 + ((x-x1)/(x2-x1)) y2
        i1=np.argmin(abs(np.array(svalm)-sv))
        i2=np.argmin(np.array(svalm[::-1])-sv)
        sv1=svalm[i1]
        sv2=svalm[i2]
        svr=sv2-sv1
        parint=svalinterp(sv,parm,i1,i2)
    return parint

def mam(spt,spterr,parm,errhandle='normal'): #SpT, SpT err, parameter name, 'normal' or 'mag' or 'none'
    sv=sval2(spt)
    if np.isnan(sv)=='nan':
        return np.float('nan'),np.float('nan')
    val=mamajeksv(sv,parm)
    valp=mamajeksv(sv+spterr,parm)
    valm=mamajeksv(sv-spterr,parm)
    #print(spt,sv,val,valp,valm)
    vm,vp=sorted((valm,valp))
    vmerr=val-vm
    vperr=vp-val
    if errhandle=='normal':
        verr=np.nanmean([vmerr,vperr])
    if errhandle=='mag':
        verr=magl([vmerr,vperr])
    if errhandle=='none':
        verr=[vmerr,vperr]
    return val,verr

def mampar(par1,par1err,parmlab1,parmlab2,errhandle='normal'): #SpT, SpT err, parameter name, 'normal' or 'mag' or 'none'
    val=mamajek2(par1,parmlab1,parmlab2)
    valp=mamajek2(par1+par1err,parmlab1,parmlab2)
    valm=mamajek2(par1-par1err,parmlab1,parmlab2)
    #print(val,valp,valm)
    vm,vp=sorted((valm,valp))
    vmerr=val-vm
    vperr=vp-val
    if errhandle=='normal':
        verr=np.nanmean([vmerr,vperr])
    if errhandle=='mag':
        verr=magl([vmerr,vperr])
    if errhandle=='none':
        verr=[vmerr,vperr]
    return val,verr

def parminterp(val,parm1lab,parm2lab,i1,i2): #sval of target, 'mamajek param', nearest svalm index i1, 2nd nearest sval index i2. linear interp.
    parm1=Mamajek[parm1lab]
    parm2=Mamajek[parm2lab]
    x=val
    x1,x2,y1,y2=parm1[i1],parm1[i2],parm2[i1],parm2[i2]
    y = ((x2-x)/(x2-x1))*y1 + ((x-x1)/(x2-x1))*y2
    return y

def mamajek2(val,parm1lab,parm2lab): #target value of parm 1, input 'parm1', output 'parm2'
    parm1=Mamajek[parm1lab]
    parm2=Mamajek[parm2lab]
#     print(parm1)
#     print(parm2)
    if np.isnan(val):
        parmval=np.float('nan')
    elif val in parm1:
        parm2s=[parm2[i] for i in range(len(parm1)) if parm1[i]==val]
#         print(i)
#         print(parm2[i])
        parmval=np.nanmedian(parm2s)
    else:
        #linear interp:
        #y = ((x2-x)/(x2-x1)) y1 + ((x-x1)/(x2-x1)) y2
        i1=np.nanargmin(abs(np.array(parm1)-val))
#         print(i1)
        i2=i1+(-1+np.nanargmin(abs(np.array([parm1[i1-1],np.float('nan'),parm1[i1+1]])-val)))
#         print(i2)
#         print(parm2[i1],parm2[i2])
        parmval=parminterp(val,parm1lab,parm2lab,i1,i2)
    return parmval
    
#print useful info for user:
print('\n------------------')
print('\nStandard names:\tstds')
print('Standard SpTs:\tsspt')
print('Target names:\tcgs\n')
print('')
print('Useful functions: (use help() to see inputs)')
print("open file: \t\topendat")
print("write file:\t\twritedat")
print('error-weighted mean:\term')
print('error-weighted flux mean:\tmagl')
print("calc V from B,R:\tweiv")
print("line of best fit:\tpolyfit")
print("deg to hms:\t\thmsdms")
print("hms to deg:\t\tdmshms")
print("open .fits:\t\topfits")
print("wavelength, flux:\twf")
print("plot spectrum:\tsplot")
print('barycentric corr:\tbcv')
print('Doppler shift:\tRVShift')
print('spec type to value:\tsval')
print('spec value to type:\tsvaltotype')
print('')
print('Mamajek Suite:')
print('Parameters:')
print(labs)
print('param from SpT:\tmamajek')
print('param from sval:\tmamajeksv')
print('param from SpT w/ err:\tmam')
print('param from other param w/ err:\tmam')

def rcpar():
    plt.rcParams.update({'font.size': 8,'lines.linewidth':1, 'font.family':'serif','mathtext.fontset':'dejavuserif',
                         'xtick.major.pad':2,'ytick.major.pad':2,'xtick.minor.pad':1,'ytick.minor.pad':1,'axes.labelpad':0,
                         'figure.constrained_layout.h_pad':0,'figure.constrained_layout.w_pad':0,'savefig.pad_inches':0.02,
                        'axes.titlepad':2})

def saveps(edir,tit):
    plt.savefig(edir+tit+'.eps',bbox_inches='tight', format='eps')