# -*- coding: utf-8 -*-
"""
Created on Fri Sep 23 19:35:02 2022

@author: ivandep
"""

# Import modules
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from JanbuFunc import Janbufuncs,failureSurfaces,discretizeDomain
from scipy.stats import norm

# Prepare inputs---------------------------------------------------------------
# Number of surfaces
Nsurf=12
# Slope angle
angle=30.0
# Range of depths
d=np.linspace(0,1,num=Nsurf)
# Number of points along surface
Np=30
# Slope height
H=18.0
# Number of discretization points in x direction
Ndx=31
# Boundaries
xMin=-40; xMax=80; yMin=-30
# Unit weight
gamma=19.0

# Fit functions to Janbu diagrams
regN0, polyN0, reg_x, poly_x, reg_y, poly_y=Janbufuncs()
# Get stability coefficients and coordinates of failure surfaces
N0,Coord=failureSurfaces(regN0, polyN0, reg_x, poly_x, reg_y, poly_y, angle,Nsurf,Np,H,d) 
# Discretize the domain
Coord,Depth=discretizeDomain(Coord,Ndx,xMin,xMax,yMin,angle,H)

# Cost of failure
Cf=1

# Cost of investigation
Ci=0.01

# Measurement error
eps=1.0

# Model error
epsM=0.05

# Minimum factor of safety
FsLim=1.0


# Correlation length
theta_x=50.0
theta_z=50.0

# Mean
muPrior=np.reshape(40+0.2*9*Depth,(np.shape(Depth)[0],1))

muActual=np.reshape(40+0.2*9*Depth,(np.shape(Depth)[0],1))

# Deviation
sigmaPrior=muPrior*0.2*0+(40*0.2)

sigmaActual=muActual*0.03*0+(40*0.03)

# Correlation functions
# Markov correlation function
def markov(x1,z1,x2,z2,thetax,thetaz):
    return np.exp(-2*np.abs(x1-x2)/thetax)*np.exp(-2*np.abs(z1-z2)/thetaz)

# Ellipsoidal
def ellipsoid(x1,z1,x2,z2,thetax,thetaz):
    return np.exp(-2*np.sqrt(((x1-x2)/thetax)**2+((z1-z2)/thetaz)**2))

def FsSlopeStability(su,N0,Coord,Ns,Np,gamma,H):
    Fs=np.zeros(Ns)
    
    for i in range(Ns):
        Fs[i]=np.mean(su[i*Np:(i+1)*Np])*N0[i]/(gamma*H)
    
    return np.min(Fs)

def slopelPf(Nsamp,Mean,Cov,epsM,FsLim,N0,Coord,Nsurf,Np,gamma,H):
    # Allocate vector to store Fs values
    Fs=np.zeros(Nsamp)
    
    # Decompose the covariance matrix
    A=np.linalg.cholesky(Cov)
    
    # Random vector
    U=np.random.normal(loc=0, scale=1, size=(Npoints,Nsamp))

    # Generate realisations
    Val=np.dot(Mean,np.ones((1,Nsamp)))+(np.dot(A,U))
    
    for i in range(Nsamp):
        # Generate a random realisation
        Fs[i]=FsSlopeStability(Val[:,i],N0,Coord,Nsurf,Np,gamma,H)
    
    Pf=np.sum(Fs<=FsLim)/Nsamp
    
    return  Pf, Fs

def slopelPfLS(Nsamp,Mean,Cov,epsM,FsLim,N0,Coord,Nsurf,Np,gamma,H):
    # Allocate vector to store Fs values
    Pfi=np.zeros(Nsamp)
    
    # Decompose the covariance matrix
    A=np.linalg.cholesky(Cov)
    
    # Random vector
    U=np.random.normal(loc=0, scale=1, size=(Npoints,Nsamp))

    # Generate realisations
    Val=np.dot(Mean,np.ones((1,Nsamp)))+(np.dot(A,U))
    
    for i in range(Nsamp):
        # Generate a random realisation
        Fs=FsSlopeStability(Val[:,i],N0,Coord,Nsurf,Np,gamma,H)
        Pfi[i]=norm.cdf(-(Fs-FsLim)/epsM)
    
    Pf=np.mean(Pfi)
    
    return  Pf, Pfi

def slopelPfFOSM(Mean,Cov,epsM,FsLim,N0,Coord,Nsurf,Np,gamma,H):
    # Constant
    Pd=gamma*H
    # Allocate vector of Pf
    Pf=np.zeros(Nsurf)
    
    for i in range(Nsurf):
        muFs=np.mean(Mean[i*Np:(i+1)*Np])*N0[i]/Pd
        sigFs=np.sqrt((N0[i]/(Pd*Np))**2*np.sum(np.diag(Cov)[i*Np:(i+1)*Np])+epsM**2)
        beta=(muFs-FsLim)/sigFs
        Pf[i]=norm.cdf(-beta)
    
    return  np.max(Pf), Pf

def trialAnalysis(Coord,Nt,Ntsamp,Mean,Cov,eps,epsM,FsLim,N0,Nsurf,Np,gamma,H):
    # Trial locations--------------------------------------------------------------
    # Number of points
    Npoints=np.shape(Coord)[0]
    # Number of trial location
    #Nt=10

    # Number of samples per trial
    #Ntsamp=100

    # Allocate failure probability array
    Pft=np.zeros((Nt,Ntsamp))

    # Select trial locations randomly
    #trialLoc=np.random.choice(np.linspace(Np*Nsurf,Npoints-1,
    #                                      num=Npoints-Np*Nsurf,
    #                                      dtype=int),size=Nt)
    trialLoc=np.linspace(Np*Nsurf,Npoints-1,num=Npoints-Np*Nsurf,dtype=int)
    
    # Mean at trial locations
    meanTrial=Mean[trialLoc]

    # Covariance at trial location
    covTrial=Cov[trialLoc,:][:,trialLoc]

    # Cholesky decomposition
    Atrial=np.linalg.cholesky(0*covTrial+eps**2*np.eye(np.size(trialLoc)))

    # Random vector
    Utrial=np.random.normal(loc=0, scale=1, size=(np.size(trialLoc),Ntsamp))

    # Trial values
    trialVal=np.dot(meanTrial,np.ones((1,Ntsamp)))+(np.dot(Atrial,Utrial))

    # Iterate over trial locations
    for i in range(Nt):
        # Allocate the observation vector
        Hobs=np.zeros((1,Npoints))
        
        # Change to 1
        Hobs[0,trialLoc[i]]=1
        
        for j in range(Ntsamp):
            
            # Vector of observations
            Obs=np.zeros((1,1))
            Obs[0,0]=trialVal[i,j]
            
            # Covariance matrix of observations
            CovObs=np.zeros((1,1))
            # Independent measurements error
            CovObs[0,0]=eps**2
            
            # Conditional mean
            MeanCon=Mean+np.dot(Cov,np.dot(np.transpose(Hobs),np.dot(
                np.linalg.inv(np.dot(Hobs,np.dot(Cov,np.transpose(Hobs)))
                                              +CovObs),(Obs-np.dot(Hobs,Mean)))))
            
            # Conditional variance
            CovCon=Cov-np.dot(Cov,np.dot(np.transpose(Hobs),np.dot(
                np.linalg.inv(np.dot(Hobs,np.dot(Cov,np.transpose(Hobs)))+
                              CovObs),np.dot(Hobs,Cov))))
            
            # Calculate failure probability
            #Pftemp, Fs=slopelPf(Ntsamp,MeanCon,CovCon,epsM,FsLim,N0,Coord,Nsurf,Np,gamma,H)
            #Pftemp, Pfall=slopelPfFOSM(MeanCon,CovCon,epsM,FsLim,N0,Coord,Nsurf,Np,gamma,H)#setlPf(Ns,MeanCon,CovCon,x,z,dSigma,deltaLim)
            Pftemp, Pfall=slopelPfLS(Ns,MeanCon,CovCon,epsM,FsLim,N0,Coord,Nsurf,Np,gamma,H)
            Pft[i,j]=Pftemp
        
            
    return trialLoc, Pft
    
def randomFieldUpdate(Val,Mean,Cov,Coord,loc,eps):
    # Take the measurement at the selected location and update random field
    Npoints=np.shape(Coord)[0]

    # Observation vector
    H=np.zeros((1,Npoints))
    # Change to 1
    H[0,loc]=1

    # Vector of actual observations
    actObs=np.zeros((1,1))
    actObs[0,0]=Val[loc,0]

    # Covariance of actual observations
    CovActObs=np.zeros((1,1))
    # Measurement error
    CovActObs[0,0]=eps**2

    # Conditional mean
    MeanCon=Mean+np.dot(Cov,np.dot(np.transpose(H),np.dot(
        np.linalg.inv(np.dot(H,np.dot(Cov,np.transpose(H)))
                                      +CovActObs),(actObs-np.dot(H,Mean)))))

    # Conditional variance
    CovCon=Cov-np.dot(Cov,np.dot(np.transpose(H),np.dot(
        np.linalg.inv(np.dot(H,np.dot(Cov,np.transpose(H)))+CovActObs),
        np.dot(H,Cov))))
    
    return MeanCon,CovCon

# Discretization--------------------------------------------------------------
# Number of points
Npoints=np.shape(Coord)[0]

# Correlation matrix
Cov=np.zeros((Npoints,Npoints))

for i in range(Npoints):
    for j in range(Npoints):
        Cov[i,j]=ellipsoid(Coord[i,0],Coord[i,1],Coord[j,0],Coord[j,1],theta_x,theta_z)

# Prior mean and covariance----------------------------------------------------
# Mean vector
Mean=muPrior

# Scale the covariance with prior variance
C=np.multiply(np.dot(sigmaPrior**2,np.ones((1,np.shape(Depth)[0]))),Cov)

# Actual realisation-----------------------------------------------------------
# Scale the covariance with actual variance
CActual=np.multiply(np.dot(sigmaActual**2,np.ones((1,np.shape(Depth)[0]))),Cov)
# Cholesky decomposition
A=np.linalg.cholesky(CActual)

# Random vector
U=np.random.normal(loc=0, scale=1.0, size=(Npoints,1))

# realisation
Val=muActual+(np.dot(A,U))

# Plot
plt.scatter(Coord[Np*Nsurf:,0],Coord[Np*Nsurf:,1],c=Val[Np*Nsurf:])
plt.colorbar()
plt.xlabel('$ x $')
plt.ylabel('$ y $')
# plt.savefig(r'C:\Users\ivandep\Desktop\Work\Naturfareforum\SlopeStability\actualval'+str(Ci)+'.png',dpi=600)
plt.show()

# Calculate probability
# Number of samples
Ns=100

# Probability and samples
# Pf0, Fs=slopelPf(Ns,Mean,C,epsM,FsLim,N0,Coord,Nsurf,Np,gamma,H)

Pf0, Pfall=slopelPfLS(Ns,Mean,C,epsM,FsLim,N0,Coord,Nsurf,Np,gamma,H)
print('P_F=% .3e' % Pf0)

# Plot samples
# plt.hist(Fs,bins=20)
# plt.ylabel('Frequency')
# plt.xlabel('$ F_S $')
# plt.grid()
# plt.savefig(r'C:\Users\ivandep\Desktop\Work\Naturfareforum\SlopeStability\hist-FS-init'+str(Ci)+'.png',dpi=600)
# plt.show()
# plt.show()

# Initial cost
C0=Ci+Cf*Pf0

# Number of investigations
Ni=10

# Number of trial location
Nt=np.shape(Coord)[0]-Np*Nsurf

# Number of samples per trial
Ntsamp=10

# Average trial cost function
Cta=np.zeros((Ni,Nt))

# Average trial value of information
VoIta=np.zeros((Ni,Nt))

# Vector of locations
actLoc=np.zeros((Ni),dtype=int)

# Vector of actual probabilities
Pf=np.zeros(Ni+1)
Pf[0]=Pf0

# Cost of information
Ca=np.zeros(Ni+1)
Ca[0]=0.0

# Vector of actual value of information
VoIa=np.zeros(Ni+1)

# Initialize mean vector
upMean=Mean
# Initialize covariance
upCov=C

for i in range(Ni):
    
    print('Ni=%d' % i)
    
    # Trial analysis
    trialLoc, Pft= trialAnalysis(Coord,Nt,Ntsamp,upMean,upCov,eps,epsM,FsLim,N0,
                                 Nsurf,Np,gamma,H)
    
    # Calculate cost
    Cta[i,:]=np.mean(Ci*(i+1)+Cf*Pft,1)
    
    # Calculate trial value of information
    VoIta[i,:]=Cf*(Pf0-np.mean(Pft,1))
    
    plt.plot(VoIta[i,:])
    plt.xlabel('Trial investigations')
    plt.ylabel('$ VoI $')
    plt.grid()
    # plt.savefig(r'C:\Users\ivandep\Desktop\Work\Naturfareforum\SlopeStability\trialVoI-'+str(i)+str(Ci)+'.png',dpi=600)
    plt.show()
    
    # Plot VoI with locations
    # Plot the actual values and measurement locations
    plt.scatter(Coord[trialLoc,0],Coord[trialLoc,1],c=VoIta[i,:])
    plt.colorbar()
    #plt.clim(12, 20) 
    plt.xlabel('$ x $')
    plt.ylabel('$ z $')
    # plt.savefig(r'C:\Users\ivandep\Desktop\Work\Naturfareforum\SlopeStability\VoIloc-'+str(i)+str(Ci)+'.png',dpi=600)
    plt.show()
    
    # Location with the maximum VoI
    actLoc[i]=trialLoc[np.argmax(VoIta[i,:])]
    
    # Take measurement and update mean and covariance
    upMean,upCov=randomFieldUpdate(Val,upMean,upCov,Coord,actLoc[i],eps)
        
    # Plot the actual values and measurement locations
    plt.scatter(Coord[Np*Nsurf:,0],Coord[Np*Nsurf:,1],c=Val[Np*Nsurf:])
    plt.colorbar()
    #plt.clim(12, 20) 
    plt.scatter(Coord[actLoc[0:i+1],0],Coord[actLoc[0:i+1],1],c='r',marker='o')
    plt.xlabel('$ x $')
    plt.ylabel('$ z $')
    # plt.savefig(r'C:\Users\ivandep\Desktop\Work\Naturfareforum\SlopeStability\val-'+str(i)+'.png',dpi=600)
    plt.show()
    
    # Plot updated mean with measurement locations
    plt.scatter(Coord[Np*Nsurf:,0],Coord[Np*Nsurf:,1],c=upMean[Np*Nsurf:])
    plt.colorbar()
    #plt.clim(12, 20) 
    plt.scatter(Coord[actLoc[0:i+1],0],Coord[actLoc[0:i+1],1],c='r',marker='o')
    plt.xlabel('$ x $')
    plt.ylabel('$ z $')
    # plt.savefig(r'C:\Users\ivandep\Desktop\Work\Naturfareforum\SlopeStability\meancon-'+str(i)+'.png',dpi=600)
    plt.show()
    
    # Plot updated variance with measurement locations
    plt.scatter(Coord[Np*Nsurf:,0],Coord[Np*Nsurf:,1],c=np.sqrt(np.diag(upCov))[Np*Nsurf:])
    plt.colorbar()
    #plt.clim(0, 1.6) 
    plt.scatter(Coord[actLoc[0:i+1],0],Coord[actLoc[0:i+1],1],c='r',marker='x')
    plt.xlabel('$ x $')
    plt.ylabel('$ z $')
    # plt.savefig(r'C:\Users\ivandep\Desktop\Work\Naturfareforum\SlopeStability\stdevcon-'+str(i)+'.png',dpi=600)
    plt.show()

    # Check updated failure probability
    # Probability and samples
    # Pf[i], Fs=slopelPf(Ns,upMean,upCov,epsM,FsLim,N0,Coord,Nsurf,Np,gamma,H)
    Pf[i], Pfall=slopelPfLS(Ns,upMean,upCov,epsM,FsLim,N0,Coord,Nsurf,Np,gamma,H)
    #setlPf(Ns,upMean,upCov,x_d,z_d,dSigma,deltaLim)
    
    # Plot samples
    # plt.hist(Fs,bins=20)
    # plt.ylabel('Frequency')
    # plt.xlabel('$ F_S $')
    # plt.grid()
    # plt.savefig(r'C:\Users\ivandep\Desktop\Work\Naturfareforum\SlopeStability\hist-settlement-'+str(i)+str(Ci)+'.png',dpi=600)
    # plt.show()
    
    # Cost of information 
    Ca[i+1]=Ci*(i+1)
    
    # Value of information of the first sample
    VoIa[i+1]=Cf*(Pf0-Pf[i])
    
# Final results
plt.plot(VoIa,marker='o')
plt.plot(Ca,'--',marker='x')
plt.ylabel('VoI,cost of investigation')
plt.xlabel('Number of investigations')
plt.grid()
plt.legend(['VoI','Cost of investigation'])
# plt.savefig(r'C:\Users\ivandep\Desktop\Work\Naturfareforum\SlopeStability\voicost.png',dpi=600)
plt.show()

plt.plot(VoIa-Ca,marker='o')
plt.ylabel('VoI-cost of investigation')
plt.xlabel('Number of investigations')
plt.grid()
# plt.savefig(r'C:\Users\ivandep\Desktop\Work\Naturfareforum\SlopeStability\voi-cost.png',dpi=600)
plt.show()

plt.plot(Pf,marker='o')
plt.ylabel('Failure probability')
plt.xlabel('Number of investigations')
plt.grid()
# plt.savefig(r'C:\Users\ivandep\Desktop\Work\Naturfareforum\SlopeStability\pf.png',dpi=600)
plt.show()
