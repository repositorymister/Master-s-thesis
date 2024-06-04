# -*- coding: utf-8 -*-
"""
Created on Wed Oct 26 08:22:50 2022

@author: ivandep
"""
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from scipy.stats import norm
from shapely.geometry import Polygon, Point
from shapely.ops import nearest_points
import pandas as pd
import time 


start_time = time.time()
def Janbufuncs():    
    
    # Read values from the chart
    
    N0_data=np.array([
        [12.5,	10.9,	10,	    9.2,	8,	6.7,	6.2,	6,	5.8],
        [10.08,	9.25,	8.5,	7.75,	6.85,	6.2,	5.9,	5.8,	5.76],
        [8.82,	8.05,	7.25,	6.75,	6.45,	5.98,	5.78,	5.75,	5.75],
        [7.94,	7.25,	6.5,	6.4,	6.05,	5.85,	5.75,	5.74,	5.74],
        [7.28,	6.65,	6.2,	6.1,	5.92,	5.82,	5.74,	5.73,	5.73],
        [6.6,	6.3,	5.85,	5.85,	5.8,	5.8,	5.73,	5.72,	5.72],
        [6.22,	5.98,	5.75,	5.75,	5.75,	5.75,	5.72,	5.71,	5.71],
        [5.8,	5.6,	5.6,	5.6,	5.6,	5.6,	5.6,	5.6,	5.6]])
    
    angle=np.array([15, 20, 25, 30, 35, 40, 45, 50])
    
    d=np.array([0, 0.1, 0.2, 0.3, 0.5, 1.0, 1.5, 2.0, 3.0])
    
    x1,x2=np.meshgrid(angle,d)
    
    
    x=np.array([np.ndarray.flatten(x1.T),np.ndarray.flatten(x2.T)]).T
    
    y=np.ndarray.flatten(np.log(N0_data))
    
    id_reg=y<np.log(11.1)
    
    polyN0 = PolynomialFeatures(degree=5)
    x_r=polyN0.fit_transform(x[id_reg,:])
    y_r=y[id_reg]
    
    regN0 = LinearRegression()
    
    regN0.fit(x_r,y_r)
    
    # coef=regN0.coef_
    
    pred=np.exp(regN0.predict(x_r))
    
    plt.plot(np.exp(y_r)); plt.plot(pred)
    plt.show()
    
    # Predict original data
    pred1=np.exp(regN0.predict(polyN0.fit_transform(x)))
    
    plt.plot(angle,np.reshape(np.exp(y),(np.size(angle),np.size(d))),label=['d=0', 'd=0.1', 'd=0.2','d=0.3','d=0.5','d=1','d=1.5','d=2','d=3'])
    plt.plot(angle,np.reshape(pred1,(np.size(angle),np.size(d))),c='C4',ls='--')
    plt.plot(angle,11+0*angle,label='Limit',ls='-.',c='gray')
    plt.legend()
    plt.grid()
    plt.xlabel('Slope angle [$ ^{\circ} $]')
    plt.ylabel('Stability number')
    plt.show()
    
    
    # X-coordinate-----------------------------------------------------------------
    x_coord=np.array([2.5, 1.8, 1.37, 1.01, 0.85, 0.68, 0.5, 0.35])
    
    poly_x = PolynomialFeatures(degree=3)
    ang_x=poly_x.fit_transform(np.reshape(angle,(np.size(angle),1)))
    
    
    reg_x = LinearRegression()
    
    reg_x.fit(ang_x,x_coord)
    
    #coef_x=reg_x.coef_
    
    pred_x=reg_x.predict(ang_x)
    
    plt.plot(angle,x_coord,'C1'); plt.plot(angle,pred_x,c='C4',ls='--')
    plt.legend(['Diagram','Prediction'])
    plt.grid()
    plt.xlabel('Slope angle [$ ^{\circ} $]')
    plt.ylabel('Unit abcissa of center')
    plt.show()
    
    # Y-coordinate-----------------------------------------------------------------
    
    d_y=np.array([0,1,2,3])
    
    y_coord=np.array([
        [2.4,	2.7,	2.9,	3.32],
        [2,	    2.25,	2.58,	3.1],
        [1.72,	1.96,	2.4,	2.98],
        [1.56,	1.85,	2.35,	2.91],
        [1.48,	1.75,	2.26,	2.86],
        [1.41,	1.7,	2.22,	2.84],
        [1.4,	1.68,	2.22,	2.835],
        [1.41,	1.68,	2.2,	2.835]
    ])
    
    
    x3,x4=np.meshgrid(angle,d_y)
    
    x_y=np.array([np.ndarray.flatten(x3.T),np.ndarray.flatten(x4.T)]).T
    
    y_y=np.ndarray.flatten(y_coord)
    
    
    poly_y = PolynomialFeatures(degree=5)
    ang_y=poly_y.fit_transform(x_y)
    
    reg_y = LinearRegression()
    
    reg_y.fit(ang_y,y_y)
    
    #coef_y=reg_y.coef_
    
    pred_y=reg_y.predict(ang_y)
    
    plt.plot(angle,np.reshape(pred_y, (np.shape(y_coord))),c='C4',ls='--')
    plt.plot(angle,y_coord,label=['d=0', 'd=1', 'd=2','d=3']) 
    plt.legend()
    plt.grid()
    plt.xlabel('Slope angle [$ ^{\circ} $]')
    plt.ylabel('Unit ordinate of center')
    plt.show()


    return regN0, polyN0, reg_x, poly_x, reg_y, poly_y


def JanbuPredict(regN0, polyN0, reg_x, poly_x, reg_y, poly_y, angle,d):

    # Predict N0 value
    N0=np.exp(regN0.predict(polyN0.fit_transform(np.reshape(np.array([angle,d]),(1,2)))))
    # Predict x-coordinate
    x=reg_x.predict(poly_x.fit_transform(np.reshape(np.array([angle]),(1,1))))
    # Predict y coordinate
    y=reg_y.predict(poly_y.fit_transform(np.reshape(np.array([angle,d]),(1,2))))
    
    return N0, x, y


def failureSurface(d,x,y,H,beta,Np):
    # Slope height
    #H=10.0
    
    # X-coordinate of the slope center
    p=x*H
    
    # Y-coordinate of the slope center
    q=y*H
    
    # Radius
    R=H*(y+d)
    
    # Slope line inclination
    a=np.tan(np.radians(beta))
    # Line constant
    b=0.0
    
    # First intersection
    x1=((p-a*(b-q))+np.sqrt((p-a*(b-q))**2-(1+a**2)*(p**2+(b-q)**2-R**2)))/((1+a**2))
    y1=a*x1+b
    
    # Second intersection
    x2=((p-a*(b-q))-np.sqrt((p-a*(b-q))**2-(1+a**2)*(p**2+(b-q)**2-R**2)))/((1+a**2))
    y2=a*x2+b
    
    
    # Intersections with the horizontal lines
    # y=0
    y3=0
    x3=p-np.sqrt(R**2-q**2)
    # y=H
    y4=H
    x4=p+np.sqrt(R**2-(H-q)**2)
    
    # Check intersections
    if(y1>H):
        xEnd=x4
        yEnd=y4
    else:
        xEnd=x1
        yEnd=y1
    
    if(y2<0):
        xStart=x3
        yStart=y3
    else:
        xStart=x2
        yStart=y2
        
    # First angle
    alpha1=1.5*np.pi-np.arctan(np.abs(p-xStart)/np.abs(q-yStart))
    
    # Second angle
    alpha2=2*np.pi-np.arctan(np.abs(q-yEnd)/np.abs(p-xEnd))
    
    # Create circle
    alpha=np.linspace(alpha1,alpha2,num=Np)
    xCircle=p+R*np.cos(alpha)
    yCircle=q+R*np.sin(alpha)
    
    plt.plot(xCircle,yCircle)
    plt.plot([np.min([xStart, -H]),0,H/np.tan(np.radians(beta)),
              np.max([H/np.tan(np.radians(beta))+H,xEnd])],[0,0,H,H])
    plt.show()
    
    return xCircle,yCircle

def failureSurfaces(regN0, polyN0, reg_x, poly_x, reg_y, poly_y, angle,Ns,Np,H,d):
    N0=np.zeros(Ns)
    x=np.zeros(Ns)
    y=np.zeros(Ns)
    xCoord=np.zeros((Np,Ns))
    yCoord=np.zeros((Np,Ns))
    
    
    for i in range(Ns):
        # Calculate stability coefficient and the center of failure surface
        N0[i], x[i], y[i] = JanbuPredict(regN0, polyN0, reg_x, poly_x, reg_y, 
                                         poly_y, angle,d[i])
    
        # Calculate points along surface
        xCoord[:,i],yCoord[:,i]=failureSurface(d[i],x[i],y[i],H,angle,Np)
    
    plt.plot(xCoord,yCoord)
    plt.plot([np.min(xCoord),0,H/np.tan(np.radians(angle)),np.max(xCoord)],[0,0,H,H])
    plt.savefig('Failure Surfaces')
    plt.show()
    
    # Create coordinate vector
    Coord=np.zeros((Ns*Np,2))

    Coord[:,0]=np.reshape(xCoord.T,(Ns*Np))
    Coord[:,1]=np.reshape(yCoord.T,(Ns*Np))
    
    return N0, Coord

def discretizeDomain(Coord,Ndx,xMin,xMax,yMin,angle,H, theta_y):
    # Number of divisions in x direction
    x_d=np.linspace(xMin,xMax,num=Ndx)
    # Minimum value in y direction, yMin
    
    
    for i in range(Ndx):
        # Find y coordinate
        if(x_d[i]<=0):
            yMax=0
            # Number of points
            Nyp=np.int16(np.round(((yMax-yMin) / theta_y) *2)) + 1
            # Story coordinates temporary
            CoordTemp=np.zeros((Nyp,2))
            CoordTemp[:,0]=x_d[i]
            CoordTemp[:,1]=np.linspace(yMin,yMax,num=Nyp)
            # Append values
            Coord=np.vstack((Coord,CoordTemp))
            
        elif(x_d[i]>0 and x_d[i]<H/np.tan(np.radians(angle))):
            yMax=np.tan(np.radians(angle))*x_d[i]
            # Number of points
            Nyp=np.int16(np.round(((yMax-yMin) / theta_y) *2)) + 1
            # Story coordinates temporary
            CoordTemp=np.zeros((Nyp,2))
            CoordTemp[:,0]=x_d[i]
            CoordTemp[:,1]=np.linspace(yMin,yMax,num=Nyp)
            # Append values
            Coord=np.vstack((Coord,CoordTemp))
        else:
            yMax=H
            # Number of points
            Nyp=np.int16(np.round(((yMax-yMin) / theta_y) *2)) + 1 
            # Story coordinates temporary
            CoordTemp=np.zeros((Nyp,2))
            CoordTemp[:,0]=x_d[i]
            CoordTemp[:,1]=np.linspace(yMin,yMax,num=Nyp)
            # Append values
            Coord=np.vstack((Coord,CoordTemp))
    
    # Plot all of the points
    plt.scatter(Coord[:,0],Coord[:,1])
    plt.show()
    
    # Calculate depths
    Depth=np.zeros(np.shape(Coord)[0])
    
    for i in range(np.shape(Coord)[0]):
        # Find y coordinate
        if(Coord[i,0]<=0):
            yMax=0
            # Depth
            Depth[i]=yMax-Coord[i,1]
            
        elif(Coord[i,0]>0 and Coord[i,0]<H/np.tan(np.radians(angle))):
            yMax=np.tan(np.radians(angle))*Coord[i,0]
            # Depth
            Depth[i]=yMax-Coord[i,1]
        else:
            yMax=H
            # Depth
            Depth[i]=yMax-Coord[i,1]

    return Coord,Depth
    
# Number of surfaces
Nsurf=12
# Slope height
H=10.0
# Slope angle
angle=np.degrees(np.pi / 12)
# Range of depths
d=np.linspace(0,1,num=Nsurf)
# Number of points along surface
Np=30
# Number of discretization points in x direction
Ndx=61
# Boundaries
xMin=-40; xMax=80; yMin=-30
# Unit weight
gamma=19.0

# Fit functions to Janbu diagrams
regN0, polyN0, reg_x, poly_x, reg_y, poly_y=Janbufuncs()
# Get stability coefficients and coordinates of failure surfaces
N0,Coord=failureSurfaces(regN0, polyN0, reg_x, poly_x, reg_y, poly_y, angle,Nsurf,Np,H,d)

# Discretize the domain, domain discretization based on 2points per SOF property-set only for y-discretizations
#X-discretizations based on judgement, probabaly should be too high because it signfies boreholes(measurement points)
Coord,Depth=discretizeDomain(Coord,Ndx,xMin,xMax,yMin,angle,H, 5.0)

Npoints = Coord.shape[0]
trialLoc=np.linspace(Np*Nsurf,Npoints-1,num=Npoints-Np*Nsurf,dtype=int)

#Extract each 'x' and total y-discretizations(counts) along it for a borehole
unique_xs, first_indices, counts = np.unique(Coord[:, 0][trialLoc], return_counts=True, return_index=True)

# Cost of failure
Cf=1000000

# Cost of investigation
borehole_depth = Depth[Np*Nsurf:][first_indices]
CPTU_cost = 8000 + 1000 * (borehole_depth - 20)
Ci = CPTU_cost

#Measurement error for layer
eps_layer = 0.2

# Model error
epsM=0.05

# Minimum factor of safety
FsLim=1.0

# Correlation length
theta_x=46.0
theta_z=0.8

#Original(prior) lognormal parameters of the field
def original_parameters(su0, rsu, z):
    su = su0 + rsu * z
    return su.reshape(-1, 1) 

#Convert to Equivalent Normal Parameters
def equiv_norm(mu, sigma):
    normal_sigma = np.sqrt(np.log(1 + sigma ** 2 / mu ** 2))
    normal_mu = np.log(mu) - 0.5 * normal_sigma ** 2
    return normal_mu, normal_sigma


#CPTU Nkt and measurement interpretation
mean_eps_CPT = 1
COV_eps_CPT = 0.1
eps = mean_eps_CPT

mean_Nk = 9
COV_Nk = 0.01

def CPTU_interpretation(Depth, su):
    sigma_Nk = mean_Nk * COV_Nk
    sigmaln_Nk = np.sqrt(np.log(1 + (sigma_Nk ** 2 / mean_Nk ** 2)))
    muln_Nk = np.log(mean_Nk) - ((1 / 2) * sigmaln_Nk ** 2)
    Nkt = np.random.lognormal(muln_Nk, sigmaln_Nk, su.shape)

    mu_eps = mean_eps_CPT
    COV_eps = COV_eps_CPT
    sigma_eps = mu_eps * COV_eps
    sigmaln_eps = np.sqrt(np.log(1 + (sigma_eps ** 2 / mu_eps ** 2)))
    muln_eps = np.log(mu_eps) - ((1 / 2) * sigmaln_eps ** 2)
    eps = np.random.lognormal(muln_eps,sigmaln_eps,su.shape)
    sigma_v0 = np.maximum(Depth.reshape(-1, 1) * gamma, 1e-10)
    qt_measured = (su*Nkt + np.log(sigma_v0))/eps

    su_observed = (qt_measured - np.log(sigma_v0))/mean_Nk
    return su_observed

def check(x, layer_boundary):
    coordinate_layer = np.hstack((x.reshape(-1, 1), layer_boundary))
    slope_boundary = np.array([[xMin, yMin], [xMax, yMin], [xMax, H], [H/np.tan(np.pi / 12), H], [0,0], [xMin, 0]])
    polygon = Polygon(slope_boundary)
    corrected_boundary = []
    for point in coordinate_layer:
        point_shapely = Point(point)
        if not polygon.contains(point_shapely):
            # Get the nearest point on the polygon if outside
            nearest_geom = nearest_points(point_shapely, polygon.boundary)[1]
            corrected_boundary.append(list(nearest_geom.coords[0]))
        else:
            corrected_boundary.append(point)
    return np.array(corrected_boundary)[:,1].reshape(-1, 1)

# Correlation functions
# Markov correlation function
def markov(x1,z1,x2,z2,thetax,thetaz):
    return np.exp(-2*np.abs(x1-x2)/thetax)*np.exp(-2*np.abs(z1-z2)/thetaz)

# Ellipsoidal
def ellipsoid(x1,z1,x2,z2,thetax,thetaz):
    return np.exp(-2*np.sqrt(((x1-x2)/thetax)**2+((z1-z2)/thetaz)**2))

#Creates a multi-linear approximation of mean layer boundary
def piecewise_function(x, x1, y1, x2, y2):
    if x <= x1:
        return y1
    elif x1 < x < x2:
        return ((y2 - y1) / (x2 - x1)) * (x - x1) + y1
    elif x >= x2:
        return y2
    
#Merges two layers based on the boundary property
def complete_soil(coordinates, RF1, RF2, dividing_y, Nsamp):
    N, Nsamp = RF1.shape
    # Initialize arrays for assigned soil properties and binary indicators
    assigned_su = np.zeros((N, Nsamp))
    binary = np.zeros((N, Nsamp))

    for i, (a, b) in enumerate(coordinates):
        
        # Check where condition for the boundary is met
        second_layer = b < dividing_y[i]
        first_layer = ~second_layer #simply means not 2nd layer  

        # Assign values based on the condition
        assigned_su[i, second_layer] = RF2[i, second_layer]
        assigned_su[i, first_layer] = RF1[i, first_layer]

        # Set binary indicators
        binary[i, second_layer] = 1  # Indicates RF2 was used
        binary[i, first_layer] = 0  # Indicates RF1 was used

    return binary, assigned_su

def FsSlopeStability(su,N0,Coord,Ns,Np,gamma,H):
    Fs=np.zeros(Ns)
   
    for i in range(Ns):
        Fs[i]=np.mean(su[i*Np:(i+1)*Np])*N0[i]/(gamma*H)
   
    return np.min(Fs), np.argmin(Fs)

def slopelPfLS(Nsamp,Mean1,Cov1, Mean2, Cov2, Mean_layer, Cov_layer, epsM,FsLim,N0,Coord,Nsurf,Np,gamma,H):
    
    # Allocate vector to store Fs values
    Pfi = np.zeros(Nsamp)
       
    # Generate realisations
    Val1 = np.random.multivariate_normal(Mean1.flatten(), Cov1, size=Nsamp).T
    Val2 = np.random.multivariate_normal(Mean2.flatten(), Cov2, size=Nsamp).T
    
    '''
    Some numerical imprecision cause Covariance matrices not to be positive-definite(strictly speaking). Numpy built-in function
    multivariate_normal used here solves that issue. Infact a infinitesimal jitter maybe added along the diagonal to make the
    eigenvalues positive, consquently making the matrix positive definite. Scipy.linalg.cholesky provides exact info about
    which element violates postive definiteness.
    '''
    
    Val_layer = check(Coord[:,0], np.random.multivariate_normal(Mean_layer.flatten(), Cov_layer).reshape(-1, 1)).flatten()
    
    #Independent soil realizations(Val1 and Val2) are merged along the layer boundary(Val_layer)
    _, Val =  complete_soil(Coord, Val1, Val2, Val_layer, Nsamp)
    
    # Convert back to lognormal realizations----------------------------------->
    Val=np.exp(Val) 
    slip_circle = np.zeros(Nsamp)
    
    for i in range(Nsamp):
        # Generate a random realisation
        Fs, circle=FsSlopeStability(Val[:,i],N0,Coord,Nsurf,Np,gamma,H)
        Pfi[i]=norm.cdf(-(Fs-FsLim)/epsM)
        slip_circle[i] = circle
        
    Pf=np.mean(Pfi)
   
    return  Pf, Pfi, slip_circle


def trialAnalysis(Coord,Nt,Ntsamp,Mean1,Cov1, Mean2, Cov2, Mean_layer, Cov_layer, eps, eps_layer, 
                  epsM,FsLim,N0,Nsurf,Np,gamma,H):
    # Trial locations--------------------------------------------------------------
    Npoints=np.shape(Coord)[0]

    # Allocate failure probability array
    Pft=np.zeros((Ndx, Ntsamp))

    trialLoc=np.linspace(Np*Nsurf,Npoints-1,num=Npoints-Np*Nsurf,dtype=int)
    n = trialLoc[0]
    # Mean at trial locations
    meanTrial1=Mean1[trialLoc]
    meanTrial2=Mean2[trialLoc]

    # Covariance at trial location
    covTrial1=Cov1[trialLoc,:][:,trialLoc]
    covTrial2=Cov2[trialLoc,:][:,trialLoc]

    # Trial values------------------------------------------------------------->
    trialVal1=meanTrial1
    trialVal2=meanTrial2

    #Extract each 'x' and total y-discretizations(counts) along it for a borehole
    unique_xs, first_indices, counts = np.unique(Coord[:, 0][trialLoc], return_counts=True, return_index=True)

    first_indices +=  n 

    for i, (x, start_idx , count) in enumerate(zip(unique_xs, first_indices, counts)):

        #I is introduced to utilize Numpy's indexing capabilties
        I = np.eye(count)

        #H-linear function matrix indexed by 1's along the borehole
        Hobs = np.zeros((count,Npoints))
        Hobs[:,start_idx : start_idx + count] = I

        for j in range(Ntsamp):

            # Vector of observations
            x = start_idx - n
            Obs1 =CPTU_interpretation(Depth[x:x + count], trialVal1[x:x + count]) #----------------------------------------->
            Obs2 = CPTU_interpretation(Depth[x:x + count], trialVal2[x:x + count]) #---------------------------------------->

            # Independent measurements error
            # Mean - lognormal------------------------------------------------->
            muTrial1 = np.exp(np.diag(covTrial1[x:x+count, x:x+count]).reshape(-1,1) + meanTrial1[x:x+count])
            muTrial2 = np.exp(np.diag(covTrial2[x:x+count, x:x+count]).reshape(-1,1) + meanTrial2[x:x+count])

            # Convert to equivalent normal------------------------------------->
            #Eps can be different 
            CovLnEps1=np.log(1.0+eps**2/muTrial1**2)
            CovLnEps2 = np.log(1.0+eps**2/muTrial2**2)

            CovObs1 = np.diag(CovLnEps1.flatten())
            CovObs2 = np.diag(CovLnEps2.flatten())

            #Shear Strength Update
            #Layer1
            MeanCon1=Mean1+np.dot(Cov1,np.dot(np.transpose(Hobs),np.dot(
                    np.linalg.inv(np.dot(Hobs,np.dot(Cov1,np.transpose(Hobs)))
                                                  +CovObs1),(Obs1-np.dot(Hobs,Mean1)))))
            CovCon1=Cov1-np.dot(Cov1,np.dot(np.transpose(Hobs),np.dot(
                    np.linalg.inv(np.dot(Hobs,np.dot(Cov1,np.transpose(Hobs)))+
                                  CovObs1),np.dot(Hobs,Cov1))))

            #Layer2
            MeanCon2=Mean2+np.dot(Cov2,np.dot(np.transpose(Hobs),np.dot(
                    np.linalg.inv(np.dot(Hobs,np.dot(Cov2,np.transpose(Hobs)))
                                                  +CovObs2),(Obs2-np.dot(Hobs,Mean2)))))
            CovCon2=Cov2-np.dot(Cov2,np.dot(np.transpose(Hobs),np.dot(
                    np.linalg.inv(np.dot(Hobs,np.dot(Cov2,np.transpose(Hobs)))+
                                  CovObs2),np.dot(Hobs,Cov2))))


            '''
            Layer Boundary Update-wont require parameters' conversion because of initial Guassian Assumption.Observation 
            is set to expected value of the mean of the boundary. Update to conditional mean is redundant. Layer Observation 
            is a white-noise process with a lower standard deviation than original prior distribution.
            '''
            Mean_layer_trial = Mean_layer[trialLoc]
            Obs_layer = Mean_layer_trial[x:x+count]  
            CovObs_layer = np.eye(count) * eps_layer ** 2

            #Conditional Mean and Covariance for layer
            MeanCon_layer=Mean_layer+np.dot(Cov_layer,np.dot(np.transpose(Hobs),np.dot(
                    np.linalg.inv(np.dot(Hobs,np.dot(Cov_layer,np.transpose(Hobs)))
                                                  +CovObs_layer),(Obs_layer-np.dot(Hobs,Mean_layer)))))

            CovCon_layer=Cov_layer-np.dot(Cov_layer,np.dot(np.transpose(Hobs),np.dot(
                    np.linalg.inv(np.dot(Hobs,np.dot(Cov_layer,np.transpose(Hobs)))+
                                  CovObs_layer),np.dot(Hobs,Cov_layer))))

            Pftemp, Pfall,_=slopelPfLS(Ns,MeanCon1,CovCon1, MeanCon2, CovCon2, MeanCon_layer, CovCon_layer,
                                     epsM,FsLim,N0,Coord,Nsurf,Np,gamma,H)
            Pft[i,j]=Pftemp
    return Pft

def randomFieldUpdate(Val1, Val2, Val_layer, Mean1, Cov1, Mean2, Cov2, Mean_layer, Cov_layer,
                      Coord, loc, eps, eps_layer, Nstart): # Val needs to be log(Val)-->
    # Take the measurement at the selected boreho and update random field
    #Nstart will be first index value of trialLoc(locations after slip circle discretizations), Nstart=Np*Nsurf
    Npoints=np.shape(Coord)[0]

    #cHECK TRIAL ANALYSIS for reference
    unique_xs, first_indices, counts = np.unique(Coord[:, 0][Nstart:], return_counts=True, return_index=True)

    first_indices += Nstart
    count, index = counts[loc], first_indices[loc]

    # Observation vector
    I = np.eye(count)

    H=np.zeros((count,Npoints))
    # Change to 1
    H[:,index:index + count]=I

    # Vector of actual observations
    actObs1 = CPTU_interpretation(Depth[index:index+count], Val1[index:index+count])
    actObs2 = CPTU_interpretation(Depth[index:index+count], Val2[index:index+count])

    # Covariance of actual observations
    # Measurement error
    # Convert to equivalent normal--------------------------------------------->
    CovLnEps1=np.log(1.0+eps**2/np.exp(actObs1)**2)
    CovLnEps2=np.log(1.0+eps**2/np.exp(actObs2)**2)

    # Assign value
    CovActObs1=np.diag(CovLnEps1.flatten())
    CovActObs2=np.diag(CovLnEps2.flatten())

    #Conditional mean and variance for layer1
    MeanCon1=Mean1+np.dot(Cov1,np.dot(np.transpose(H),np.dot(
        np.linalg.inv(np.dot(H,np.dot(Cov1,np.transpose(H)))
                                      +CovActObs1),(actObs1-np.dot(H,Mean1)))))

    CovCon1=Cov1-np.dot(Cov1,np.dot(np.transpose(H),np.dot(
        np.linalg.inv(np.dot(H,np.dot(Cov1,np.transpose(H)))+CovActObs1),
        np.dot(H,Cov1))))

    # Conditional variance and mean for layer2
    MeanCon2=Mean2+np.dot(Cov2,np.dot(np.transpose(H),np.dot(
        np.linalg.inv(np.dot(H,np.dot(Cov2,np.transpose(H)))
                                      +CovActObs2),(actObs2-np.dot(H,Mean2)))))

    CovCon2=Cov2-np.dot(Cov2,np.dot(np.transpose(H),np.dot(
        np.linalg.inv(np.dot(H,np.dot(Cov2,np.transpose(H)))+CovActObs2),
        np.dot(H,Cov2))))

    #Mean and Conditional Variance for the layer
    actObs_layer = Val_layer[index:index+count]
    CovActObs_layer = eps_layer ** 2 * np.eye(len(actObs_layer))

    #np.dot is matrix multiplication for 2D arrays, inner product for 1D arrays
    K = Cov_layer @ H.T @ np.linalg.inv(H @ Cov_layer @ H.T + CovActObs_layer) #Kalman gain
    
    CovCon_layer = Cov_layer -  K @ H @ Cov_layer
    MeanCon_layer = Mean_layer + K @ (actObs_layer - H @ Mean_layer)
    
    return MeanCon1, CovCon1, MeanCon2, CovCon2, MeanCon_layer, CovCon_layer


'''
Prior knowledge(for prior analysis and trial analysis) and actual statistics(assumption that there is one, we take measurements 
from this. original parameters -> lognormal
'''
# Prepare inputs---------------------------------------------------------------

su00, su01 = 36. , 18.
rsu0, rsu1 = 1.2, 1. 
CoVPrior, CoVActual = 0.25, 0.05

#Layer 1
muPrior0 = original_parameters(su00, rsu0, Depth)
sigmaPrior0 = CoVPrior * muPrior0
sigmaActual0 = CoVActual * muPrior0

#Layer 2
muPrior1 = original_parameters(su01, rsu1, Depth)
sigmaPrior1 = CoVPrior * muPrior1
sigmaActual1 = CoVActual * muPrior1

# Equivalent normal parameters - Prior----------------------------------------->
#Layer 1
muLnPrior0, sigmaLnPrior0 = equiv_norm(muPrior0, sigmaPrior0)
muLnActual0, sigmaLnActual0 = equiv_norm(muPrior0, sigmaActual0)

#Layer 2
muLnPrior1, sigmaLnPrior1 = equiv_norm(muPrior1, sigmaPrior1)
muLnActual1, sigmaLnActual1 = equiv_norm(muPrior1, sigmaActual1)
#Implementation
Nstart = Np * Nsurf
Mean1 = muLnPrior0
Mean2 = muLnPrior1
sigma1 = sigmaLnPrior0
sigma2 = sigmaLnPrior1
color='black'

# Create Initial Covariance matrices(Loop not required)
x, z = Coord[:,0], Coord[:,1]
corr = ellipsoid(x.reshape(-1, 1), z.reshape(-1, 1), x.reshape(1, -1), z.reshape(1, -1), theta_x, theta_z)
#Layer-1
C1 = np.diag(sigma1.flatten()) @ corr @ np.diag(sigma1.flatten()) 
C1_Actual =  np.diag(sigmaLnActual0.flatten()) @ corr @ np.diag(sigmaLnActual0.flatten()) 
#Layer-2
C2 = np.diag(sigma1.flatten()) @ corr @ np.diag(sigma1.flatten())
C2_Actual = np.diag(sigmaLnActual1.flatten()) @ corr @ np.diag(sigmaLnActual1.flatten())

#Ndx maybe used if theta_x for higher than theta_x of the Su
x1, y1 =  -10, -2
x2, y2 = 50, 2
theta_layer = 15.0
piecewise_function = np.vectorize(piecewise_function)
Mean_layer = piecewise_function(Coord[:,0], x1, y1, x2, y2).reshape(-1, 1)
Corr_layer = ellipsoid(x.reshape(-1, 1), 0, x.reshape(1, -1), 0, theta_layer, np.inf)
Sigma_layer = 1.13
Cov_layer = Sigma_layer ** 2 * Corr_layer

#Create actual realizations for measurements
Val1 = np.random.multivariate_normal(Mean1.flatten(), C1_Actual).reshape(-1, 1)
Val2 = np.random.multivariate_normal(Mean2.flatten(), C2_Actual).reshape(-1, 1) 
Val_layer = check(Coord[:,0], np.random.multivariate_normal(Mean_layer.flatten(), Cov_layer).reshape(-1, 1))

#Merge the realizations
binary, Val = complete_soil(Coord, Val1, Val2, Val_layer, 1)

# Plot------------------------------------------------------------------------>
plt.scatter(Coord[Np*Nsurf:,0],Coord[Np*Nsurf:,1],c=np.exp(Val[Np*Nsurf:]))
plt.plot(Coord[Np*Nsurf:,0], Val_layer[Np*Nsurf:,0], linewidth='4.0', color=color)
plt.colorbar()
plt.xlabel('$ x $')
plt.ylabel('$ y $')
# plt.savefig(r'C:\Users\ivandep\Desktop\Work\Naturfareforum\SlopeStability\actualval'+str(Ci)+'.png',dpi=600)
plt.show()

# Plot------------------------------------------------------------------------>
plt.scatter(Coord[Np*Nsurf:,0],Coord[Np*Nsurf:,1],c=binary[Np*Nsurf:])
plt.plot(Coord[Np*Nsurf:,0], Val_layer[Np*Nsurf:,0], linewidth='4.0', color=color)
plt.colorbar()
plt.xlabel('$ x $')
plt.ylabel('$ y $')
# plt.savefig(r'C:\Users\ivandep\Desktop\Work\Naturfareforum\SlopeStability\actualval'+str(Ci)+'.png',dpi=600)
plt.show()

# Calculate probability
# Number of samples
Ns=100

# Probability and samples

Pf0, Pfall,prior_circle=slopelPfLS(Ns,Mean1,C1,Mean2, C2, Mean_layer, Cov_layer, epsM,FsLim,N0,Coord,Nsurf,Np,gamma,H)
print('P_F=% .3e' % Pf0)


# Initial cost
C0=Ci+Cf*Pf0

# Number of investigations
Ni=5

# Number of trial location----------------------------------------------------->
Npoints=np.shape(Coord)[0]
Nt=Ndx

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
upMean1, upMean2, upMean_layer = Mean1, Mean2, Mean_layer
# Initialize covariance
upCov1, upCov2, upCov_layer=C1, C2, Cov_layer

#Generate ymax and ymin bounds for each borehole, useful in plotting
ymx= []
ymn = []
cud = Coord[trialLoc] #Just to avoid varibles override
for i in unique_xs:
    y_vals = cud[cud[:,0] == i, 1]
    ymin = np.min(y_vals)
    ymax = np.max(y_vals)
    ymx.append(ymax)
    ymn.append(ymin)
ymx = np.array(ymx)
ymn = np.array(ymn)

#Launch investigations
for i in range(Ni):
    print(i)
    Pft = trialAnalysis(Coord, Nt, Ntsamp, upMean1, upCov1, upMean2, upCov2, upMean_layer, upCov_layer, eps, eps_layer,
                   epsM, FsLim, N0, Nsurf, Np, gamma, H)

    #Calculate Cost
    Cta[i,:]=np.mean(Ci*(i+1)+Cf*Pft,1)

    #Calculate trial VOI
    VoIta[i,:]=1.0*(Pf0-np.mean(Pft,1))
    plt.plot(VoIta[i,:])
    plt.xlabel('Trial investigations')
    plt.ylabel('$ VoI $')
    plt.grid()
    plt.show()

    #Solely for plotting purposes
    VoIta_plot = np.repeat(VoIta[i,:], counts)
    plt.scatter(Coord[trialLoc,0],Coord[trialLoc,1],c=VoIta_plot)
    plt.colorbar()
    plt.xlabel('$ x $')
    plt.ylabel('$ z $')
    plt.show()

    #location with Maximum VoI
    actLoc[i]=np.argmax(VoIta[i,:])

    #Take the measurement at selected locations and calculated updated parameters
    upMean1,upCov1, upMean2, upCov2, upMean_layer, upCov_layer=randomFieldUpdate(Val1,Val2, Val_layer, upMean1,upCov1, upMean2, 
                                                                                 upCov2, upMean_layer, upCov_layer, Coord,
                                                                                 actLoc[i], eps, eps_layer, Nstart)


    # Convert to actual values-------------------------------------------------> Layer1
    # Get diagonal values
    upCovDiag1=np.reshape(np.diagonal(upCov1),np.shape(upMean1))
    upMeanAct1=np.exp(upMean1+0.5*upCovDiag1)
    upCovAct1=upMeanAct1*np.sqrt(np.exp(upCovDiag1)-1.0)
    # Convert to actual values-------------------------------------------------> Layer2
    # Get diagonal values
    upCovDiag2=np.reshape(np.diagonal(upCov2),np.shape(upMean2))
    upMeanAct2=np.exp(upMean2+0.5*upCovDiag2)
    upCovAct2=upMeanAct2*np.sqrt(np.exp(upCovDiag2)-1.0)

    # Plot the actual values and measurement locations------------------------->
    _, c = complete_soil(Coord, Val1, Val2, Val_layer, 1)
    plt.scatter(Coord[Np*Nsurf:,0],Coord[Np*Nsurf:,1],c=np.exp(c[Np * Nsurf:]))
    plt.plot(Coord[Np*Nsurf:,0], Val_layer[Np*Nsurf:], linewidth='4.0', color=color)
    plt.colorbar()
    plt.vlines(x=unique_xs[actLoc[0:i+1]], ymin=ymn[actLoc[0:i+1]], ymax=ymx[actLoc[0:i+1]],
               linewidth=2.0, color='blue')
    plt.xlabel('$ x $')
    plt.ylabel('$ z $')

    plt.show()
    
    updated_layer = check(Coord[:,0], np.random.multivariate_normal(upMean_layer.flatten(), upCov_layer).reshape(-1, 1))

    # Plot the updated mean and measurement locations------------------------->
    _, c = complete_soil(Coord, upMeanAct1, upMeanAct2, updated_layer, 1)
    plt.scatter(Coord[Np*Nsurf:,0],Coord[Np*Nsurf:,1],c=c[Np * Nsurf:])
    plt.plot(Coord[Np*Nsurf:,0], updated_layer[Np*Nsurf:], linewidth='4.0', color=color)
    plt.colorbar()
    plt.vlines(x=unique_xs[actLoc[0:i+1]], ymin=ymn[actLoc[0:i+1]], ymax=ymx[actLoc[0:i+1]],
               linewidth=2.0, color='blue')
    plt.xlabel('$ x $')
    plt.ylabel('$ z $')

    plt.show()

    # Plot the updated variance and measurement locations------------------------->
    _, c = complete_soil(Coord, upCovAct1, upCovAct2, updated_layer, 1)
    plt.scatter(Coord[Np*Nsurf:,0],Coord[Np*Nsurf:,1],c=c[Np * Nsurf:])
    plt.plot(Coord[Np*Nsurf:,0], updated_layer[Np*Nsurf:], linewidth='4.0', color=color)
    plt.colorbar()
    plt.vlines(x=unique_xs[actLoc[0:i+1]], ymin=ymn[actLoc[0:i+1]], ymax=ymx[actLoc[0:i+1]],
               linewidth=2.0, color='blue')
    plt.xlabel('$ x $')
    plt.ylabel('$ z $')

    plt.show()

    # Check updated failure probability
    # Probability and samples
    Pf[i + 1], Pfall, final_circle=slopelPfLS(Ns,upMean1,upCov1, upMean2, upCov2, upMean_layer, upCov_layer,
                              epsM,FsLim,N0,Coord,Nsurf,Np,gamma,H)
    

    # Cost of information
    Ca[i + 1]= Ci[actLoc[i]] + np.sum(Ca)

    # Value of information of the first sample
    VoIa[i + 1] = Cf * (Pf0 - Pf[i + 1])
    print(Pf[i + 1])

end_time = time.time()
total_time = end_time - start_time
print(total_time)

# Final results
plt.plot(VoIa,marker='o')
plt.plot(Ca,'--',marker='x')
plt.ylabel('VoI,cost of investigation')
plt.xlabel('Number of investigations')
plt.grid()
plt.legend(['VoI','Cost of investigation'])
plt.show()

plt.plot(VoIa-Ca,marker='o')
plt.ylabel('VoI-cost of investigation')
plt.xlabel('Number of investigations')
plt.grid()
plt.show()


plt.plot(Pf,marker='o')
plt.ylabel('Failure probability')
plt.xlabel('Number of investigations')
plt.grid()
plt.show()

df_Pf = pd.DataFrame(Pf)
df_VoIa = pd.DataFrame(VoIa)
df_actLoc = pd.DataFrame(actLoc)
df_prior_circles = pd.DataFrame(prior_circle)
df_final_circles = pd.DataFrame(final_circle)

df_Pf.to_csv('Pf.csv', index=False)
df_VoIa.to_csv('VoIa=460.8.csv', index=False)
df_actLoc.to_csv('actLoc.csv', index=False)
df_prior_circles.to_csv('prior_circles.csv', index=False)
df_final_circles.to_csv('final_circles.csv', index=False)

