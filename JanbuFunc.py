# -*- coding: utf-8 -*-
"""
Created on Wed Oct 26 08:22:50 2022

@author: ivandep
"""
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

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
    
    #coef=regN0.coef_
    
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
    
    # plt.plot(xCircle,yCircle)
    # plt.plot([np.min([xStart, -H]),0,H/np.tan(np.radians(beta)),
    #           np.max([H/np.tan(np.radians(beta))+H,xEnd])],[0,0,H,H])
    # #plt.scatter([x1,x2,x3,x4],[y1,y2,y3,y4])
    # #plt.scatter([xStart,xEnd],[yStart,yEnd])
    # plt.show()
    
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
    plt.show()
    
    # Create coordinate vector
    Coord=np.zeros((Ns*Np,2))

    Coord[:,0]=np.reshape(xCoord.T,(Ns*Np))
    Coord[:,1]=np.reshape(yCoord.T,(Ns*Np))
    
    return N0, Coord

# def FsSlopeStability(su,N0,Coord,Ns,Np,gamma,H):
#     Fs=np.zeros(Ns)
    
#     for i in range(Ns):
#         Fs[i]=np.mean(su[i*Np:(i+1)*Np])*N0[i]/(gamma*H)
    
#     return Fs


def discretizeDomain(Coord,Ndx,xMin,xMax,yMin,angle,H):
    # Discretize the domain
    # Number of divisions in x direction
    #Ndx=31
    # Disretize
    x_d=np.linspace(xMin,xMax,num=Ndx)
    # Minimum value in y direction
    #yMin=-30.0
    
    
    for i in range(Ndx):
        # Find y coordinate
        if(x_d[i]<=0):
            yMax=0
            # Number of points
            Nyp=np.int16((yMax-yMin)/2)+1
            # Story coordinates temporary
            CoordTemp=np.zeros((Nyp,2))
            CoordTemp[:,0]=x_d[i]
            CoordTemp[:,1]=np.linspace(yMin,yMax,num=Nyp)
            # Append values
            Coord=np.vstack((Coord,CoordTemp))
            
        elif(x_d[i]>0 and x_d[i]<H/np.tan(np.radians(angle))):
            yMax=np.tan(np.radians(angle))*x_d[i]
            # Number of points
            Nyp=np.int16((yMax-yMin)/2)+1
            # Story coordinates temporary
            CoordTemp=np.zeros((Nyp,2))
            CoordTemp[:,0]=x_d[i]
            CoordTemp[:,1]=np.linspace(yMin,yMax,num=Nyp)
            # Append values
            Coord=np.vstack((Coord,CoordTemp))
        else:
            yMax=H
            # Number of points
            Nyp=np.int16((yMax-yMin)/2)+1
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
    
# # Prepare inputs---------------------------------------------------------------
# # Number of surfaces
# Ns=12
# # Slope angle
# angle=26.5
# # Range of depths
# d=np.linspace(0,1,num=Ns)
# # Number of points along surface
# Np=30
# # Slope height
# H=20.0
# # Number of discretization points in x direction
# Ndx=31
# # Boundaries
# xMin=-40; xMax=80; yMin=-30

# # Fit functions to Janbu diagrams
# regN0, polyN0, reg_x, poly_x, reg_y, poly_y=Janbufuncs()
# # Get stability coefficients and coordinates of failure surfaces
# N0,Coord=failureSurfaces(regN0, polyN0, reg_x, poly_x, reg_y, poly_y, angle,Ns,Np,H,d) 
# # Discretize the domain
# Coord,Depth=discretizeDomain(Coord,Ndx,xMin,xMax,yMin,angle,H)