 
# coding: utf-8

# # Heat Diffusion in Soils
# 
# This Jupyter Notebook gives an example how to implement a 1D heat diffusion model in Python.
# 
# First we need to import the packages which we will be using:
# 

# In[1]:


import numpy as np
import scipy.integrate as spi
import MyTicToc as mt
import matplotlib.pyplot as plt
from collections import namedtuple
import HeatDiffusionPython as hdp

# plot figures inline
#%matplotlib inline 

# plot figures as interactive graphs...
#%matplotlib qt

### Definition of functions
# Then we need to define the functions which we will be using:
# BndTTop for calculating the rain as a function of time;
# watertFlux for calculating all water flows in the domain;
# DivwaterFlux for calculating the divergence of the waterflow across the cells in the domain.


#make the temperature viscosity table
def TempVis(T):
    Tini = [0, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    mu = [1.787, 1.519, 1.307, 1.002, 0.798, 0.653, 0.547, 0.467, 0.404, 0.355, 0.315, 0.282]
    p = np.polyfit(Tini, mu, 11)
    p = np.poly1d(p)
    mu1 = p(T)
    return mu1

#make Ksat          #maybe make array from this?
def Ksat(T, sPar):
    Ksat = sPar.kappa / TempVis(T) * sPar.rhoW * 9.81
    return Ksat


#making S effective
def SEFF(hw, sPar):
    hc=-hw #ha is zero
    Seff= (1+((hc*(hc>0))*sPar.alpha)**sPar.n)**-sPar.m
    return Seff

#making theta 
def Theta(hw, sPar):
    Seff=SEFF(hw, sPar)
    theta_w=Seff*(sPar.theta_sat-sPar.theta_res)+sPar.theta_res
    return theta_w

#making krw (relative permeability) for every hw
def krwfun(hw, sPar, mDim):
    nIN = mDim.nIN
    Seff=SEFF(hw, sPar,)
    nr,nc = hw.shape
    
    krw=Seff**3
    i = np.arange(1, nIN-1)   
    Krw=np.zeros((nIN,nc))
    Krw[0]= krw[0]
    Krw[i] = np.minimum(krw[i-1], krw[i])  
    Krw[-1] = krw[-1]
    return Krw

# boundary top condition
def BndwTop(t, bPar):    #t> 25 and 200 -0.001
    bndT= -0.001*(t > bPar.tMin) *(t < bPar.tMax) 
    return bndT


# part of the richardson equation Ksat*krw(gradient*hw+flux*z))
def waterFlux(t, T, hw, sPar, mDim, bPar):
    nIN = mDim.nIN
    dzN = mDim.dzN
    ksat= Ksat(T, sPar)
    krw=krwfun(hw, sPar, mDim)
    nr,nc = hw.shape
    q = np.zeros((nIN,nc))

    # Flux in all intermediate nodes
    i= np.arange(1, nIN - 1)
    q[i] = -ksat[i] * krw[i] * ((hw[i] - hw[i-1])/dzN[i-1] + 1)

    # Lower boundary:
    if bPar.botCon == "ROBIN":
        q[0] = -bPar.krob*(hw[0] - bPar.h0)
    if bPar.botCon == "GRAVITY":
        q[0] = -ksat[0]*krw[0]
    
    #top boundary conditions 
    #bndw = BndwTop(t, bPar) 
    q[nIN-1] = 0  # or bndw

    return q


#def Cfun(sPar, hw):
#    hc=-hw
#    Seff=SEFF(sPar,hw)
#    dSeffdh = sPar.alpha*sPar.m/ (1-sPar.m)*Seff**(1/sPar.m)*\(1-Seff**(1/sPar.m))**sPar.m*(hc>0)+(hc<=0) * 0
#    return (sPar.theta_sat-sPar.theta_res)*dSeffdh


#Part of the richardson equation (C(hw)+Sw*Ss**w)
def Cfuncmplx(hw, sPar):
    
    dh=np.sqrt(np.finfo(float).eps)
    if np.iscomplexobj(hw):
       hcmplx = hw.real+1j*dh
    else:
        hcmplx = hw.real +1j*dh
    theta =Theta(hcmplx, sPar)
    C =theta.imag/dh
    return C

def Caccentfun(hw, sPar, mDim):
    theta=Theta(hw, sPar)
    Sw=theta/sPar.theta_sat
    Chw=Cfuncmplx(hw, sPar)
    beta=4.5e-10
    Sws=sPar.rhoW*9.81*(sPar.Cv+sPar.theta_sat*beta)
    C=Chw+Sw*Sws
    
    C[mDim.nN-1] = 1/mDim.dzIN[mDim.nIN-2] * (hw[mDim.nN-1]>0) + C[mDim.nN-1] * (hw[mDim.nN-1]<=0)
    return C


def dhwdtFun(t, hw, sPar, mDim, bPar):
    nr,nc = hw.shape
    nN = mDim.nN
    dzIN = mDim.dzIN

    divqW = np.zeros([nN,nc]).astype(hw.dtype)

    C = Caccentfun(hw, sPar, mDim)
    qW = waterFlux(t, hw, sPar, mDim, bPar)
    
    # Calculate divergence of flux for all nodes
    i = np.arange(0,nN)
    divqW[i] = -(qW[i + 1] - qW[i]) / (dzIN[i] * C[i])
    return divqW
    
def main():
    # Then we start running our model.
    # First we require the domain discretization

    # Domain
    nIN = 101
    # soil profile until 1 meters depth
    zIN = np.linspace(-1, 0, num=nIN).reshape(nIN, 1)
    # nIN = np.shape(zIN)[0]
    zN = np.zeros(nIN - 1).reshape(nIN - 1, 1)
    zN[0,0] = zIN[0,0]
    zN[1:nIN - 2,0] = (zIN[1:nIN - 2,0] + zIN[2:nIN - 1,0]) / 2
    zN[nIN - 2,0] = zIN[nIN - 1]
    nN = np.shape(zN)[0]

    i = np.arange(0, nN - 1)
    dzN = (zN[i + 1,0] - zN[i,0]).reshape(nN - 1,1)
    dzIN = (zIN[1:,0] - zIN[0:-1,0]).reshape(nIN - 1,1)

    # collect model dimensions in a namedtuple: modDim
    modDim = namedtuple('ModDim', ['zN', 'zIN', 'dzN', 'dzIN', 'nN', 'nIN'])
    mDim = modDim(zN=zN,
                  zIN=zIN,
                  dzN=dzN,
                  dzIN=dzIN,
                  nN=nN,
                  nIN=nIN)

    # ## Definition of material properties
    # In this section of the code we define the material properties

    # Soil Properties SAND from table 2.4 Mayer & Hassanizadeh ch 2

    rhoW = 1000  # [kg/m3] density of water
    theta_res=0.045
    theta_sat=0.38
    n= 3
    m= 1-(1/3)
    alpha=2 #m^-1
    Cv=1e-8
    kappa = 0.05 #m/day

    # collect soil parameters in a namedtuple: soilPar
    soilPar = namedtuple('soilPar', ['rhoW','n', 'm','alpha', 'Cv', 'kappa','theta_res', 'theta_sat'])
    sPar = soilPar(rhoW=np.ones(np.shape(zN))*rhoW, n=np.ones(np.shape(zN))*n, m=np.ones(np.shape(zN))*m, alpha=np.ones(np.shape(zN))*alpha, Cv=np.ones(np.shape(zN))*Cv, kappa=np.ones(np.shape(zN))*kappa, theta_res=np.ones(np.shape(zN))*theta_res, 
                   theta_sat =np.ones(np.shape(zN))*theta_sat)
                   


    # ## Definition of the Boundary Parameters
    # boundary parameters
    # collect boundary parameters in a named tuple boundpar...
    boundPar = namedtuple('boundPar', ['tMin','tMax', 'botCon', 'h0','krob'])
    bPar = boundPar(tMin=25,
                    tMax=200,
                    botCon='ROBIN',
                    h0 = -1,
                    krob=0.01)


    # ## Initial Conditions
    # Initial Conditions
    HIni= -0.75-zN
    # Time Discretization
    tOut = np.linspace(0, 600, 120)  # time
    nOut = np.shape(tOut)[0]

    # ## Implement using the built-in ode solver

    mt.tic()

    def intFun(t, T, hw):
        if len(hw.shape) == 1:
            hw = hw.reshaspe(nN,1)
        nf = dhwdtFun(t, hw, sPar, mDim, bPar)
        nv = hdp.DivHeatflux(t, T, sPar, mDim, bPar) 
        dhdwT = np.concatenate(nf,nv)
        return dhdwT

    def jacFun(t, y):
        if len(y.shape)==1:
            y = y.reshape(mDim.nN,1)
        
        nr, nc = y.shape
        dh = np.sqrt(np.finfo(float).eps)
        ycmplx = np.repeat(y,nr,axis=1).astype(complex)
        c_ex = np.eye(nr)*1j*dh
        ycmplx = ycmplx + c_ex
        dfdy = intFun(t,T,ycmplx).imag/dh      #make T complex
        return spi.coo_matrix(dfdy)

    
   # H0 = HIni.squeeze()
    #print(H0)
    HODE = spi.solve_ivp(intFun, [tOut[0], tOut[-1]], HIni.squeeze(), method='BDF',
                         t_eval=tOut, vectorized=True, rtol=1e-6)
    mt.toc()

    plt.close('all')
    fig1, ax1 = plt.subplots(figsize=(7, 4))
    for ii in np.arange(0, nN, 5):
        ax1.plot(HODE.t, HODE.y[ii,:], '-', label=ii)
    ax1.legend()
    ax1.set_title('Water pressure head, impermeable top and  ROBIN condition (ODE)')
    ax1.grid(b=True)
    ax1.set_xlabel('time [days]')
    ax1.set_ylabel('Water pressure head [m]')

    fig2, ax2 = plt.subplots(figsize=(4, 7))

    for ii in np.arange(0, HODE.t.size, 1):
        ax2.plot(HODE.y[:, ii], zN[:, 0], '-')
    ax2.set_title('Water pressure head vs. depth, impermeable top and  ROBIN condition (ODE)')
    #ax2.legend()
    ax2.grid(b=True)
    ax2.set_xlabel('Water pressure head [m]')
    ax2.set_ylabel('Depth [m]')
    
    thODE = Theta(HODE.y, sPar)
    
    fig3, ax3 = plt.subplots(figsize=(7, 7))
    for ii in np.arange(0, HODE.t.size, 1):
        ax3.plot(thODE[:, ii], zN[:, 0], '-', label=ii)
    ax3.grid(b=True)
    #ax3.legend()
    ax3.set_title('Water content vs. depth, impermeable top and ROBIN condition (ODE)')
    ax3.set_xlabel('water content [-]')
    ax3.set_ylabel('depth [m]')
    plt.show()
    # plt.savefig('myfig.png')

if __name__ == "__main__":
    main()
