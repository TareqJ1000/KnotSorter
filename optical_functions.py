#  Sme preliminary packages 

import numpy as np
import scipy
import scipy.special
from scipy.fft import fft2, fftfreq, ifft2, fftshift, ifftshift
import matplotlib.pyplot as plt
from IPython.display import display
import threading
from io import StringIO 
import sys
import ipywidgets as widgets
import math
import os


import plotly
import plotly.graph_objs as go

import diffractsim
diffractsim.set_backend("CPU")
from diffractsim import nm, mm, cm, um

from PIL import Image

# Optical Functions


'''
This propagates the beam using a Fresnal Diffraction Transfer Function approach. 

PARAMETERS
u1 - source plane
L - length of the numerical window (in units of w0)
la - lambda (m-1)
z - propagation distance (w0)
'''

def propTF(u1,L,la,z):
    M,nn=u1.shape
    dx=L/M
    fx=np.arange(-1/(2*dx),1/(2*dx),1/L)
    Fx, Fy = np.meshgrid(fx, fx)
    H=np.exp(-1j*np.pi*la*z*(Fx**2+Fy**2))
    
    H = fftshift(H)
    U2=H*fft2(fftshift(u1))
    u2=ifftshift(ifft2(U2))

    
    return u2 


'''
This function implements the Fraufoner Diffraction Transfer Approach instead. Translation from Voelz
u1 - source field
L1 - full observation side length 
la - wavelength 
z - prop distance


returns: 
u2 - observation field at z
L2 - observation field side lengths
'''

def propFF(u1, L1, la, z, isInverse = False):
    #Some initial calcs from the source field
    
    M, nn = np.shape(u1)
    dx1 = L1/M
    k = 2*np.pi/la
    
    # compute params for observation plane
    
    L2 = (la*z)/dx1
    dx2 = (la*z)/L1
    
    # compute field at observation plane 
    x2 = np.arange(-L2/2,L2/2,dx2)
    
    xx2, yy2 = np.meshgrid(x2, x2)
    
    # Fraufofner transfer function? 

    c = (1/(1j*la*z))*np.exp(1j*(k/2*z)*(xx2**2 + yy2**2))
    
    if(isInverse):
        u2 = c*fftshift(ifft2(ifftshift(u1)))*dx1**2
        
    else:
        u2 = c*ifftshift(fft2(fftshift(u1)))*dx1**2
    
    return u2, L2



# LG modes 

'''
# Generates the normalized LG field.
# Expression adapted from Saleh. The normalization factor is adapted from wikipedia, 
# then modified silghtly so that the LG mode expression in wikipedia and Saleh are equivalent
#
# RHO, PHI - polar coordinate field
# ell - azimuthal index
# p - radial index
# w0 - beam waist (in units of ... w0)
# h - grid step size
# z - propagation distance
# k - wavenumber
'''

def LG(RHO,PHI,ell,p,w0,h,z,k):
    wL = (2*np.pi)/(k)
    z_o = np.pi*w0**2/wL
    w_z = lambda z: w0*np.sqrt(1 + (z/z_o)**2)
    R_z = lambda z: z*(1+(z_o/z)**2)
    zeta_z = lambda z: np.arctan(z/z_o)

    N_factor = 1
    if (z == 0):
        AK = N_factor * np.exp(-(RHO/w0)**2) * ((RHO/w0)**abs(ell) * 
                                                    scipy.special.eval_genlaguerre(p,abs(ell),2*(RHO/w0)**2) * np.exp(1j * ell * PHI))
    else:
        AK = N_factor * (w0/w_z(z)) * np.exp(-(RHO/w_z(z))**2) * ((RHO/w_z(z))**abs(ell) * 
                                                    scipy.special.eval_genlaguerre(p,abs(ell),2*(RHO/w_z(z))**2) * np.exp(1j * ell * PHI))*np.exp(-1j*k*z) * np.exp(-1j*k*(RHO**2/(2*R_z(z)))) * np.exp(1j*(abs(ell)+2*p + 1)*zeta_z(z))
    reNormFactor = np.sqrt(np.sum(np.conj(AK)*AK * h**2)) # We use this to further normalize the field with respect to 
    return  AK/reNormFactor


# Function which plots the phase and intensity of the field

'''
Jointly plots the intensity and phase patterns of the complex field.

Ex - complex 2D array -  field
'''

def TotInt(Ex):
    fig, ax =  plt.subplots(1,2,figsize=(10,10))
    ax[0].set_title('Intensity')
    intensity = ax[0].imshow(abs(Ex)**2,cmap="gist_gray")
    ax[0].axis('off')
    cbar=fig.colorbar(intensity, ax=ax[0])
    
    ax[1].set_title('Phase')
    phase = ax[1].imshow(np.angle(Ex), cmap="hsv", interpolation='nearest')
    ax[1].axis('off')
    cbar=fig.colorbar(phase, ax=ax[1])
    plt.show()
    
    
'''
Cartesian to Polar coordinates 
x,y - x and y coordinates
'''

def cart2pol(x, y):
    rho = np.sqrt((x)**2 + (y)**2)
    phi = np.arctan2(y, x)
    return(rho, phi)



def lens_phase(rr,lens_rad, k, f): 
    t = 1
    pupil_func = (np.where(rr**2 < lens_rad**2,t, np.zeros_like(rr)))
    trans_func = pupil_func*np.exp(-1j*(2*k/f)*(rr)**2)
    
    return pupil_func*np.exp(-1j*(2*k/f)*(rr)**2)


# Pupil function which we convolve with the outgoing field 

def pupil_function(rr, lens_rad):
    t = 1
    return (np.where(rr**2 < lens_rad**2,t, np.zeros_like(rr)))
    
# Class holding the OAM beams

class oamModes():
    def __init__(self, oamBeam, output_chan_field):
        self.oamBeam = oamBeam 
        self.output_chan_field = output_chan_field


def output_chan(X, Y, rad_spot, maxx, num_of_spots):
    N = len(X)
    spot_loc_x = []
    spot_loc_y = []
    
    for ii in range(num_of_spots):
        spot_loc_x.append(np.random.uniform(-maxx+rad_spot,maxx-rad_spot))
        spot_loc_y.append(np.random.uniform(-maxx+rad_spot,maxx-rad_spot))
    
    fields = np.empty((num_of_spots, N, N), dtype=np.complex64)
    # Space definition 
    for ii in range(num_of_spots):
        X=np.linspace(-maxx,maxx,N) + spot_loc_x[ii]
        Y=np.linspace(-maxx,maxx,N) + spot_loc_y[ii]
        h=np.abs(X[1]-X[2]) # Step size
        xx,yy=np.meshgrid(X,Y)
        r, phi= cart2pol(xx,yy)
        
        fields[ii] = pupil_function(r, rad_spot)
    
    return fields  # In principle, it suffices to return fields. 

'''
Generates knots
# This function intialises the knot field that we want to generate.
# rr, phi - field coordinates
# w0 - input beam waist (in mm)
# knotType - string - selects the polynomial characteristic of the knot
# shapeParams - float list - list of knot parameters (a,b,kk) expected
'''

def setKnotType(rr, phi, w0,  knotType, shapeParams): 
    
    rs = rr/w0 # dimensionless, scaled beam coordinate
    a,b,kk = shapeParams
    
    if (knotType == 'Trefoil'): # Input beam profile (Trefoil)
        AK=np.exp(-(rs/(np.sqrt(2)*kk))**2)*(1 - rs**2 - 4 * (a**2 - b**2) * rs**3 - rs**4 + rs**6 - 2 *(a - b)**2 * (rs*np.exp(-1j*phi))**3 - 2 *(a + b)**2 * (rs*np.exp(1j*phi))**3)
    
    if (knotType == 'Hopflink'): # Input beam profile (Hopf Link)
        AK = (1 - 2*(1+a**2 - b**2)*rs**2 + rs**4 - 2*(a**2 + b**2)*np.cos(2*phi)*rs**2 - 1j*4*a*b*np.sin(2*phi)*rs**2)*np.exp(-(rs/(np.sqrt(2)*kk))**2)

    if (knotType == 'Cinquefoil'): # Input beam profile (Cinquefoil)
        AK = np.exp(-(rs/(np.sqrt(2)*kk))**2) * (1 + rs**2 - 2*rs**4 - 16*(a**2 - b**2)*rs**5 - 2*rs**6 + rs**8 + rs**10 - (8*((a-b)**2)*(rs**5)*np.exp(-1j*5*(phi))) - (8*((a+b)**2)*(rs**5)*np.exp(1j*5*(phi))))
   
    return AK


# This function generates phase gratings in reminisce of OAM gratings 


def OAMWithGratings(l,rows,cols,xoffset,yoffset,a):
    
    crow, ccol = int(rows / 2)+xoffset, int(cols / 2)+yoffset
    mask = np.zeros((rows,cols)) +0j
    fx = 10.0
    fy = 0.0
    phi = np.zeros((rows,cols))+0j

    for i in range (rows):
        for j in range (cols):
            
            x = i - crow
            y = j - ccol
            alpha = l*np.arctan2(x,y) + 2*np.pi*10*y/500
            g = a[0] + a[1]*np.cos(2*alpha)  + a[2]*np.cos(3*alpha) + a[3]*np.cos(4*alpha)
                  #g = 1/4*(1 + np.cos(alpha) + np.cos(2*alpha) +  np.cos(3*alpha)+np.cos(4*alpha))
                  #g = np.mod(alpha,2*np.pi)
            mask[i,j] = g
            

    return(mask)






