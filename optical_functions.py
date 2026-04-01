#  Sme preliminary packages 

import numpy as np
import scipy
import scipy.special
from scipy.fft import fft2, fftfreq, ifft2, fftshift, ifftshift
import matplotlib.pyplot as plt

import math
import os

from PIL import Image


# CONSTANTS 

nm = 1e-9 
um = 1e-6
mm = 1e-3
cm = 1e-2

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
phase - boolean - controls whether or not we wanna plot the phase
'''

def TotInt(Ex, phase=True, cmappy='hot', enable_colourbar='True'):

    if (phase==False): # Just plot the intensity
        fig, ax = plt.subplots(1,1, figsize=(8,4))
        #ax.set_title('Intensity')
        intensity = ax.imshow(abs(Ex)**2, cmap=cmappy)
        ax.axis('off')
        if (enable_colourbar):
            cbar = fig.colorbar(intensity, ax=ax)
    
    else:

        fig, ax =  plt.subplots(1,2,figsize=(8,4))
        ax[0].set_title('Intensity')
        intensity = ax[0].imshow(abs(Ex)**2,cmap=cmappy)
        ax[0].axis('off')
        cbar=fig.colorbar(intensity, ax=ax[0])
        
        ax[1].set_title('Phase')
        phase = ax[1].imshow(np.angle(Ex), cmap="hsv", interpolation='nearest')
        ax[1].axis('off')
        if (enable_colourbar):
            cbar = fig.colorbar(phase, ax=ax[1])
        #cbar=fig.colorbar(phase, ax=ax[1])
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
    
    fields = np.empty((num_of_spots, N, N), dtype=np.complex128)
    # Space definition 
    for ii in range(num_of_spots):
        X=np.linspace(-maxx,maxx,N) + spot_loc_x[ii]
        Y=np.linspace(-maxx,maxx,N) + spot_loc_y[ii]
        h=np.abs(X[1]-X[2]) # Step size
        xx,yy=np.meshgrid(X,Y)
        r, phi= cart2pol(xx,yy)
        
        fields[ii] = pupil_function(r, rad_spot)
    
    return fields  # In principle, it suffices to return fields. 


# Function that outputs channels at more predefined, symmetric points

def output_chan_symmetric(X, Y, rad_spot, maxx, num_of_spots, chan_sep=1.0):
    N = len(X)
    spot_loc_x = []
    spot_loc_y = []
    
    for ii in range(int(num_of_spots/2)):
        
            # Add a 'positive' and 'negative' spot
            spot_loc_x.append((ii+1)*chan_sep*mm)
            spot_loc_y.append(0)
            
            spot_loc_x.append(-(ii+1)*chan_sep*mm)
            spot_loc_y.append(0)
    
    fields = np.empty((num_of_spots, N, N))
    # Space definition 
    for ii in range(num_of_spots):
        X=np.linspace(-maxx,maxx,N) + spot_loc_x[ii]
        Y=np.linspace(-maxx,maxx,N) + spot_loc_y[ii]
        h=np.abs(X[1]-X[2]) # Step size
        xx,yy=np.meshgrid(X,Y)
        r, phi= cart2pol(xx,yy)
        
        fields[ii] = pupil_function(r, rad_spot)
    
    return fields # In principle, it suffices to return fields. 

# This creates a specific triangle-like configuration for the symmetric sorting of three modes

def output_chan_triangle(X, Y, rad_spot, maxx, chan_sep=1.0):
    
    # The y-offset is set so that we form an equilbrium triangle 
    
    y_offset = (np.sqrt(3)*(chan_sep*mm))/2
    
    N = len(X)
    spot_loc_x = []
    spot_loc_y = []

    # First two symmetric spots
    spot_loc_x.append(chan_sep * mm)
    spot_loc_y.append(0)

    spot_loc_x.append(-chan_sep * mm)
    spot_loc_y.append(0)

    # Third spot centered horizontally, shifted down vertically
    spot_loc_x.append(0)
    spot_loc_y.append(-y_offset)

    num_of_spots = 3  # Explicitly define since we're overriding symmetry
    fields = np.empty((num_of_spots, N, N))

    # Generate spots
    for ii in range(num_of_spots):
        X_shifted = np.linspace(-maxx, maxx, N) + spot_loc_x[ii]
        Y_shifted = np.linspace(-maxx, maxx, N) + spot_loc_y[ii]
        h = np.abs(X_shifted[1] - X_shifted[2])
        xx, yy = np.meshgrid(X_shifted, Y_shifted)
        r, phi = cart2pol(xx, yy)

        fields[ii] = pupil_function(r, rad_spot)

    return fields


# This creates channel spots arranged evenly on a circle

def output_chan_circle(X, Y, rad_spot, maxx, num_of_spots, circle_radius=1.0):
    
    """
    Place `num_of_spots` pupil apertures evenly spaced on a circle of radius
    `circle_radius` (in mm) centered at the origin.

    Parameters
    ----------
    X, Y : ndarray
        Base coordinate vectors (only the length is used to infer grid size).
    rad_spot : float
        Radius of each pupil aperture.
    maxx : float
        Half-width of the numerical window used to build the meshgrid.
    num_of_spots : int
        Number of channels to place on the circle.
    circle_radius : float, optional
        Circle radius in millimeters. Default is 1.0 mm.

    Returns
    -------
    fields : ndarray
        Array of shape (num_of_spots, N, N) containing the pupil masks.
    """

    N = len(X)

    # Compute centers on a circle (convert radius to meters via mm constant)
    angles = np.linspace(0, 2 * np.pi, num_of_spots, endpoint=False)
    spot_loc_x = circle_radius * mm * np.cos(angles)
    spot_loc_y = circle_radius * mm * np.sin(angles)

    fields = np.empty((num_of_spots, N, N), dtype=np.complex128)

    for ii in range(num_of_spots):
        X_shifted = np.linspace(-maxx, maxx, N) + spot_loc_x[ii]
        Y_shifted = np.linspace(-maxx, maxx, N) + spot_loc_y[ii]
        h = np.abs(X_shifted[1] - X_shifted[2])
        xx, yy = np.meshgrid(X_shifted, Y_shifted)
        r, phi = cart2pol(xx, yy)

        fields[ii] = pupil_function(r, rad_spot)

    return fields


# This creates channel spots arranged evenly on a circle

def output_chan_circle(X, Y, rad_spot, maxx, num_of_spots, circle_radius=1.0):
    """
    Place `num_of_spots` pupil apertures evenly spaced on a circle of radius
    `circle_radius` (in mm) centered at the origin.

    Parameters
    ----------
    X, Y : ndarray
        Base coordinate vectors (only the length is used to infer grid size).
    rad_spot : float
        Radius of each pupil aperture.
    maxx : float
        Half-width of the numerical window used to build the meshgrid.
    num_of_spots : int
        Number of channels to place on the circle.
    circle_radius : float, optional
        Circle radius in millimeters. Default is 1.0 mm.

    Returns
    -------
    fields : ndarray
        Array of shape (num_of_spots, N, N) containing the pupil masks.
    """

    N = len(X)

    # Compute centers on a circle (convert radius to meters via mm constant)
    angles = np.linspace(0, 2 * np.pi, num_of_spots, endpoint=False)
    spot_loc_x = circle_radius * mm * np.cos(angles)
    spot_loc_y = circle_radius * mm * np.sin(angles)

    fields = np.empty((num_of_spots, N, N), dtype=np.complex128)

    for ii in range(num_of_spots):
        X_shifted = np.linspace(-maxx, maxx, N) + spot_loc_x[ii]
        Y_shifted = np.linspace(-maxx, maxx, N) + spot_loc_y[ii]
        h = np.abs(X_shifted[1] - X_shifted[2])
        xx, yy = np.meshgrid(X_shifted, Y_shifted)
        r, phi = cart2pol(xx, yy)

        fields[ii] = pupil_function(r, rad_spot)

    return fields


# This creates a diagonal-like configuration pattern

def output_chan_triangle(X, Y, rad_spot, maxx, chan_sep=1.0):
    
    # The y-offset is set so that we form an equilbrium triangle 
    
    y_offset = (np.sqrt(3)*(chan_sep*mm))/2
    
    N = len(X)
    spot_loc_x = []
    spot_loc_y = []

    # First two symmetric spots
    spot_loc_x.append(chan_sep * mm)
    spot_loc_y.append(0)

    spot_loc_x.append(-chan_sep * mm)
    spot_loc_y.append(0)

    # Third spot centered horizontally, shifted down vertically
    spot_loc_x.append(0)
    spot_loc_y.append(-y_offset)

    num_of_spots = 3  # Explicitly define since we're overriding symmetry
    fields = np.empty((num_of_spots, N, N))

    # Generate spots
    for ii in range(num_of_spots):
        X_shifted = np.linspace(-maxx, maxx, N) + spot_loc_x[ii]
        Y_shifted = np.linspace(-maxx, maxx, N) + spot_loc_y[ii]
        h = np.abs(X_shifted[1] - X_shifted[2])
        xx, yy = np.meshgrid(X_shifted, Y_shifted)
        r, phi = cart2pol(xx, yy)

        fields[ii] = pupil_function(r, rad_spot)

    return fields

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
    i = 1j
    
    if (knotType == 'Trefoil'): # Input beam profile (Trefoil)
        AK=np.exp(-(rs/(np.sqrt(2)*kk))**2)*(1 - rs**2 - 4 * (a**2 - b**2) * rs**3 - rs**4 + rs**6 - 2 *(a - b)**2 * (rs*np.exp(-1j*phi))**3 - 2 *(a + b)**2 * (rs*np.exp(1j*phi))**3)
    
    if (knotType == 'Hopflink'): # Input beam profile (Hopf Link)
        AK = (1 - 2*(1+a**2 - b**2)*rs**2 + rs**4 - 2*(a**2 + b**2)*np.cos(2*phi)*rs**2 - 1j*4*a*b*np.sin(2*phi)*rs**2)*np.exp(-(rs/(np.sqrt(2)*kk))**2)

    if (knotType == 'Cinquefoil'): # Input beam profile (Cinquefoil)
        AK = np.exp(-(rs/(np.sqrt(2)*kk))**2) * (1 + rs**2 - 2*rs**4 - 16*(a**2 - b**2)*rs**5 - 2*rs**6 + rs**8 + rs**10 - (8*((a-b)**2)*(rs**5)*np.exp(-1j*5*(phi))) - (8*((a+b)**2)*(rs**5)*np.exp(1j*5*(phi))))
        
    if (knotType == 'Figure-8'): # Input beam profile (Figure-8)
    
        AK = result = (
        (8 * a**3 * rs**6 * np.exp(-2 * i * phi)) +
        (8 * a**3 * rs**6 * np.exp(2 * i * phi)) +
        (16 * a**3 * rs**4 * np.exp(-2 * i * phi)) +
        (16 * a**3 * rs**4 * np.exp(2 * i * phi)) +
        (8 * a**3 * rs**2 * np.exp(-2 * i * phi)) +
        (8 * a**3 * rs**2 * np.exp(2 * i * phi)) +
        (12 * a**2 * rs**8) +
        (24 * a**2 * rs**6) -
        (24 * a**2 * rs**2) -
        (12 * a**2) +
        (6 * a * b**2 * rs**6 * np.exp(-2 * i * phi)) +
        (6 * a * b**2 * rs**6 * np.exp(2 * i * phi)) +
        (12 * a * b**2 * rs**4 * np.exp(-2 * i * phi)) +
        (12 * a * b**2 * rs**4 * np.exp(2 * i * phi)) +
        (24 * a * b * rs**2 * np.exp(-2 * i * phi)) -
        (24 * a * b * rs**2 * np.exp(2 * i * phi)) -
        (4 * b**3 * rs**4 * np.exp(-4 * i * phi)) +
        (4 * b**3 * rs**4 * np.exp(4 * i * phi)) -
        (3 * b**2 * rs**8) -
        (6 * b**2 * rs**6) +
        (6 * b**2 * rs**2) +
        (3 * b**2) -
        (16 * rs**8) +
        (32 * rs**6) -
        (32 * rs**2) +
        16)*np.exp(-(rs/(np.sqrt(2)*kk))**2) 
    
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


# This is a simple routine that applies normalization to the field, given also the numerical step size. 

def norm_field(field,h):
    norm_fac=np.sqrt(np.sum(np.abs(field*h)**2))
    return field/norm_fac 


# This function computes the shannon entropy in d-dimensions 

def shannon_entropy(x,d):
    return (-x*np.log2(x/(d-1)) - (1-x)*np.log2(1-x))

# Blazed diffraction grating that we used to simulate creating a knotted beam using an SLM

def Hologram(A,hx,hy,LA): 
  # A -> Complex amplitude of the beam 
  # hx, hy -> x,y step-size
  # LA -> grating periodicity. This is usually expressed in terms of wavelength units
  # Normalization of the input beam
    
    nn=np.sum(np.abs(A)**2)*hx*hy
    NU=A/np.sqrt(nn)
      # Amplitude and phase pattern 
    Amp=np.abs(NU)
    PHI=np.angle(NU)
      # Grating
    mm=Amp.shape
    x1,y1=np.meshgrid(hx*np.arange(1,mm[1]+1),hy*np.arange(1,mm[0]+1))
      # Inverse Sinc fucntion
    ss=np.linspace(-np.pi,0,2000)
    sincc=np.sin(ss)/ss
    sincc[np.isnan(sincc)]=1
      # Amplitude masking 
    M=1+np.interp(Amp,sincc,ss)/np.pi
    M[np.isnan(M)]=0
      # Phase Hologram
    F=np.mod(PHI-np.pi*M+(2*np.pi*(x1+y1))/LA,2*np.pi)
      # Full Hologram
    return M*F


# Just a simple function that wraps a field from [0,2pi] to [-pi, +pi]

def wrap_to_domain(field):
    N = len(field)
    for ii in range(N):
        for jj in range(N):
            if(field[ii,jj] > np.pi):
                field[ii,jj] -= 2*np.pi
                
    return field








