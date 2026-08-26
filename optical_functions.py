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
    """Propagate a square, uniformly sampled field with Fresnel diffraction.

    This is the transfer-function implementation from Voelz, Eqs. (5.1)-(5.2).
    ``L`` is the full side length of both the source and observation grids. The
    longitudinal carrier phase ``exp(1j*k*z)`` is intentionally omitted because
    it has no effect on the transverse intensity or on subsequent phase-only
    modulation.
    """
    if z == 0:
        return np.array(u1, dtype=np.complex128, copy=True)

    M,nn=u1.shape
    if M != nn:
        raise ValueError("propTF requires a square field array.")
    if L <= 0 or la <= 0:
        raise ValueError("L and wavelength must be positive.")

    dx=L/M
    # Writing the frequency grid in terms of integer samples avoids occasional
    # off-by-one lengths caused by floating-point np.arange endpoints.
    fx=(np.arange(M) - M//2)/L
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


def propagate_legacy_fft(field, phase_maps):
    """Propagate through one to three legacy phase planes.

    The historical one-plane sorter applies a centered forward FFT after its
    phase plate. The two-plane sorter then applies its second phase plate in
    that Fourier plane and returns with an inverse FFT. Extending the same
    convention gives the alternating sequence forward, inverse, forward for
    three phase planes.

    No normalization is introduced here so the one- and two-plane results
    retain the exact scaling of the original implementation.
    """
    field = np.asarray(field)
    phase_maps = np.asarray(phase_maps)
    if field.ndim != 2:
        raise ValueError("field must be a two-dimensional array.")
    if phase_maps.ndim != 3:
        raise ValueError("phase_maps must have shape (planes, rows, columns).")
    if not 1 <= len(phase_maps) <= 3:
        raise ValueError("The legacy sorter supports one, two, or three planes.")
    if phase_maps.shape[1:] != field.shape:
        raise ValueError("Every legacy phase map must match the input field shape.")

    propagated = np.asarray(field, dtype=np.complex128)
    for plane_index, phase_map in enumerate(phase_maps):
        modulated = propagated*phase_map
        if plane_index % 2 == 0:
            propagated = fftshift(fft2(modulated))
        else:
            propagated = ifft2(ifftshift(modulated))
    return propagated

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



def lens_phase(rr, lens_rad, k, f):
    """Return the complex transmittance of an ideal paraxial thin lens.

    The phase is ``exp(-1j*k*r**2/(2*f))`` (Voelz Eq. 6.12). ``lens_rad``
    may be ``None`` to omit the finite circular pupil.
    """
    if f == 0:
        raise ValueError("The focal length must be non-zero.")

    if lens_rad is None:
        pupil_func = np.ones_like(rr, dtype=float)
    else:
        if lens_rad <= 0:
            raise ValueError("The lens aperture radius must be positive.")
        pupil_func = np.where(rr <= lens_rad, 1.0, 0.0)

    return pupil_func*np.exp(-1j*k*rr**2/(2*f))


def padded_grid_size(grid_size, padding_factor=1.0):
    """Return a centered padded size while preserving the source-grid parity."""
    if int(grid_size) != grid_size or grid_size <= 0:
        raise ValueError("grid_size must be a positive integer.")
    if not np.isfinite(padding_factor) or padding_factor < 1:
        raise ValueError("padding_factor must be at least 1.")

    grid_size = int(grid_size)
    padded_size = max(grid_size, int(np.ceil(grid_size*padding_factor)))
    # Equal padding on both sides keeps the optical axis on the same sample.
    if (padded_size-grid_size) % 2:
        padded_size += 1
    return padded_size


def center_pad(array, output_shape, fill_value=0):
    """Embed a 2-D array in the center of a larger array."""
    array = np.asarray(array)
    if array.ndim != 2:
        raise ValueError("center_pad requires a two-dimensional array.")
    if len(output_shape) != 2:
        raise ValueError("output_shape must contain two dimensions.")
    if any(outer < inner for outer, inner in zip(output_shape, array.shape)):
        raise ValueError("The padded shape cannot be smaller than the input.")
    if any((outer-inner) % 2 for outer, inner in zip(output_shape, array.shape)):
        raise ValueError("Centered padding requires equal padding on both sides.")

    output = np.full(output_shape, fill_value, dtype=np.result_type(
        array.dtype, np.asarray(fill_value).dtype
    ))
    starts = [(outer-inner)//2 for outer, inner in zip(output_shape, array.shape)]
    output[
        starts[0]:starts[0]+array.shape[0],
        starts[1]:starts[1]+array.shape[1],
    ] = array
    return output


def center_crop(array, output_shape):
    """Return the centered 2-D crop with ``output_shape``."""
    array = np.asarray(array)
    if array.ndim != 2:
        raise ValueError("center_crop requires a two-dimensional array.")
    if len(output_shape) != 2:
        raise ValueError("output_shape must contain two dimensions.")
    if any(outer < inner for outer, inner in zip(array.shape, output_shape)):
        raise ValueError("The crop shape cannot exceed the input shape.")
    if any((outer-inner) % 2 for outer, inner in zip(array.shape, output_shape)):
        raise ValueError("Centered cropping requires equal cropping on both sides.")

    starts = [(outer-inner)//2 for outer, inner in zip(array.shape, output_shape)]
    return array[
        starts[0]:starts[0]+output_shape[0],
        starts[1]:starts[1]+output_shape[1],
    ]


def fresnel_sampling_diagnostics(grid_side_length, grid_size, wavelength,
                                 propagation_distance=None,
                                 focal_length=None,
                                 lens_radius=None,
                                 padding_factor=1.0):
    """Return dimensionless Fresnel and thin-lens sampling diagnostics.

    For the transfer-function propagator, ``tf_ratio <= 1`` satisfies Voelz's
    oversampling condition ``dx >= wavelength*z/L``. For a finite thin lens,
    ``lens_ratio >= 1`` satisfies ``f_number >= dx/wavelength``.
    """
    if grid_side_length <= 0 or grid_size <= 0 or wavelength <= 0:
        raise ValueError("Grid length, grid size, and wavelength must be positive.")

    dx = grid_side_length/grid_size
    computational_size = padded_grid_size(grid_size, padding_factor)
    computational_length = dx*computational_size
    diagnostics = {
        "dx": dx,
        "computational_grid_size": computational_size,
        "computational_side_length": computational_length,
        "effective_padding_factor": computational_size/grid_size,
    }

    if propagation_distance is not None:
        diagnostics["tf_ratio"] = (
            wavelength*abs(propagation_distance)/(computational_length*dx)
        )
        diagnostics["critical_distance"] = computational_length*dx/wavelength

    if focal_length is not None and lens_radius is not None:
        if lens_radius <= 0:
            raise ValueError("The lens aperture radius must be positive.")
        f_number = abs(focal_length)/(2*lens_radius)
        diagnostics["f_number"] = f_number
        diagnostics["minimum_sampled_f_number"] = dx/wavelength
        diagnostics["lens_ratio"] = f_number/(dx/wavelength)

    return diagnostics


def build_fresnel_lens_kernels(field_shape, grid_side_length, wavelength,
                               stages, rr, lens_radius=None,
                               padding_factor=1.0):
    """Precompute propagation and lens kernels for a phase/lens train.

    Each stage is a mapping with three distances in metres::

        phase plane -> z_to_lens -> thin lens(f) -> z_after_lens

    The final stage ends at the detector plane. Earlier stages end at the next
    phase plane. ``padding_factor`` expands the computational window without
    changing the pixel pitch. The returned kernels can be reused for every
    input mode when evaluating a single candidate sorter geometry.
    """
    rows, cols = field_shape
    if rows != cols:
        raise ValueError("Fresnel lens trains require square field arrays.")
    if rr.shape != field_shape:
        raise ValueError("rr must have the same shape as the propagated field.")

    computational_size = padded_grid_size(rows, padding_factor)
    dx = grid_side_length/rows
    computational_length = dx*computational_size
    fx = (
        np.arange(computational_size)-computational_size//2
    )/computational_length
    Fx, Fy = np.meshgrid(fx, fx)
    if computational_size == rows:
        computational_rr = rr
    else:
        coordinates = dx*(
            np.arange(computational_size)-computational_size//2
        )
        computational_xx, computational_yy = np.meshgrid(
            coordinates, coordinates
        )
        computational_rr = np.sqrt(
            computational_xx**2+computational_yy**2
        )
    k = 2*np.pi/wavelength
    kernels = []

    for index, stage in enumerate(stages):
        missing = {"z_to_lens", "focal_length", "z_after_lens"} - set(stage)
        if missing:
            raise ValueError(
                f"Optical stage {index} is missing: {', '.join(sorted(missing))}."
            )

        z_to_lens = float(stage["z_to_lens"])
        focal_length = float(stage["focal_length"])
        z_after_lens = float(stage["z_after_lens"])
        if z_to_lens < 0 or z_after_lens < 0:
            raise ValueError("Free-space propagation distances cannot be negative.")

        h_before = np.exp(
            -1j*np.pi*wavelength*z_to_lens*(Fx**2 + Fy**2)
        )
        h_after = np.exp(
            -1j*np.pi*wavelength*z_after_lens*(Fx**2 + Fy**2)
        )
        kernels.append({
            "before": fftshift(h_before),
            "lens": lens_phase(
                computational_rr, lens_radius, k, focal_length
            ),
            "after": fftshift(h_after),
        })

    return kernels


def _propagate_with_kernel(field, transfer_function):
    """Apply a precomputed, Voelz-style Fresnel transfer function."""
    spectrum = fft2(fftshift(field))
    return ifftshift(ifft2(transfer_function*spectrum))


def propagate_fresnel_lens_train(field, phase_maps, grid_side_length,
                                 wavelength, stages, rr,
                                 lens_radius=None, kernels=None,
                                 return_intermediate=False,
                                 padding_factor=1.0,
                                 return_padded=False):

    """Propagate through one, two, or three phase/lens stages.

    ``phase_maps`` must contain complex unit-modulus transmittances. A stage is
    applied after each phase plane, including the final propagation to the
    detector. More than three planes are rejected deliberately so configuration
    mistakes do not silently create a different architecture. With padding,
    the incident field is zero padded and the phase plates are unity padded;
    the field remains expanded through the full train and is cropped only at
    the detector unless ``return_padded`` is true.
    """
    phase_maps = np.asarray(phase_maps)
    if phase_maps.ndim != 3:
        raise ValueError("phase_maps must have shape (planes, rows, columns).")
    if not 1 <= len(phase_maps) <= 3:
        raise ValueError("The sorter supports one, two, or three phase planes.")
    if len(stages) != len(phase_maps):
        raise ValueError("There must be exactly one optical stage per phase plane.")
    if phase_maps.shape[1:] != np.shape(field):
        raise ValueError("Every phase map must match the input field shape.")

    base_shape = np.shape(field)
    if base_shape[0] != base_shape[1]:
        raise ValueError("Fresnel lens trains require square field arrays.")
    computational_size = padded_grid_size(base_shape[0], padding_factor)
    computational_shape = (computational_size, computational_size)

    if kernels is None:
        kernels = build_fresnel_lens_kernels(
            base_shape, grid_side_length, wavelength, stages, rr,
            lens_radius=lens_radius, padding_factor=padding_factor,
        )
    if len(kernels) != len(phase_maps):
        raise ValueError("The number of kernel sets must match the phase planes.")
    if any(kernel["lens"].shape != computational_shape for kernel in kernels):
        raise ValueError(
            "The supplied kernels do not match the requested padding factor."
        )

    propagated = center_pad(
        np.asarray(field, dtype=np.complex128),
        computational_shape,
        fill_value=0.0,
    )
    # Unity padding makes the extra phase-plate area optically neutral. The
    # incident field is zero padded, then allowed to diffract throughout the
    # larger window instead of wrapping around its original FFT boundary.
    padded_phase_maps = [
        center_pad(phase_map, computational_shape, fill_value=1.0+0.0j)
        for phase_map in phase_maps
    ]
    intermediate = []
    for phase_map, stage_kernels in zip(padded_phase_maps, kernels):
        propagated = propagated*phase_map
        after_phase = propagated
        propagated = _propagate_with_kernel(propagated, stage_kernels["before"])
        before_lens = propagated
        propagated = propagated*stage_kernels["lens"]
        after_lens = propagated
        propagated = _propagate_with_kernel(propagated, stage_kernels["after"])
        if return_intermediate:
            intermediate.append({
                "after_phase": after_phase,
                "before_lens": before_lens,
                "after_lens": after_lens,
                "stage_output": propagated,
            })

    output = propagated if return_padded else center_crop(propagated, base_shape)
    if return_intermediate:
        return output, intermediate
    return output


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

def output_chan_circle(X, Y, rad_spot, maxx, num_of_spots, circle_radius=1.0,
                       coordinate_mode="legacy"):
    
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

    if coordinate_mode not in {"legacy", "physical"}:
        raise ValueError("coordinate_mode must be 'legacy' or 'physical'.")

    if coordinate_mode == "physical":
        # The output plane of the Fresnel TF propagator has the same coordinate
        # sampling as the input plane. Build the pupils directly on that grid.
        xx, yy = np.meshgrid(np.asarray(X), np.asarray(Y))
        for ii, (center_x, center_y) in enumerate(zip(spot_loc_x, spot_loc_y)):
            shifted_radius = np.sqrt(
                (xx-center_x)**2 + (yy-center_y)**2
            )
            fields[ii] = pupil_function(shifted_radius, rad_spot)
        return fields

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
    """d-dimensional Shannon entropy with well-defined endpoint limits."""
    x = float(np.clip(x, 0.0, 1.0))
    if d < 2:
        raise ValueError("The alphabet dimension d must be at least 2.")

    error_term = 0.0 if x == 0 else -x*np.log2(x/(d-1))
    correct_probability = 1-x
    correct_term = (
        0.0 if correct_probability == 0
        else -correct_probability*np.log2(correct_probability)
    )
    return error_term+correct_term


def balanced_detector_throughput(efficiency_matrix, method="geometric_mean"):
    """Combine accepted detector efficiencies across the input alphabet.

    Each row of ``efficiency_matrix`` corresponds to one input and each column
    to an accepted detector window. The row sum is therefore the fraction of
    that input's incident power accepted by the detector bank. The geometric
    mean is the default because it rewards throughput while penalizing a
    solution that sacrifices any one input mode.

    The accepted fractions are clipped to ``[0, 1]`` to suppress floating-point
    excursions beyond the physical power-fraction range. The unclipped raw
    matrix remains available to callers for diagnostics.
    """
    matrix = np.asarray(efficiency_matrix, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] == 0:
        raise ValueError("efficiency_matrix must be a non-empty 2D array.")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("efficiency_matrix must contain only finite values.")
    if np.any(matrix < -1e-12):
        raise ValueError("Detector efficiencies cannot be negative.")

    accepted = np.clip(matrix.sum(axis=1), 0.0, 1.0)
    if method == "geometric_mean":
        throughput = (
            0.0 if np.any(accepted == 0.0)
            else np.exp(np.mean(np.log(accepted)))
        )
    elif method == "minimum":
        throughput = np.min(accepted)
    elif method == "arithmetic_mean":
        throughput = np.mean(accepted)
    else:
        raise ValueError(
            "throughput_metric must be 'geometric_mean', 'minimum', "
            "or 'arithmetic_mean'."
        )
    return float(throughput), accepted


def complex_field_fidelity(field_a, field_b):
    """Return the normalized, global-phase-invariant complex-field overlap."""
    field_a = np.asarray(field_a, dtype=np.complex128)
    field_b = np.asarray(field_b, dtype=np.complex128)
    if field_a.shape != field_b.shape:
        raise ValueError("The two fields must have the same shape.")
    if not np.all(np.isfinite(field_a)) or not np.all(np.isfinite(field_b)):
        raise ValueError("Fields must contain only finite values.")

    power_a = np.vdot(field_a, field_a).real
    power_b = np.vdot(field_b, field_b).real
    if power_a <= 0 or power_b <= 0:
        raise ValueError("Field fidelity is undefined for a zero-power field.")
    fidelity = abs(np.vdot(field_a, field_b))**2/(power_a*power_b)
    return float(np.clip(fidelity, 0.0, 1.0))


def intensity_fidelity(field_a, field_b):
    """Return the squared Bhattacharyya overlap of two intensity patterns."""
    field_a = np.asarray(field_a)
    field_b = np.asarray(field_b)
    if field_a.shape != field_b.shape:
        raise ValueError("The two fields must have the same shape.")
    if not np.all(np.isfinite(field_a)) or not np.all(np.isfinite(field_b)):
        raise ValueError("Fields must contain only finite values.")

    intensity_a = np.abs(field_a)**2
    intensity_b = np.abs(field_b)**2
    power_a = intensity_a.sum()
    power_b = intensity_b.sum()
    if power_a <= 0 or power_b <= 0:
        raise ValueError("Intensity fidelity is undefined for a zero-power field.")
    fidelity = np.sum(np.sqrt(intensity_a*intensity_b))**2/(power_a*power_b)
    return float(np.clip(fidelity, 0.0, 1.0))

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








