import numpy as np 
from scipy.fft import fft2, ifft2, fftshift, ifftshift
import matplotlib.pyplot as plt

import plotly
import plotly.graph_objs as go


# CONSTANTS 
um = 1e-6 
mm = 1e-3
nm = 1e-9

# Fresnel Propagator Function

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
    H=np.exp(-1j*np.pi*0.25*la*z*(Fx**2+Fy**2))
    
    H = fftshift(H)
    U2=H*fft2(fftshift(u1))
    u2=ifftshift(ifft2(U2))
    
    
    return u2 


'''
An alternate version of the Fresnel Propagator code developed by Christian Howard

PARAMETERS
X, Y - meshgrid over which the field is defined
u0 - field being propagated
wl - wavelength 
z - longitudinal distance of propagation

'''

def propTF_C(X, Y, u0, z, wl=500*mm):
    
    k = 2*np.pi/wl # wave number magnitude 

    def h(x, y, z):
          return np.exp(1j*k*z)/(1j*wl*z) * np.exp(1j*k*(x**2+y**2)/(2*z))
    
    h_data = h(X, Y, z)
    u0_fft = fft2(u0)
    h_data_fft = fft2(h_data)
    u0_prop_fft = u0_fft * h_data_fft
    u0_prop = ifftshift(ifft2(u0_prop_fft))

    return u0_prop

# After constructing the knot at z = 0, we use the Fresnel Propagator to evolve the knot
# Fresnel propagation using the Transfer function approach
# Based on Computational Fourier Optics by Voelz 
#
# PARAMETERS
# 
# Z0 - Z-propagation distance
# NZ - number of measurements
# AK - waist plane field to propagate
# maxx - Half length of the numerical window (w0)
# la (lambda_0) - Wavelength of the simulated beam
# forwardProp -- determines if we are performing a forward propagation with the beam or not
# compress_len -- centered n x n region retained at every plane. None keeps
#                 the complete propagation grid.
# system_C -- switch over to Christian's implementation of the Fresnel Propagator
# X, Y -- Cartesian meshgrid

def PropKnots(Z0,NZ, AK, maxx, la, forwardProp, compress_len=None, system_C = False, X=[], Y=[]):
   # Propagation space
    Z=np.linspace(0,Z0,NZ)
    # Propagation steps
    dz=np.abs(Z[0]-Z[1])
    print(dz)
    # Initialization the propagation 
    U=AK
    if U.ndim != 2 or U.shape[0] != U.shape[1]:
        raise ValueError("PropKnots expects a square 2D input field.")
    sim_len = len(U)
    if compress_len is None:
        compress_len = sim_len
    if int(compress_len) != compress_len:
        raise ValueError("compress_len must be an integer or None.")
    compress_len = int(compress_len)
    if not 1 <= compress_len <= sim_len:
        raise ValueError("compress_len must lie between 1 and the field size.")
    if (sim_len-compress_len) % 2:
        raise ValueError(
            "The field size and compress_len must have matching parity for a "
            "centered crop."
        )
    crop_start = (sim_len-compress_len)//2
    crop_stop = crop_start+compress_len

    def center_crop(field):
        return field[crop_start:crop_stop, crop_start:crop_stop]
    #print(f"Compression Index:{compress_index}")
    # Check that the oversampiling criterion value is less than dx, 
    # if it is greater, then we may run into artefacts
    M,nn=U.shape
    dx=(2*maxx)/M

    # Also, decompress the X and Y fields to match the compression grid
    #XX = X[compress_index:-compress_index, compress_index:-compress_index]
    #YY = Y[compress_index:-compress_index, compress_index:-compress_index]
    #print(f"dx:{dx}")
    #print(f"Oversampling criterion:{(la*dz)/(2*maxx)}") 
    
    # Forward Propagation 
    
    if (forwardProp):
        # Saving the field at every plane. Use compress_len to reduce memory usage
        F = np.zeros((NZ+1,compress_len, compress_len),dtype=complex)
        F[0] = center_crop(AK)
        print(" INITIATING FORWARD PROPAGATION ")
        for ii in range(0,NZ):
            print(ii)
            if (system_C):
                 U = propTF_C(X, Y, U, dz, wl=la) # Propagated field over a distance dz
            else:
                U=propTF(U,2*maxx,la,dz) # Field at plane z->+z_0

            F[ii+1] = center_crop(U)
            #F[ii+1] = U
        return F
    
    else: # Backwards propagation
        print(" INITIATING BACKWARD PROPAGATION ")
        U=AK
        FB = np.zeros((NZ,compress_len,compress_len),dtype=complex)
        for ii in range(0,NZ):
            print(ii)
            if (system_C):
                 U = propTF_C(X, Y, U, -dz, wl=la)
            else:       
                U=propTF(U,2*maxx,la,-dz) # Field at plane z->-z_0

            FB[ii] = center_crop(U)

        return FB


'''
Computes the singularity points

F (SQUARE, complex array): The complete field whose singularity points we need to compute. NORMALIZE THIS, or the threshold variable won't work!!!!
A (boolean): indicates whether we are propagating forward or backwards
'''

def Singular(F, A, min_contour_length=10):
    """
    Computes the singularity points from zero contours.
    
    Parameters:
    -----------
    F : array
        Complex field array (should be normalized)
    A : bool
        True for forward propagation, False for backward
    min_contour_length : int, optional
        Minimum number of points in a contour to be considered valid.
        Contours shorter than this are ignored to reduce noise. Default is 10.
    """
    
    X=[]
    Y=[]
    Z=[]
    leng = len(F[0]) # Pay attention to the size of your input array here 
    ll=np.arange(leng) 
    xf,yf=np.meshgrid(ll,ll)
    #JJ=np.sqrt((xf-(leng/2))**2+(yf-(leng/2))**2)>thresh

    for ii in range(0,len(F)):
        print(ii) 

        # Some of the contours become jittery enough that we don't end up collecting all of the intersection points. 
        # We apply a smoothing filter onto the contours to reduce the jittery-ness

        realF = np.real(F[ii])
        imF = np.imag(F[ii])

        contour1 = plt.contour(realF,0,colors='b');
        contour2 = plt.contour(imF,0,colors='r');

        xi = np.array([])
        yi = np.array([])
        # Use get_paths() directly on the QuadContourSet object
        for path in contour1.get_paths():
            # Filter out small contours
            if len(path.vertices) < min_contour_length:
                continue
                
            for path2 in contour2.get_paths():
                # Filter out small contours
                if len(path2.vertices) < min_contour_length:
                    continue
                    
                xinter, yinter = find_intersections(path.vertices, path2.vertices)
                if (len(xinter) == len(yinter)):
                    xi = np.append(xi, xinter)
                    yi = np.append(yi, yinter)
                else:
                    print(f"CRAPPY INTERSECTION DETECTED")

        X=np.concatenate((X,xi))
        Y=np.concatenate((Y,yi))
        if A==True:
            Z=np.concatenate((Z,ii*(xi*0+1)))
        else:
            Z=np.concatenate((Z,-ii*(xi*0+1)))
    return [X,Y,Z]

def Singular2(F, A, region_size=None):
    """Locate phase singularities, optionally inside a centered subregion.

    Parameters
    ----------
    F : array_like
        Propagated complex fields with shape ``(planes, rows, columns)``.
    A : bool
        If true, plane indices are positive; otherwise they are negative.
    region_size : int or (int, int), optional
        Height and width of the centered region searched in each plane. An
        integer selects a square region. ``None`` searches the complete field.

    Returns
    -------
    list of numpy.ndarray
        The x, y, and z coordinates of the intersections. Even when a crop is
        used, x and y remain coordinates on the original full field.
    """
    fields = np.asarray(F)
    if fields.ndim != 3:
        raise ValueError(
            "F must have shape (number_of_planes, rows, columns)."
        )

    _, field_rows, field_cols = fields.shape
    if region_size is None:
        region_rows, region_cols = field_rows, field_cols
    elif np.isscalar(region_size):
        if int(region_size) != region_size:
            raise ValueError("region_size must contain integer dimensions.")
        region_rows = region_cols = int(region_size)
    else:
        if len(region_size) != 2:
            raise ValueError("region_size must be an integer or (rows, columns).")
        region_rows, region_cols = region_size
        if int(region_rows) != region_rows or int(region_cols) != region_cols:
            raise ValueError("region_size must contain integer dimensions.")
        region_rows, region_cols = int(region_rows), int(region_cols)

    if not (1 <= region_rows <= field_rows and 1 <= region_cols <= field_cols):
        raise ValueError(
            "region_size must be positive and no larger than the propagated field."
        )

    row_start = (field_rows-region_rows)//2
    col_start = (field_cols-region_cols)//2
    row_stop = row_start+region_rows
    col_stop = col_start+region_cols

    X = []
    Y = []
    Z = []

    def contour_paths(contour):
        if hasattr(contour, "get_paths"):
            return contour.get_paths()
        return [
            path
            for collection in contour.collections
            for path in collection.get_paths()
        ]

    for ii, full_field in enumerate(fields):
        print(ii)
        cropped_field = full_field[row_start:row_stop, col_start:col_stop]
        real_part = np.real(cropped_field)
        imaginary_part = np.imag(cropped_field)
        # Matplotlib otherwise draws a contour at the nearest data value when
        # zero lies outside the cropped range, which can create false centers.
        if not (
            real_part.min() <= 0 <= real_part.max()
            and imaginary_part.min() <= 0 <= imaginary_part.max()
        ):
            continue
        figure, axis = plt.subplots()
        xi = np.array([])
        yi = np.array([])
        try:
            real_contour = axis.contour(real_part, 0, colors="b")
            imaginary_contour = axis.contour(
                imaginary_part, 0, colors="r"
            )
            for real_path in contour_paths(real_contour):
                for imaginary_path in contour_paths(imaginary_contour):
                    xinter, yinter = find_intersections(
                        real_path.vertices, imaginary_path.vertices
                    )
                    if len(xinter) != len(yinter):
                        raise RuntimeError(
                            "Mismatched singularity-intersection coordinates."
                        )
                    # Contour coordinates are local to the crop. Translate them
                    # back to full-field pixel coordinates for downstream use.
                    xi = np.append(xi, xinter+col_start)
                    yi = np.append(yi, yinter+row_start)
        finally:
            plt.close(figure)

        X = np.concatenate((X, xi))
        Y = np.concatenate((Y, yi))
        direction = 1 if A else -1
        Z = np.concatenate((Z, direction*ii*np.ones_like(xi)))

    return [X, Y, Z]


def find_intersections(A, B):
    #this function adapted from https://stackoverflow.com/questions/3252194/numpy-and-line-intersections#answer-9110966
    # with improved handling of edge cases
    
    # Handle empty input
    if len(A) < 2 or len(B) < 2:
        return np.array([]), np.array([])
    
    # min, max and all for arrays
    amin = lambda x1, x2: np.where(x1<x2, x1, x2)
    amax = lambda x1, x2: np.where(x1>x2, x1, x2)
    aall = lambda abools: np.dstack(abools).all(axis=2)
    
    def slope(line):
        d = np.diff(line, axis=0)
        # Add small epsilon to avoid division by zero
        d_x = d[:,0]
        d_x = np.where(np.abs(d_x) < 1e-10, 1e-10, d_x)
        return d[:,1] / d_x

    x11, x21 = np.meshgrid(A[:-1, 0], B[:-1, 0])
    x12, x22 = np.meshgrid(A[1:, 0], B[1:, 0])
    y11, y21 = np.meshgrid(A[:-1, 1], B[:-1, 1])
    y12, y22 = np.meshgrid(A[1:, 1], B[1:, 1])

    m1, m2 = np.meshgrid(slope(A), slope(B))
    
    # Add small epsilon to avoid division by zero and handle parallel lines
    m1_safe = np.where(np.abs(m1) < 1e-10, 1e-10, m1)
    m2_safe = np.where(np.abs(m2) < 1e-10, 1e-10, m2)
    
    m1inv, m2inv = 1/m1_safe, 1/m2_safe
    
    # Avoid division by zero in denominator (parallel lines)
    denom = 1 - m1*m2inv
    denom = np.where(np.abs(denom) < 1e-10, 1e-10, denom)
    
    yi = (m1*(x21-x11-m2inv*y21) + y11) / denom
    xi = (yi - y21)*m2inv + x21

    xconds = (amin(x11, x12) < xi, xi <= amax(x11, x12), 
              amin(x21, x22) < xi, xi <= amax(x21, x22) )
    yconds = (amin(y11, y12) < yi, yi <= amax(y11, y12),
              amin(y21, y22) < yi, yi <= amax(y21, y22) )

    # Filter out invalid intersections (NaN or Inf)
    valid_mask = aall(xconds) & np.isfinite(xi) & np.isfinite(yi)
    
    return xi[valid_mask], yi[valid_mask]

'''
Makes a nice plot of the singularities.
Ord - singularity points 
zScale - scales the z-coordinate by a factor for plotting purposes
'''

def KnotPlot(Ord, zScale):
    # Configure the trace.
    trace = go.Scatter3d(
        x=Ord[0],  # <-- Put your data instead
        y=Ord[1],  # <-- Put your data instead
        z=Ord[2]*(zScale),  # <-- Put your data instead
        mode='markers',
        marker={
            'size': 3,
            'opacity': 0.8,})
    # Configure the layout.
    layout = go.Layout(
        margin={'l': 0, 'r': 0, 'b': 0, 't': 0},
        scene=dict(
            xaxis=dict(showticklabels=False, title=''),
            yaxis=dict(showticklabels=False, title=''),
            zaxis=dict(showticklabels=False, title='')
        ))
    data = [trace]
    plot_figure = go.Figure(data=data, layout=layout)
    # Render the plot.
    plotly.offline.iplot(plot_figure)
    # Return figure
    return plot_figure
