# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 14:15:29 2024

@author: tjaou104
"""

import numpy as np 
import scipy as sp
from scipy import ndimage
import pygad
import yaml 
import argparse
from yaml import Loader 
import pickle as pkl

from optical_functions import LG, propFF, propTF, cart2pol, oamModes, output_chan, output_chan_symmetric, output_chan_triangle, output_chan_circle, setKnotType, norm_field, shannon_entropy
from scipy.fft import ifft2, ifftshift, fft2, fftshift

import matplotlib.pyplot as plt 

#from diffractsim import cm, mm, um 
import os

# physical constants

cm = 1e-2
mm = 1e-3
um = 1e-6 
nm = 1e-9

# Remove if used outside of the cluster 

parser=argparse.ArgumentParser(description='test')
parser.add_argument('--ii', dest='ii', type=int,
    default=None, help='')
args = parser.parse_args()
shift = args.ii

# *** OMIT IN CLUSTER 

# shift = 5

# *** OMIT IN CLUSTER

# This function keeps track of the generation number + best fitness

# Load configuration file

stream = open(f"configs/ga{shift}.yaml", 'r')
cnfg = yaml.load(stream, Loader=Loader)

# Backward compatibility: supply defaults for new config keys
cnfg.setdefault('circle_radius', 1.5)  # mm, used for circular output channel layouts
cnfg.setdefault('fitness_func', 'secret_key' ) 
cnfg.setdefault('alpha', 1.0) # This controls whether we choose the worst performing channel as a bottleneck for our function or no
                              # at alpha=1.0, we recover our old fitness function behavior

''' 
Global/Optimization Parameters 
'''

N = cnfg['dim']
num_of_output_chans = cnfg['num_output_chans']
output_chan_width = cnfg['output_chan_width'] * mm # in mm 
channel_sep = cnfg['channel_sep'] 
circle_radius = cnfg['circle_radius']

# Some parameters specifying the LG modes

LG_modes = cnfg['LG_modes']
w0 = cnfg['w0'] * mm # in mm!!

isKnot = cnfg['isKnot']
knotType = cnfg['knotType']
shapeParams = cnfg['shapeParams']

simulateLens = cnfg['simulateLens'] # Do we simulate the phase effects of our lenses, or do we neglect them and take the fourier and inverse fourier transform? 
fourier_lens = cnfg['fourier_length']*cm # fourier length of both lens in cm

multiPhaseLens = cnfg['multiPhaseLens'] # Enables multiple phase screens in the near field and the far field. 
multiPhase = cnfg['multiPhase'] # Enables a simplified experiment where we propagate the knot through multiple phase modulations. Does not involve a lens

# Settings specific for the Multi-Phase Experiment

num_phase_maps_near = cnfg['num_phase_maps_near']  # Number of phase maps in the near field
num_phase_maps_far = cnfg['num_phase_maps_far'] # Number of phase maps in the far field
z_o = cnfg['z_o']*cm # Distance between successive phase maps (if applicable). We use Fresnel propagation

# Determine total number of phase maps 

num_of_phase_maps = num_phase_maps_near + num_phase_maps_far

# Settings specific for rotating the incident field

rot_angle = eval(cnfg['rot_angle'])
fixedRotation = cnfg['fixedRotation']
randomRotation = cnfg['randomRotation']

GFilterStrength = cnfg['gauss_filter_sigma'] # sigma parameter for the gaussian filter .. apply to initial population and in computing the fitness param. 

'''
GA Parameters
'''

num_generations = int(eval(cnfg['num_of_gens']))
gen_start = int(eval(cnfg['gen_start'])) # Number of generations for the initial population

num_parents_mating = cnfg['parents_mating']

sol_per_pop = cnfg['sol_per_pop'] # number of parents in the population?? 
num_genes = num_of_phase_maps*N**2 # This would refer to the number of parameters in our DNA

# Lower and upper-bound ranges of the parameterization. 

init_range_low = -np.pi
init_range_high = np.pi

parent_c = cnfg['parent_c'] # the scaling parameter for exponential decay
parent_k = cnfg['parent_k'] # controls the peak of the probability distribution

crossover_type = "single_point"
crossover_probability = cnfg['crossover_prob'] # We keep the solution untouched to the next gen if RNG is <= this number

mutation_type = cnfg['mutation_type']
mutation_probability = eval(cnfg['mutation_prob']) # probability of mutation (if the mutation type is adaptive, must be a tuple in (high probability, low probability))
# mutation_percent_genes = cnfg['mutation_percent'] # Percentage of genes to mutate. This parameter actually does nothing

random_mutation_min_val = -cnfg['random_mutation_min_val']*np.pi 
random_mutation_max_val = cnfg['random_mutation_max_val']*np.pi 

gen_saturate = cnfg['gen_saturate']

keep_elitism = cnfg['keep_elitism']

fitness_function = cnfg['fitness_func']
alpha = cnfg['alpha']

last_pop = 0

'''
Define the initial field 
'''

# Define the coordinate space 

la = 0.78*um
k=(2*np.pi)/la  # [m^-1] wavenumber    
N=128 # [Number of points per dimension]
maxx = 20*um*N # Full length of the numerical window in terms of the usual length of SLM macropixels (m)

# Space definition 
dx = maxx/N
dy = maxx/N 

#okay let's just say h here is dx or dy for now WLOG (WITH ... loss of generality)
h = dx

X = dx*(np.arange(N) - N //2)
Y = dy*(np.arange(N) - N //2)

xx,yy=np.meshgrid(X,Y)
r, phi= cart2pol(xx,yy)

''' 
Create the OAM beams that we need to sort 
'''
# Now create a list containing 'oamMode' objects 

list_of_OAMs = []

output_chans = output_chan_circle(X, Y, output_chan_width, maxx, num_of_output_chans, circle_radius=circle_radius)


if(isKnot):
    for ii in range(len(knotType)):
        list_of_OAMs.append(oamModes(setKnotType(r, phi, w0, knotType[ii], shapeParams[ii]), output_chans[ii]))
        
else:
    for ii in range(len(LG_modes)):
        list_of_OAMs.append(oamModes(LG(r, phi, LG_modes[ii][0], LG_modes[ii][1], w0,h,0,k), output_chans[ii]))


# NEW! We define a function which computes the rotated knotted field.
# We assume that we have pre-defined the parameters making up the knotted field. 
# Bad practice, I know. 

def create_rotated_knots(rot_phi):
    
    # Apply rotation operator on coords. Update: The rotation should be 
    # X_rot = np.cos(rot_phi)*X - np.sin(rot_phi)*Y
    # Y_rot = np.sin(rot_phi)*X + np.cos(rot_phi)*Y

    xx,yy=np.meshgrid(X ,Y);
    
    xx_rot = np.cos(rot_phi)*xx - np.sin(rot_phi)*yy
    yy_rot = np.sin(rot_phi)*xx + np.cos(rot_phi)*yy

    r, phi= cart2pol(xx_rot,yy_rot)

    ''' 
    Create the OAM beams that we need to sort 
    '''
    # Now create a list containing 'oamMode' objects 

    list_of_OAMs = []

    if(isKnot):
        for ii in range(len(knotType)):
            field = setKnotType(r, phi, w0, knotType[ii], shapeParams[ii])
            list_of_OAMs.append(oamModes(field, output_chans[ii]))
    else:
        for ii in range(len(LG_modes)):
            list_of_OAMs.append(oamModes(LG(r, phi, LG_modes[ii][0], LG_modes[ii][1], w0,h,0,k), output_chans[ii]))
    
    
    #plt.imshow(np.angle(list_of_OAMs[0].oamBeam))
    #plt.show()
    #input()
    return list_of_OAMs


# NEW: apply rotation onto the incident field
#print("yep")
#list_of_rotated_OAMs = create_rotated_knots(rot_angle)
#print("okay")
    
'''
We run this at the end of every generation. Here, we save the best parameters after every generation
'''

def on_gen(ga_instance):
    print(np.shape(ga_instance.population))
    print("Generation : ", ga_instance.generations_completed)
    print("Fitness of the best solution :", ga_instance.best_solution()[1])
    solution =  ga_instance.best_solution()[0]
    # Checkpoint current best phase patterns. 
    
    ga_instance_name = cnfg['ga_instance']
    
    # Create the phase map(s) by reshaping the solution array
    
    phase_maps = np.empty((num_of_phase_maps, N, N))
    
    for ii in range(num_of_phase_maps):
        # Reshape and apply filter to solutions 
        temp = np.reshape(solution[(ii)*N**2:(ii+1)*N**2], shape=(N,N))
        # Apply gaussian filter 
        temp = sp.ndimage.gaussian_filter(temp, sigma=maxx*GFilterStrength)
        phase_maps[ii] = temp

    # Create best_phases  and ga_instance directory if it doesn't exist
    if not os.path.exists("best_phases"):
        os.makedirs("best_phases")
    
    if not os.path.exists("genetic_instances"):
        os.makedirs("genetic_instances")
    
    with open(f"best_phases/{ga_instance_name}.pkl", 'wb') as file:
        pkl.dump(phase_maps, file)
    
    ga_instance.save(filename=f'genetic_instances/{ga_instance_name}')

    # Create new directory to save the plots (if it doesn't already exist)
    
    if not os.path.exists(f"plots/{ga_instance_name}"):
        os.makedirs(f"plots/{ga_instance_name}")
    
    # Save plot every 100 generations 
    
    if (ga_instance.generations_completed % 1000 == 0):
        plt.figure()
        plt.plot(ga_instance.best_solutions_fitness)
        plt.savefig(f"plots/{ga_instance_name}/fitness_{ga_instance.generations_completed}.jpg")
        plt.show()
        
        
def compute_sorting_performance(phase_maps, list_of_OAMs, alpha=1.0):

    # Make the dimensionality of our sorting in terms of # of modes
    d = len(list_of_OAMs)
    
    # Now, this is the fitness parameter 
    #sorting_performance = 0
    sorting_performance = np.zeros(d)

    # Actually, let's introduce the crosstalk matrix 
    crosstalk_matrix = np.zeros((num_of_output_chans, num_of_output_chans))
    
    # Let's introduce the secret key rate here, actually. 
    secret_key = 0

    for ii in range(d):

        # Define initial OAM field and correct output channel 

        field = list_of_OAMs[ii].oamBeam 
        
        # Do a proper normalization on the incident field 
        
        field = norm_field(field,h)
        
        # Compute the initial field intensity. This will be important for later
        
        int_knot = np.sum(np.abs(field)**2)
    
        # modulate the field by the first phase map 

        field_mod_1 = field*phase_maps[0]

        # Proceed with our chosen experiment

        # Case 1: We are doing the experiment without any lenses

        if (multiPhase): # Propagate the field by a distance z_o and apply the second phase screen
            field_after = field_mod_1

            for jj in range(1, len(phase_maps)):
                # Propagate the beam by a distance z_o
                field_after = propTF(field_after, maxx, la, z_o)

                # Apply the next phase map (if applicable)
                field_after = field_after*phase_maps[jj]

            # Propagate the beam one final time and observe the final field

            final_field = propTF(field_after, maxx, la, z_o)

        else: # Case 2: We are computing the experiment with the lens 

            if (multiPhaseLens): # Multi-phase experiment with the lens
                field_after = field_mod_1

                for kk in range(1, num_phase_maps_near):
                    # Propagate the beam by a distance z_o 
                    field_after = propTF(field_after, maxx, la, z_o)
                    # Apply the next phase map in the near field (if applicable)
                    field_after = field_after*phase_maps[kk]
                
                # Fourier transform the beam into the far field
                field_lens = fftshift(fft2(field_after))

            elif (simulateLens): # We simulate Faunhofer Diffraction for a more accurate representation of lens propagation
                field_lens, _ = propFF(field_mod_1,maxx,la,fourier_lens)
        
            
            else: # Compute the field at the front focal plane of the lens
                 field_lens = fftshift(fft2(field_mod_1))
        
            # What happens next depends on whether we have one or two phase maps
        
            if (num_phase_maps_far==0):
                # Compute the field intensity 
                final_field = field_lens
        
            else:
                # Modulate the field by the first far field map
                field_mod_2 = field_lens*phase_maps[num_phase_maps_near]
                if (multiPhaseLens):
                    field_after_2 = field_mod_2
                    
                    for ll in range(1+num_phase_maps_near, num_of_phase_maps):
                        # Propagate the beam 
                        field_after_2 = propTF(field_after_2, maxx, la, z_o)
                        # Apply phase to beam 
                        field_after_2 = field_after_2*phase_maps[ll]
            
                 # Apply inverse fourier transform onto beam
                    field_lens_2 = ifft2(ifftshift(field_after_2))

                 # simulate the lens field again. This is the final field. 
                elif (simulateLens):
                    field_lens_2, _ = propFF(field_mod_2, maxx, la, fourier_lens)
                else: 
                    field_lens_2 = ifft2(ifftshift(field_mod_2))

                final_field = field_lens_2
        
        # We normalize the final field and compute the intensity 
        
        final_field = norm_field(final_field,h)
        final_field_int = np.abs(final_field)**2
        
        # Define full set of indices, as you would summing through a for loop
        full_index = np.arange(len(output_chans))   
        # Delete ii from the list of full_index, creating a new temporary array
        temp_index = np.delete(full_index, ii)
        # Sum up the "incorrect" channels 
        incorrect_chans = 0
        # New: to construct our crosstalk matrix, let's store the individual intensities
        incorrect_chan_ints = []
        
        for ind in temp_index:
            field_in_pupil = final_field_int*output_chans[ind]
            incorrect_chan_ints.append(np.sum(field_in_pupil)/int_knot)
            incorrect_chans += np.sum(field_in_pupil)/int_knot
            
        # Now, evaluate the sorting performance 
        correct_chans = np.sum(final_field_int*output_chans[ii])/int_knot # normalization is mode-specific
        sorting_performance[ii] = correct_chans - incorrect_chans
        
        # Compute the detector effeciency 
        detect_eff = correct_chans 
        crosstalk_matrix[ii,ii] = detect_eff 
        
        # Compute the crosstalk matrix. For more than two modes, we have to be a bit more meticulous with our approach. 
        
        for jj, ind in enumerate(temp_index):
            crosstalk_eff = incorrect_chan_ints[jj]
            crosstalk_matrix[ii, ind] = crosstalk_eff

    # Compute the "QBER" using the off-diagonals of the crosstalk matrix 
    qber = ((d-1)/d**2)*(crosstalk_matrix.sum() - np.trace(crosstalk_matrix)) # This bounds the qber to 1, in principle

    # Compute the secret key rate
    secret_key = np.log2(d) - 2*shannon_entropy(qber,d)

    # NEW: We keep track of the per-channel sorting performance
    # alpha is a hyperparameter which adjusts the weight between the minimum and the sum of sorting performances across each channel. 
    # At alpha=0.0, we recover the old behavior, while at alpha=1.0, we consider the minimum
    
    print(sorting_performance)

    overall_sort_perf = alpha*np.min(sorting_performance) - ((1-alpha)/d)*np.sum(sorting_performance)

    return overall_sort_perf, crosstalk_matrix, secret_key
    
'''
This computes the fitness function that we use to improve the GA. We can adapt this to one or two phase maps
'''

def fitness_func_sorting(ga_instance, solution, solution_idx):
    
    if (randomRotation): # Instead of a fixed rotation, sample a random rotation angle from a uniform distribution. 
        rotation_angle = np.random.uniform(0, 2*np.pi)
    else: 
        rotation_angle = rot_angle
    # Create the phase map(s) by reshaping the solution array
    phase_maps = np.empty((num_of_phase_maps, N, N), dtype=np.complex128)
    
    # NEW: apply rotation onto the incident field
    list_of_rotated_OAMs = create_rotated_knots(rotation_angle)

    for ii in range(num_of_phase_maps):
        # Reshape solution to phase map 
        temp = np.reshape(solution[(ii)*N**2:(ii+1)*N**2], shape=(N,N))
        # Apply gaussian filter 
        temp = sp.ndimage.gaussian_filter(temp, sigma=maxx*GFilterStrength)
        phase_maps[ii] = np.exp(1j*temp)
    
    # Compute sorting performance 
    sorting_performance,*_ = compute_sorting_performance(phase_maps, list_of_rotated_OAMs)
    #print(sorting_performance)
    
    return np.abs(sorting_performance)

def fitness_func_crosstalk(ga_instance, solution, solution_idx):
    # Create the phase map(s) by reshaping the solution array
    phase_maps = np.empty((num_of_phase_maps, N, N), dtype=np.complex128)

    for ii in range(num_of_phase_maps):
        # Reshape solution to phase map 
        temp = np.reshape(solution[(ii)*N**2:(ii+1)*N**2], shape=(N,N))
        # Apply gaussian filter 
        temp = sp.ndimage.gaussian_filter(temp, sigma=maxx*GFilterStrength)
        phase_maps[ii] = np.exp(1j*temp)
    
    # Compute sorting performance 
    sorting_performance, crosstalk_matrix, _ = compute_sorting_performance(phase_maps)
    
    # Product of the sorting performance w/ determinant of the crosstalk matrix makes up our new metric. 
    crosstalk_neo = sorting_performance*np.linalg.det(crosstalk_matrix)

    return np.real(crosstalk_neo)


def fitness_func_secretKey(ga_instance, solution, solution_idx):
    
    if (randomRotation): # Instead of a fixed rotation, sample a random rotation angle from a uniform distribution. 
        rotation_angle = np.random.uniform(0, 2*np.pi)
    elif (fixedRotation):
        rotation_angle = rot_angle
    # Create the phase map(s) by reshaping the solution array
    phase_maps = np.empty((num_of_phase_maps, N, N), dtype=np.complex128)
    
    # NEW: apply rotation onto the incident field
    list_of_rotated_OAMs = create_rotated_knots(rotation_angle)

    for ii in range(num_of_phase_maps):
        # Reshape solution to phase map 
        temp = np.reshape(solution[(ii)*N**2:(ii+1)*N**2], shape=(N,N))
        # Apply gaussian filter 
        temp = sp.ndimage.gaussian_filter(temp, sigma=maxx*GFilterStrength)
        phase_maps[ii] = np.exp(1j*temp)
    
    # Compute sorting performance 
    sorting_performance, _, secret_key = compute_sorting_performance(phase_maps, list_of_rotated_OAMs)
    
    sorting_performance_neo = sorting_performance*secret_key
    
    return np.real(sorting_performance_neo)


def fitness_func_secretKey_crosstalk(ga_instance, solution, solution_idx):

    if (randomRotation): # Instead of a fixed rotation, sample a random rotation angle from a uniform distribution. 
        rotation_angle = np.random.uniform(0, 2*np.pi)
    elif (fixedRotation):
        rotation_angle = rot_angle
    
    # Create the phase map(s) by reshaping the solution array
    phase_maps = np.empty((num_of_phase_maps, N, N), dtype=np.complex128)
    
    # NEW: apply rotation onto the incident field
    list_of_rotated_OAMs = create_rotated_knots(rotation_angle)
    
    for ii in range(num_of_phase_maps):
        # Reshape solution to phase map 
        temp = np.reshape(solution[(ii)*N**2:(ii+1)*N**2], shape=(N,N))
        # Apply gaussian filter 
        temp = sp.ndimage.gaussian_filter(temp, sigma=maxx*GFilterStrength)
        phase_maps[ii] = np.exp(1j*temp)
    
    # Compute sorting performance 
    sorting_performance, crosstalk_matrix, secret_key = compute_sorting_performance(phase_maps, list_of_rotated_OAMs)
    
    sorting_performance_neo = sorting_performance*secret_key*np.linalg.det(crosstalk_matrix)
    
    return np.real(sorting_performance_neo)
    
# c and k are empirical scaling factors that control the probability distribution. 
# c determines how well favoured fit individuals are
# k determines how peaked is the p-dist. 

def exp_rank_selection(fitness, num_parents, ga_instance):
    
    fitness_sorted = sorted(range(len(fitness)), key=lambda l: fitness[l])
    fitness_sorted.reverse()

    # Create ranks 
    ranks = np.arange(1, ga_instance.sol_per_pop+1)

    # Compute probabilities according to exponential selection routine
    probs = parent_c * (1 - np.exp(-ranks/parent_k))
    
    # **CRITICAL: Normalize probabilities to sum to 1**
    probs = probs / np.sum(probs)
    
    probs_start, probs_end, parents = ga_instance.wheel_cumulative_probs(
        probs=probs.copy(), 
        num_parents=num_parents
    )
    parents_indices = []

    for parent_num in range(num_parents):
        rand_prob = np.random.rand()
        selected = False
        for idx in range(probs.shape[0]):
            if (rand_prob >= probs_start[idx] and rand_prob < probs_end[idx]):
                mapped_idx = fitness_sorted[idx]
                parents[parent_num, :] = ga_instance.population[mapped_idx, :].copy()
                parents_indices.append(mapped_idx)
                selected = True
                break
        
        # **Safety check: if no bin selected, choose best individual**
        if not selected:
            mapped_idx = fitness_sorted[0]
            parents[parent_num, :] = ga_instance.population[mapped_idx, :].copy()
            parents_indices.append(mapped_idx)
                
    return parents, np.array(parents_indices)


# In principle, we would save the last population of the previous GA instance, then rerun a second GA using this population as the starting one 

def on_stop(ga_instance, last_population_fitness):
    print("Initial optimization done, saving last population...")
    global last_pop 
    last_pop = ga_instance.population 
    

print("Beginning optimization...") 

# Print experiment configuration details

print("\n" + "="*80)
print("EXPERIMENT CONFIGURATION")
print("="*80)

# Determine experiment type
if multiPhaseLens:
    experiment_type = "Multi-Phase Experiment WITH Lenses"
    print(f"Experiment Type: {experiment_type}")
    print(f"  - Number of phase screens in near field: {num_phase_maps_near}")
    print(f"  - Number of phase screens in far field: {num_phase_maps_far}")
    print(f"  - Distance between phase screens (z_o): {z_o/cm:.2f} cm")
    print(f"  - Fourier lens length: {fourier_lens/cm:.2f} cm")
elif multiPhase:
    experiment_type = "Multi-Phase Experiment (No Lenses)"
    print(f"Experiment Type: {experiment_type}")
    print(f"  - Number of phase screens: {num_of_phase_maps}")
    print(f"  - Distance between phase screens (z_o): {z_o/cm:.2f} cm")
elif simulateLens:
    experiment_type = "Lens Experiment (Simulated Fraunhofer Diffraction)"
    print(f"Experiment Type: {experiment_type}")
    print(f"  - Number of phase screens: {num_of_phase_maps}")
    print(f"  - Fourier lens length: {fourier_lens/cm:.2f} cm")
else:
    experiment_type = "Standard Lens Experiment (Fourier Transform)"
    print(f"Experiment Type: {experiment_type}")
    print(f"  - Number of phase screens: {num_of_phase_maps}")

print(f"\nTotal Phase Screens Being Optimized: {num_of_phase_maps}")
print(f"  - Total genes in solution: {num_genes} ({num_of_phase_maps} × {N}²)")

# Print knot/mode parameters being sorted
print(f"\nModes Being Sorted:")
if isKnot:
    print(f"  - Type: Knotted Beams")
    print(f"  - Number of knots: {len(knotType)}")
    for idx, (knot, params) in enumerate(zip(knotType, shapeParams)):
        print(f"    [{idx+1}] Knot Type: {knot}, Shape Parameters: {params}")
else:
    print(f"  - Type: Laguerre-Gaussian (LG) Modes")
    print(f"  - Number of modes: {len(LG_modes)}")
    for idx, mode in enumerate(LG_modes):
        print(f"    [{idx+1}] LG Mode: (l={mode[0]}, p={mode[1]})")

print(f"  - Beam waist (w0): {w0/mm:.2f} mm")
print(f"  - Number of output channels: {num_of_output_chans}")

# Print rotation settings if applicable
if randomRotation:
    print(f"\nRotation: Random (0 to 2π)")
elif fixedRotation:
    print(f"\nRotation: Fixed angle = {rot_angle} rad")
else:
    print(f"\nRotation: None")

print(f"\nFitness Function: {fitness_function}")
print(f"\nSorting Performance Alpha: {alpha}")

print("="*80 + "\n")

# Select fitness function after initial optimization
if fitness_function == 'secret_key':
    fitness_func = fitness_func_secretKey
elif fitness_function == 'bread':
    fitness_func = fitness_func_secretKey_crosstalk

# We begin by optimizing just the sorting performance for the first start_gen generations

ga_instance_sorting = pygad.GA(num_generations=gen_start,
                       num_parents_mating=num_parents_mating,
                       fitness_func=fitness_func,
                       sol_per_pop=sol_per_pop,
                       num_genes=num_genes,
                       init_range_low=init_range_low,
                       init_range_high=init_range_high,
                       parent_selection_type=exp_rank_selection,
                       crossover_type=crossover_type,
                       mutation_type=mutation_type,
                       mutation_probability = mutation_probability,
                       random_mutation_min_val = random_mutation_min_val, 
                       random_mutation_max_val = random_mutation_max_val, 
                       on_generation=on_gen, 
                       keep_elitism = keep_elitism,
                       on_stop = on_stop)


# We then start another GA instance w/ the last population where we start to optimize together the sorting performance and the determinant. 

ga_instance_sorting.run()

ga_instance_crosstalk= pygad.GA(num_generations=num_generations,
                       num_parents_mating=num_parents_mating,
                       fitness_func=fitness_func,
                       sol_per_pop=sol_per_pop,
                       num_genes=num_genes,
                       init_range_low=init_range_low,
                       init_range_high=init_range_high,
                       initial_population = last_pop, 
                       parent_selection_type=exp_rank_selection,
                       crossover_type=crossover_type,
                       mutation_type=mutation_type,
                       mutation_probability = mutation_probability,
                       random_mutation_min_val = random_mutation_min_val, 
                       random_mutation_max_val = random_mutation_max_val, 
                       on_generation=on_gen, 
                       keep_elitism = keep_elitism,
                       stop_criteria=f"saturate_{gen_saturate}")


ga_instance_crosstalk.run()
