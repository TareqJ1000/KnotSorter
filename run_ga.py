# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 14:15:29 2024

@author: tjaou104
"""

import numpy as np 
import scipy as sp
import pygad
import yaml 
import argparse
from yaml import Loader 
import pickle as pkl

from optical_functions import LG, propFF, cart2pol, oamModes, output_chan, output_chan_symmetric, setKnotType, OAMWithGratings, Hologram, wrap_to_domain 
from scipy.fft import ifft2, ifftshift, fft2, fftshift

import matplotlib.pyplot as plt 

from diffractsim import cm, mm, um 
import os

# Remove if used outside of the cluster 

parser=argparse.ArgumentParser(description='test')
parser.add_argument('--ii', dest='ii', type=int,
    default=None, help='')
args = parser.parse_args()
shift = args.ii
shift = 1


# This function keeps track of the generation number + best fitness

# Load configuration file

stream = open(f"configs/ga{shift}.yaml", 'r')
cnfg = yaml.load(stream, Loader=Loader)

''' 
Global/Optimization Parameters 
'''

N = cnfg['dim']
num_of_output_chans = cnfg['num_output_chans']
output_chan_width = cnfg['output_chan_width'] * mm # in mm 
channel_sep = cnfg['channel_sep'] 

# Some parameters specifying the LG modes

LG_modes = cnfg['LG_modes']
w0 = cnfg['w0'] * mm # in mm!!

isKnot = cnfg['isKnot']
knotType = cnfg['knotType']
shapeParams = cnfg['shapeParams']
num_of_phase_maps = cnfg['num_maps'] # can be 1 or 2!

simulateLens = cnfg['simulateLens'] # Do we simulate the phase effects of our lenses, or do we neglect them and take the fourier and inverse fourier transform? 
fourier_lens = cnfg['fourier_length']*cm # fourier length of both lens in cm
GFilterStrength = cnfg['gauss_filter_sigma'] # sigma parameter for the gaussian filter .. apply to initial population and in computing the fitness param. 

'''
GA Parameters
'''

num_generations = int(eval(cnfg['num_of_gens']))
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

mutation_type = "random"
mutation_probability = cnfg['mutation_prob'] # probability of mutation 
mutation_percent_genes = cnfg['mutation_percent'] # Percentage of genes to mutate 
random_mutation_min_val = -np.pi
random_mutation_max_val =  np.pi

gen_saturate = cnfg['gen_saturate']
ga_instance_name = cnfg['ga_instance']

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

xx,yy=np.meshgrid(X,Y);
r, phi= cart2pol(xx,yy)

''' 
Create the OAM beams that we need to sort 
'''
# Now create a list containing 'oamMode' objects 

list_of_OAMs = []
output_chans = output_chan_symmetric(X,Y,output_chan_width,maxx,num_of_output_chans, chan_sep=channel_sep)

if(isKnot):
    for ii in range(len(knotType)):
        list_of_OAMs.append(oamModes(setKnotType(r, phi, w0, knotType[ii], shapeParams[ii]), output_chans[ii]))
        
else:
    for ii in range(len(LG_modes)):
        list_of_OAMs.append(oamModes(LG(r, phi, LG_modes[ii][0], LG_modes[ii][1], w0,h,0,k), output_chans[ii]))
        
    
print(list_of_OAMs)


    
'''
We run this at the end of every generation. Here, we save the best parameters after every generation
'''

def on_gen(ga_instance):
    print("Generation : ", ga_instance.generations_completed)
    print("Fitness of the best solution :", ga_instance.best_solution()[1])
    solution =  ga_instance.best_solution()[0]
    # Checkpoint current best phase patterns. 
    
    ga_instance_name = cnfg['ga_instance']
    
    # Create the phase map(s) by reshaping the solution array
    
    phase_maps = np.empty((num_of_phase_maps, N, N))
    
    for ii in range(num_of_phase_maps):
        # Reshape and apply filter to solutions 
        temp = np.reshape(solution[(ii)*N**2:(ii+1)*N**2], newshape=(N,N))
        # Apply gaussian filter 
        temp = sp.ndimage.gaussian_filter(temp, sigma=maxx*GFilterStrength)
        phase_maps[ii] = temp

    
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
'''
This computes the fitness function that we use to improve the GA. We can adapt this to one or two phase maps
'''

def fitness_func(ga_instance, solution, solution_idx):

    # Create the phase map(s) by reshaping the solution array
    phase_maps = np.empty((num_of_phase_maps, N, N), dtype=np.complex_)

    for ii in range(num_of_phase_maps):
        # Reshape solution to phase map 
        temp = np.reshape(solution[(ii)*N**2:(ii+1)*N**2], newshape=(N,N))
        # Apply gaussian filter 
        temp = sp.ndimage.gaussian_filter(temp, sigma=maxx*GFilterStrength)
        phase_maps[ii] = np.exp(1j*temp)

    # Now, this is the fitness parameter 

    sorting_performance = 0  

    for ii in range(len(list_of_OAMs)):

        # Define initial OAM field and correct output channel 

        field = list_of_OAMs[ii].oamBeam 
        
        # Do a rough normalization on the incident field 
        
        field = field/np.max(np.abs(field))

        # modulate the field by the first phase map 

        field_mod_1 = field*phase_maps[0]

        # let's simulate the propagation of the lens
        if (simulateLens):
            field_lens, _ = propFF(field_mod_1,maxx,la,fourier_lens)
        else: # Take the fourier transform 
            field_lens = fftshift(fft2(field_mod_1))
        
        # What happens next depends on whether we have one or two phase maps
        
        if(num_of_phase_maps==1):
            # Compute the field intensity 
            final_field = field_lens
        else:
            # modulate the field by the second phase map 
            field_mod_2 = field_lens*phase_maps[1]
            # simulate the lens field again. This is the final field. 
            if (simulateLens):
                field_lens_2, _ = propFF(field_mod_2, maxx, la, fourier_lens)
            else: 
                field_lens_2 = ifft2(ifftshift(field_mod_2))
            # compute the field intensity 
            # field_lens_2 = field_lens_2/np.max(np.abs(field_lens_2))
            
            final_field = field_lens_2
            #final_field_int = np.abs(field_lens_2)**2
        
        # We normalize the final field and compute the intensity 
        final_field = final_field/np.max(np.abs(final_field))
        final_field_int = np.abs(final_field)**2
        
        # Define full set of indices, as you would summing through a for loop
        full_index = np.arange(len(output_chans))   
        # Delete ii from the list of full_index, creating a new temporary array
        temp_index = np.delete(full_index, ii)
        # Sum up the "incorrect" channels 
        incorrect_chans = 0
        for ind in temp_index:
            field_in_pupil = final_field_int*output_chans[ind]
            incorrect_chans += np.sum(field_in_pupil)
        # Now, evaluate the sorting performance 
        correct_chans = np.sum(final_field_int*output_chans[ii])
        sorting_performance += correct_chans - incorrect_chans
     
    return sorting_performance

'''
Here, we create an initial population in reminisce of actual OAM holograms
'''

def initialize_population(sol_per_pop, N, num_phase_maps):
    # Start with empty array to hold our starting maps
    
    init_pop = np.empty((sol_per_pop, num_phase_maps*N**2))
    
    for ii in range(sol_per_pop):
        for jj in range(num_phase_maps):
        
            # Stochastic Generator
            oamMode = np.random.randint(-2,2)
            a = [np.random.uniform(0.0,1.0) for ii in range(4)]
            x_offset = np.random.uniform(low=0.0, high=1.0)
            y_offset = np.random.uniform(low=0.0, high=1.0)
            
            # We may apply a random, normally distributed map w/ gaussian mean 
            gauss_mean = np.random.normal(0,0.1,(N,N))
            
            final_field = OAMWithGratings(oamMode,N,N,x_offset, y_offset, a) + gauss_mean
            
            init_pop[ii,(jj)*N**2:(jj+1)*N**2] = final_field.flatten()
    return init_pop


def initialize_population_blazed(sol_per_pop, N, sigma, num_phase_maps, isKnot):
    # Start with empty array to hold our starting maps
    
    init_pop = np.empty((sol_per_pop,num_phase_maps, N, N))
    
    for ii in range(sol_per_pop):
        for jj in range(num_phase_maps):
        
            # Stochastic Generator
            # Filp a coin
            coinfilp = np.random.randint(0,2)
            
            if(isKnot == False):
                oamMode = np.random.randint(-2,3)
                LA = la*(1/np.random.uniform(0,1))
                initial_field = LG(r, phi, oamMode, 0, w0, h, 0, k)
            else:
                # Filp another coin to determine trefoil or cinquefoil 
                coinfilp_2 = np.random.randint(0,2)
                if (coinfilp_2 == 0):
                    knotType='Trefoil'
                else:
                    knotType='Cinquefoil'
                a = np.random.uniform(0,1)
                b = np.random.uniform(0,1)
                s = 1.2
                initial_field = setKnotType(r, phi, w0, knotType, shapeParams) 

            initial_field = initial_field/np.max(np.abs(initial_field))
            LA = la*(1/np.random.uniform(0,1))
    
            # Stepsize in the x and y directions are randomized. This also applies stochasticsity on the generated maps
        
            hx = np.random.uniform(1e-3*h, h)
            hy = np.random.uniform(1e-3*h, h)

            # We may apply a random, normally distributed map w/ gaussian mean 
            gauss_mean = np.random.normal(0,0.1,(N,N))
            final_field = wrap_to_domain(Hologram(initial_field, hx, hy, LA))
            
            final_field += gauss_mean
            
            # Apply a gaussian filter, too
            final_field = sp.ndimage.gaussian_filter(final_field, sigma=sigma)
            final_field =  np.pi*(np.tanh(final_field))
            init_pop[ii,jj] = final_field
        
    return init_pop

# c and k are empirical scaling factors that control the probability distribution. 
# c determines how well favoured fit individuals are
# k determines how peaked is the p-dist. 


def exp_rank_selection(fitness, num_parents, ga_instance):
    
    fitness_sorted = sorted(range(len(fitness)), key=lambda l: fitness[l])
    fitness_sorted.reverse()

    parents_sorted = np.empty((num_parents, ga_instance.population.shape[1]))

    # Create ranks 
    ranks = np.arange(1, ga_instance.sol_per_pop+1)

    # Now, compute the probabilities according to exponential selection routine
    probs = parent_c*(1 - np.exp(-ranks/parent_k))
    
    probs_start, probs_end, parents = ga_instance.wheel_cumulative_probs(probs=probs.copy(), 
                                                              num_parents=num_parents)
    parents_indices = []

    for parent_num in range(num_parents):
        rand_prob = np.random.rand()
        for idx in range(probs.shape[0]):
            if (rand_prob >= probs_start[idx] and rand_prob < probs_end[idx]):
            # The variable idx has the rank of solution but not its index in the population.
            # Return the correct index of the solution.
                mapped_idx = fitness_sorted[idx]
                parents[parent_num, :] = ga_instance.population[mapped_idx, :].copy()
                parents_indices.append(mapped_idx)
                break
                
    return parents, np.array(parents_indices)

# IT BEGINS

fitness_function = fitness_func 
ga_instance = pygad.GA(num_generations=num_generations,
                       num_parents_mating=num_parents_mating,
                       fitness_func=fitness_function,
                       sol_per_pop=sol_per_pop,
                       num_genes=num_genes,
                       init_range_low=init_range_low,
                       init_range_high=init_range_high,
                       parent_selection_type=exp_rank_selection,
                       crossover_type=crossover_type,
                       mutation_type=mutation_type,
                       mutation_percent_genes=mutation_percent_genes,
                       mutation_probability = mutation_probability,
                       random_mutation_min_val = random_mutation_min_val, 
                       random_mutation_max_val = random_mutation_max_val, 
                       on_generation=on_gen, 
                       stop_criteria=f"saturate_{gen_saturate}")

ga_instance.run()





