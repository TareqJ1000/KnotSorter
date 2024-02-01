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

from optical_functions import LG, propFF, cart2pol, oamModes, output_chan, setKnotType, OAMWithGratings

import matplotlib.pyplot as plt 

from diffractsim import cm, mm, um 
import os

# Remove if used outside of the cluster 

parser=argparse.ArgumentParser(description='test')
parser.add_argument('--ii', dest='ii', type=int,
    default=None, help='')
args = parser.parse_args()
shift = args.ii


shift=1

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

# Some parameters specifying the LG modes

LG_modes = cnfg['LG_modes']
w0 = cnfg['w0'] * mm # in mm!!

isKnot = cnfg['isKnot']
knotType = cnfg['knotType']
shapeParams = cnfg['shapeParams']
num_of_phase_maps = cnfg['num_maps'] # can be 1 or 2!

fourier_lens = cnfg['fourier_length']*cm # fourier length of both lens in cm


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

parent_selection_type = "rank"
#K_tournament = 5 # number of contestants, essentially
#keep_elitism  = 1

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

la = 0.5*um
k=(2*np.pi)/la  # [m^-1] wavenumber    
maxx = 1.5 * mm # Full length of the numerical window (m)
N=128 # [Number of points per dimension]

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
output_chans = output_chan(X,Y,output_chan_width,maxx,num_of_output_chans)

if(isKnot):
    for ii in range(len(knotType)):
        list_of_OAMs.append(oamModes(setKnotType(r, phi, w0, knotType[ii], shapeParams[ii]), output_chans[ii]))
        
else:
    for ii in range(len(LG_modes)):
        list_of_OAMs.append(oamModes(LG(r, phi, LG_modes[ii][0], LG_modes[ii][1], w0,h,0,k), output_chans[ii]))

    

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
        phase_maps[ii] = np.exp(1j*np.reshape(a=solution[(ii)*N**2:(ii+1)*N**2], newshape = (N,N)))
    
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
    phase_maps = np.empty((num_of_phase_maps, N, N))

    
    for ii in range(num_of_phase_maps):
        phase_maps[ii] = np.exp(1j*np.reshape(a=solution[(ii)*N**2:(ii+1)*N**2], newshape = (N,N)))

    # Now, this is the fitness parameter 

    sorting_performance = 0  

    for ii in range(len(list_of_OAMs)):

        # Define initial OAM field and correct output channel 

        field = list_of_OAMs[ii].oamBeam 

        # modulate the field by the first phase map 

        field_mod_1 = field*phase_maps[0]

        # let's simulate the propagation of the lens

        field_lens, _ = propFF(field_mod_1,maxx,la,fourier_lens)
        
        # Now, what happens next depends on the number of phase patterns
        
        if (num_of_phase_maps==1):
            final_field_int = np.abs(field_lens)**2
        
        else:
            # modulate the field by the second phase map 
    
            field_mod_2 = field_lens*phase_maps[1]
    
            # simulate the lens field again. This is the final field. 
    
            field_lens_2, _ = propFF(field_mod_2, maxx, la, fourier_lens)
    
            # compute the field intensity 
    
            final_field_int = np.abs(field_lens_2)**2

        # Define full set of indices, as you would summing through a for loop

        full_index = np.arange(len(output_chans))   

        # Delete ii from the list of full_index, creating a new temporary array

        temp_index = np.delete(full_index, ii)

        # Sum up the "incorrect" channels 

        incorrect_chans = 0

        for ind in temp_index:
            field_in_pupil = final_field_int*output_chans[ind]
            incorrect_chans += np.abs(field_in_pupil)**2

        # Now, evaluate the sorting performance 

        correct_chans = np.abs(final_field_int*output_chans[ii])**2

        sorting_performance += correct_chans - incorrect_chans 

    return np.mean(sorting_performance)



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




# IT BEGINS

fitness_function = fitness_func 
ga_instance = pygad.GA(num_generations=num_generations,
                       num_parents_mating=num_parents_mating,
                       fitness_func=fitness_function,
                       sol_per_pop=sol_per_pop,
                       num_genes=num_genes,
                       init_range_low=init_range_low,
                       init_range_high=init_range_high,
                       parent_selection_type=parent_selection_type,
                       crossover_type=crossover_type,
                       mutation_type=mutation_type,
                       mutation_percent_genes=mutation_percent_genes,
                       mutation_probability = mutation_probability,
                       random_mutation_min_val = random_mutation_min_val, 
                       random_mutation_max_val = random_mutation_max_val, 
                       on_generation=on_gen, 
                       stop_criteria=f"saturate_{gen_saturate}",
                       initial_population=initialize_population(sol_per_pop,N,num_of_phase_maps))

ga_instance.run()





