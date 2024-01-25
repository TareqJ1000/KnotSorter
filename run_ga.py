# -*- coding: utf-8 -*-
"""
Created on Thu Jan 25 14:15:29 2024

@author: tjaou104
"""

import numpy as np 
import scipy as sp
import pygad
import yaml 
from yaml import Loader 

from optical_functions import LG, propFF, cart2pol, oamModes, output_chan, setKnotType

import matplotlib.pyplot as plt 

from diffractsim import cm, mm, um 
import os

# This function keeps track of the generation number + best fitness

# Load configuration file

stream = open(f"configs/ga.yaml", 'r')
cnfg = yaml.load(stream, Loader=Loader)

def on_gen(ga_instance):
    print("Generation : ", ga_instance.generations_completed)
    print("Fitness of the best solution :", ga_instance.best_solution()[1])
    
    # Checkpoint current best model?
    ga_instance_name = cnfg['ga_instance']
    ga_instance.save(filename=f'ga_instances/{ga_instance_name}')
    
    
    # Create new directory to save the plots (if it doesn't already exist)
    
    if not os.path.exists(f"plots/{ga_instance_name}"):
        os.makedirs(f"plots/{ga_instance_name}")
    
    # Save plot every 100 generations 
    
    
    if (ga_instance.generations_completed % 100 == 0):
        plt.figure()
        plt.plot(ga_instance.best_solutions_fitness)
        plt.savefig(f"plots/{ga_instance_name}/fitness_{ga_instance.generations_completed}.jpg")
        plt.show()


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


fourier_lens = cnfg['fourier_length']*cm # fourier length of both lens in cm


'''
GA Parameters
'''

num_generations = int(eval(cnfg['num_of_gens']))
num_parents_mating = cnfg['parents_mating']

sol_per_pop = cnfg['sol_per_pop'] # number of parents in the population?? 
num_genes = 2*N**2 # This would refer to the number of parameters in our DNA

# Lower and upper-bound ranges of the parameterization. 

init_range_low = -np.pi
init_range_high = np.pi

parent_selection_type = "rank"
#K_tournament = 5 # number of contestants, essentially
#keep_elitism  = 1

crossover_type = "single_point"

mutation_type = "random"
mutation_percent_genes = cnfg['mutation_percent'] # probability of mutation 

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
This computes the fitness function that we use to improve the GA
'''

def fitness_func(ga_instance, solution, solution_idx):

    # Create the phase map by reshaping the solution array

    reshape_phase_1 = np.reshape(a=solution[0:N**2], newshape = (N,N))
    reshape_phase_2 = np.reshape(a=solution[N**2:2*N**2], newshape=(N,N))
    phase_map_1 = np.exp(1j*reshape_phase_1)
    phase_map_2 = np.exp(1j*reshape_phase_2)
    
    # Now, this is the fitness parameter 

    sorting_performance = 0  

    for ii in range(len(list_of_OAMs)):

        # Define initial OAM field and correct output channel 

        field = list_of_OAMs[ii].oamBeam 

        # modulate the field by the first phase map 

        field_mod_1 = field*phase_map_1

        # let's simulate the propagation of the lens

        field_lens, _ = propFF(field_mod_1,maxx,la,fourier_lens)

        # modulate the field by the second phase map 

        field_mod_2 = field_lens*phase_map_2

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
                       on_generation=on_gen)

ga_instance.run()
ga_instance.plot_fitness()


