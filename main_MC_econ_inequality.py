# -*- coding: utf-8 -*-
"""
Created on Thu Mar 21 14:18:01 2019

@author: EzechieL
"""

import numpy as np
import matplotlib.pyplot as plt
import math
#consumption_coeff_vect = [1, 0.8, 0.5, 0.3, 0.1]




nb_rep = 20
run_loop(nb_rep,expansion_coeff = 1)






def run_loop(nb_rep,expansion_coeff = 1000000):
    
    total_pop = 300*expansion_coeff
    #♦ POP USA 2018 :           327,200,000 
    # Monetary base USA : 3,584,502,000,000$
    
    # Equiv : Pop= 300 / Money = 3,500,000
    M_zero = 3500000*expansion_coeff # Total qty of Money in the system
    nb_firms = 100*expansion_coeff
    keep_min = M_zero
    for nb_consum_catg in range(1,nb_rep):
         #nb_consum_catg = 2
        nb_categories = nb_consum_catg
        #consumption_coeff_vect = [1, 0.5]
        
        start_val = 0.001
        max_val = 1
        consumption_coeff_vect = [start_val+(i+1)*(max_val-start_val)/(nb_categories+1)   for i in range(nb_consum_catg)  ]
        #consumption_coeff_vect = list(np.linspace(start_val+1*(max_val-start_val)/(nb_categories+1),max_val-1*(max_val-start_val)/(nb_categories+1),nb_consum_catg))
        #consumption_coeff_vect = list(range(1,10,nb_consum_catg))
        
        
        
        #### EQUAL WAGE ASSUMPTION
        firm_saving_coeff = 0.5
        
        equal_wages = (np.ones(nb_categories)-firm_saving_coeff)/nb_categories 
        
        wages_coeff_vect = equal_wages;
        ##wages_coeff_vect = [1, 0.8, 0.5, 0.3, 0.1]
        
        
        
        econ_MM = gen_econ_MM(wages_coeff_vect,consumption_coeff_vect)
        
        [eigVal,eigVect] = np.linalg.eig(econ_MM.T)
        indeces_of_sort = np.argsort(abs(eigVal))
        
        
        
        equilibrium_distribution = eigVect[:,indeces_of_sort[-1]] / sum(eigVect[:,indeces_of_sort[-1]] )
        second_largest_eigVal = eigVal[indeces_of_sort[-2]]
        
        

        equal_pop = np.floor(np.ones(nb_categories)*total_pop/nb_categories)

        population_per_class = equal_pop
        equilibrium_dist_perIndividual = equilibrium_distribution/ ([nb_firms]+list(population_per_class) )
        equilibrium_moneyOwned_perIndividual = equilibrium_dist_perIndividual*M_zero 
        
        
        Vpotential = [-1*math.log(i) for i in equilibrium_distribution]
        
        plt.figure(1)
        plt.title('Equilibrium distribution by consumption coeff')
        plt.xlim([-0.1,1])
        plt.plot([-0.1]+list(consumption_coeff_vect) ,equilibrium_distribution,'o-')
        
        plt.figure(2)
        plt.title('Potential by consumption coeff')
        plt.xlim([-0.1,1])
        plt.plot([-0.1]+list(consumption_coeff_vect) ,Vpotential,'o-')
        
        plt.figure(3)
        plt.title('Money owned at equilibrium by Class')
        plt.xlim([-0.1,1])
        plt.plot([-0.1]+list(consumption_coeff_vect) ,equilibrium_moneyOwned_perIndividual,'o-')
        
        keep_min = min(keep_min,min(equilibrium_moneyOwned_perIndividual))
        
        
        plt.figure(4)
        plt.title('log-Money owned at equilibrium by Class')
        plt.xlim([-0.1,1])
        log_eq_moneyOwned = [math.log(i) for i in equilibrium_moneyOwned_perIndividual]

        plt.plot([-0.1]+list(consumption_coeff_vect) ,log_eq_moneyOwned,'o-')

    plt.figure(3)
    
    plt.plot([-0.1]+list(consumption_coeff_vect),keep_min*np.ones(1+len(consumption_coeff_vect)))
    plt.show()
    print(keep_min)




def gen_econ_MM(wages_coeff_vect,consumption_coeff_vect):
    nb_categories = len(consumption_coeff_vect)

    TMP_mat = np.zeros((nb_categories+1,nb_categories+1))  # first one is firm
    TMP_mat[0,1:] = wages_coeff_vect 
    TMP_mat[1:,0] = consumption_coeff_vect
           
    ## Fill diagonals
    for i in range(nb_categories+1):   
        TMP_mat[i,i]  = 1 - sum(TMP_mat[i,:]) 
        if TMP_mat[i,i] < 0:
            print('!!WARNING!! DIAGONAL PROBABILITY IS NEGATIVE!!')
    return TMP_mat