import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import math as m
import xraydb as xr

from matplotlib.collections import PolyCollection
from matplotlib.pyplot import cm
from IPython.display import display, Latex
from itertools import combinations

plt.rcParams['font.family'] = 'Times New Roman'


def allowed_test(hkl):
    '''Checks if reflections from crystal indices are allowed.
    For fcc diamond cubic crystal structure.
    '''
    flag = False;
    h, k, l = hkl;
    if (h%2) == 1 and (k%2) == 1 and (l%2) == 1:
        flag = True;
    elif (h%2) == 0 and (k%2) == 0 and (l%2) == 0 and ((h+k+l)%4) == 0:
        flag = True;  
    if [h, k, l] == [0, 0, 0]:
        flag = False;
    return(flag)

def get_allHKL(index_max):
    '''Generates list of [h,k,l] objects with a maximum
    h,k, or l given my index_max
    '''
    hkl_list = []
    for i in range(-index_max,index_max+1):
        h = -i
        for j in range(-index_max,index_max+1):
            k = -j
            for l in range(-index_max,index_max+1):
                if allowed_test([h, k, -l]):
                    hkl_list.append(np.array([h, k, -l]))
    return(hkl_list)

def make_thetaB_list(target_energy, hkl_list, crystal_type):
    '''Generates a list of Bragg angles for associated
    crystals in hkl_list at a specified energy.
    '''
    theta_bragg = []
    for i in range(len(hkl_list)):
        if crystal_type == 'Si':
            a = 2*5.4298/np.dot(hkl_list[i],hkl_list[i])**0.5
        elif crystal_type == 'Ge':
            a = 2*5.658/np.dot(hkl_list[i],hkl_list[i])**0.5
        else:
            print('Crystal type not supported. Please enter Si or Ge.')
            exit(1)
        b = 12398.42/a
        if -1 < b/target_energy < 1: 
            angle = 180*np.arcsin(b/target_energy)/np.pi
        else:
            angle = -1.0
        theta_bragg.append(angle)
    return(theta_bragg)

def energy_filter_list(index_max, target_energy, energy_range, 
              bragg_range, crystal_type, hkl_order = False):
    '''Generates a list of [h,k,l] objects that will reach
    the specified energy within the ranges given.
    hkl_order is True if you don't want to include objects 
    that are just reorderings of h,k, and l.
    '''
    hkl_list = get_allHKL(index_max)
    bragg = make_thetaB_list(target_energy, hkl_list, crystal_type)
    low_energy_ang = make_thetaB_list(target_energy - energy_range[0]/1000, 
                               hkl_list, crystal_type)
    high_energy_ang = make_thetaB_list(target_energy + energy_range[1]/1000, 
                                hkl_list, crystal_type)
    poss_hkl = []
    for i in range(len(hkl_list)):
        if (bragg[i]!=-1.0 and bragg[i]>bragg_range[0] and 
            low_energy_ang[i]<bragg_range[1] and high_energy_ang[i]>bragg_range[0]):
            poss_hkl.append(hkl_list[i])
        else:
            pass
    hkls = []
    if hkl_order is True:
        for i in poss_hkl:
            temp = []
            if (abs(i[0]) >= abs(i[1]) and abs(i[0]) >= abs(i[2]) 
                and abs(i[1]) >= abs(i[2])):
                hkls.append(i)
        return(hkls)
    else:
        return(poss_hkl)

def Ry(theta):
    '''Generates a matrix to rotate theta degrees around
    the y-axis.
    '''
    return(np.matrix([[m.cos(theta), 0, m.sin(theta)],
                   [0, 1, 0],
                   [-m.sin(theta), 0, m.cos(theta)]]))
 
def Rz(theta):
    '''Generates a matrix to rotate theta degrees around
    the z-axis.
    '''
    return(np.matrix([[m.cos(theta), -m.sin(theta), 0],
                   [m.sin(theta), m.cos(theta), 0],
                   [0, 0, 1]]))

def get_angle(vect_1, vect_2):
    '''Finds the angle between two vectors. Output in radians.'''
    if np.dot(vect_1, vect_1) > 0.1:
        if np.dot(vect_2, vect_2) > 0.1:
            denom = (np.dot(vect_1, vect_1)*np.dot(vect_2, vect_2))**0.5;
        else:
            denom = 100000000.;
    else:
        denom = 100000000.;
    return(np.arccos(np.dot(vect_1, vect_2)/denom))

def get_angle_2(vect_1,vect_2):
    '''Finds the angle between two vecotors but with handedness.
    Output in radians.
    '''
    denom = (np.dot(vect_1, vect_1)*np.dot(vect_2, vect_2))**0.5
    dot = np.dot(vect_1, vect_2)
    ratio = dot/denom
    ang = np.arccos(ratio.round(10))
    if np.cross(vect_1,vect_2)<0:
        angle = ang
    else:
        angle = -ang 
    return(angle)

def get_Rz(vector):
    '''Get rotation from the x-axis around z-axis for inputted vector.'''
    return get_angle(np.array([1, 0, 0]), vector*np.array([1,1,0]))

def get_Ry(vector):
    '''Get rotation from the z-axis around y-axis for inputted vector.'''
    return get_angle(np.array([0, 0, 1]), vector*np.array([1,0,1]))

def matmul(vector, matx):
    '''Multiplies a vector and matrix and returns a vector.'''
    temp = np.matmul(vector, matx);
    temp_v = [temp[0,0], temp[0,1], temp[0,2]];
    return(temp_v)

def get_rot_matrix(vect):
    '''Finds the matrix that rotates a given vector to the z-axis.
    '''
    matrix_z = Rz(get_Rz(vect));
    temp = np.matmul(vect, matrix_z);
    temp_v = [temp[0,0], temp[0,1], temp[0,2]];
    matrix_y = Ry(get_Ry(temp_v));
    rot_matrix = np.matmul(matrix_z, matrix_y);
    return(rot_matrix)

def alpha_list(crystal_hkl, hkl_list):
    '''Generates a list of alpha angles given a crystal_hkl
    for a list of [hkl] reciprocal lattice vectors.
    '''
    alpha = []
    for i in hkl_list:
        angle = 180*get_angle(i,crystal_hkl)/np.pi
        alpha.append(angle)
    return(alpha)

def make_phi_list(hkl_list, crystal_hkl, phi0_hkl):
    '''Generate a list of phi angles for a given crystal 
    taking phi0_hkl to be 0 degrees.
    '''
    phi_list = [];
    rotmat = get_rot_matrix(crystal_hkl);
    xy_phi0_vect = matmul(phi0_hkl, rotmat)[0:2];
    for i in hkl_list:
        xy_hkl = matmul(i, rotmat)[0:2];
        phi_angle = get_angle_2(xy_phi0_vect, xy_hkl);
        phi_list.append(phi_angle*180/np.pi)
    return(phi_list)

def hkl_harmonic_pass(hkl):
    '''Tests if a given [h,k,l] object is a harmonic.'''
    flag = True;
    h, k, l = hkl;
    for harm in range(2, 20):
        if (h%harm)==0 and (k%harm)==0 and (l%harm)==0:
            hsub = h / harm;
            ksub = k / harm;
            lsub = l / harm;
            flag = flag and not(allowed_test([hsub, ksub, lsub]));
    return(flag)

def get_allHKL_no_harmonics(index_max):
    '''Generates a list of allowed [hkl] objects excluding harmonics.'''
    all_hkl = []
    for i in range(-index_max, index_max+1):
        h = i
        for j in range(-index_max, index_max+1):
            k = j
            for l in range(-index_max, index_max+1):
                if allowed_test([h, k, l]) and hkl_harmonic_pass([h, k, l]):
                    all_hkl.append([h, k, l])
    return(all_hkl)

def remove_reordering(hkl_list):
    '''Removes any [hkl] that are just reordering, 
    ex. [khl], [klh], or [hlk]'''
    no_reordering = []
    check = []
    for i in hkl_list:
        test = sorted(i, reverse = True)
        if test in check:
            pass
        else:
            check.append(test)
            no_reordering.append(i)
    return(no_reordering)



class Reflection:
    
    def __init__(self, hkl, comp):
        self.hkl = hkl
        self.comp = comp
    
    def allowed_test(self):
        '''Check if the reflection is allowed.'''
        flag = False;
        h, k, l = self.hkl;
        if (h%2)==1 and (k%2)==1 and (l%2)==1:
            flag = True;
        elif (h%2)==0 and (k%2)==0 and (l%2)==0 and ((h+k+l)%4)==0:
            flag = True;  
        if [h, k, l] == [0, 0, 0]:
            flag = False;
        return(flag)
    
    def theta_bragg(self, energy):
        '''Find the bragg angle for the reflection 
        to reach the inputted energy.
        '''
        if self.comp == 'Si':
            a = 2*5.4298/np.dot(self.hkl, self.hkl)**0.5
        elif self.comp == 'Ge':
            a = 2*5.658/np.dot(self.hkl, self.hkl)**0.5
        else:
            print('Crystal type isn\'t supported. Please enter Si or Ge.')
        b = 12398.42/a
        if -1 < b/energy < 1: 
            angle = 180*np.arcsin(b/energy)/np.pi
        else:
            angle = -1.0
        return(angle)
   
    def alpha(self, crystal_hkl):
        '''Find the alpha angle for inputted crystal to reach the reflection'''
        return(180*get_angle(self.hkl, crystal_hkl)/np.pi)

    def phi(self, crystal_hkl, phi0_hkl):
        '''Find the phi angle of the reflection for a given crystal with 
        a given phi0 reflection.
        '''
        rotation_matrix = get_rot_matrix(crystal_hkl)
        phi0_vector = matmul(phi0_hkl, rotation_matrix)[0:2]
        hkl_vector = matmul(self.hkl, rotation_matrix)[0:2]
        return(get_angle_2(phi0_vector, hkl_vector)*180/np.pi)
    
    def bragg_filter(self, energy, energy_range, bragg_range):
        '''Return True or False if the bragg angle for the reflection
        to reach an energy (and an energy window around it) is within
        a specified range.
        '''
        bragg_angle = self.theta_bragg(energy)
        low_limit = self.theta_bragg(energy - energy_range[0])
        high_limit = self.theta_bragg(energy - energy_range[1])
        if (bragg_angle != -1.0 and bragg_angle > bragg_range[0] 
            and low_limit < bragg_range[1] and high_limit > bragg_range[0]):
            return(True)
        else:
            return(False)
    
    def mech_filter(self, energy, crystal_hkl, mech_angle_range):
        '''Return True or False if the mechanical angle (bragg+alpha) for 
        the reflection is within a specified range.
        '''
        mech_angle = self.theta_bragg(energy) + self.alpha(crystal_hkl)
        if mech_angle_range[0] < mech_angle < mech_angle_range[1]:
            return(True)
        else:
            return(False)
    
    def get_energy(self, angle):
        '''Find the Bragg angle the reflection will be at when reaching
        the inputted energy.
        '''
        if self.comp == 'Si':
            a = 2*5.4298/np.dot(self.hkl, self.hkl)**0.5
        elif self.comp == 'Ge':
            a = 2*5.658/np.dot(self.hkl, self.hkl)**0.5
        else:
            print('Crystal type isn\'t supported. Please enter Si or Ge.')
        b = 12398.42/a
        energy = b/np.sin(angle / 180 * np.pi)
        return(energy)
    
    
def hkl_selection(crystal_hkl, index_max, mech_angle_range, 
                  bragg_angle_range, energy, energy_range, 
                  crystal_type, save_to_excel, file_path,
                  file_name):
    flag = True
    for i in energy:
        energy_str = str(i)
        hkl_list = get_allHKL(index_max)
        hkl_noreordered_list = remove_reordering(hkl_list)
        hkl_filtered_list = []
        for j in hkl_noreordered_list:
            possible_hkl = Reflection(j, crystal_type)
            if possible_hkl.bragg_filter(
                    i, energy_range, 
                    bragg_angle_range) == True:
                hkl_filtered_list.append(possible_hkl)
            else:
                pass
        print('-----------------------------------------------------')
        table = []
        for k in crystal_hkl:
            g0 = crystal_type + ' ['+str(k[0][0])+' '+str(k[0][1])+' '+str(k[0][2])+']'
            phi0 = '['+str(k[1][0])+' '+str(k[1][1])+' '+str(k[1][2])+']'
            for l in hkl_filtered_list:
                if l.mech_filter(i, k[0], mech_angle_range) == True:
                    h_index = str(l.hkl[0])
                    k_index = str(l.hkl[1])
                    l_index = str(l.hkl[2])
                    indices = '['+h_index+' '+k_index+' '+l_index+']'
                    bragg_angle = l.theta_bragg(i)
                    alpha = l.alpha(k[0])
                    mechanical_angle = bragg_angle + alpha
                    phi = l.phi(k[0], k[1])
                    table.append([g0, indices, bragg_angle, alpha, 
                                  mechanical_angle, phi0, phi])
        if len(table) == 0:
            print('There are no options with specified conditions for ' + energy_str + 'eV')
            continue
        table_sorted = sorted(table, key = lambda x: (x[2], -x[3]), reverse = True)
        df = pd.DataFrame(table_sorted)
        df.columns = ['$G_{0}$','$G_{hkl}$',r'$\theta_B$ (deg)',r'$\alpha$ (deg)', 
                      '$\theta_M$ (deg)', '$G_{\phi = 0}$', '$\phi$ (deg)']
        crystal_hkl_str = str(k[0][0])+str(k[0][1])+str(k[0][2])
        phi0_str = str(k[1][0])+str(k[1][1])+str(k[1][2])
        display(Latex('Table for target energy of ' + energy_str + ' eV'))
        display(df.round(2))
        df.columns = ['G_0', 'G_hkl', 'Bragg (deg)', 'alpha (deg)', 'Mech (deg)', 'G_(phi=0)', 'phi (deg)']
        if save_to_excel == True:
            file = file_path + file_name + '.xlsx'
            if flag == True:
                writer = pd.ExcelWriter(file, engine = 'xlsxwriter')
                flag = False
            else:
                writer = pd.ExcelWriter(file, engine = 'openpyxl', mode = 'a')
            df.to_excel(writer, sheet_name = 'Table ' + energy_str + 'eV')
            writer.close()
    print('-----------------------------------------------------')
    
def sbca_selection(refl_max_index, crystal_max_index, crystal_angle_range, 
                       bragg_range, target_energy, energy_range, crystal_type, 
                       save_to_excel, file_path, file_name, return_dict=False):
    flag = True
    df_dict = {}
    for e in target_energy:
        best_hkl = []
        energy_str = str(e)
        if return_dict == False:
            print('---------------------------------------------')
        poss_hkls = energy_filter_list(refl_max_index, e, energy_range, bragg_range, crystal_type, True);
        hkls = []
        check = []
        for i in poss_hkls:
            a = [abs(i[0]), abs(i[1]), abs(i[2])]
            if a in check:
                pass
            else:
                check.append(a)
                hkls.append(i)
        poss_xtal_hkls = get_allHKL_no_harmonics(crystal_max_index);
        xtal_hkls = []
        for i in poss_xtal_hkls:
            if abs(i[0]) >= abs(i[1]) and abs(i[0]) >= abs(i[2]) and abs(i[1]) >= abs(i[2]):
                xtal_hkls.append(i)
        for i in hkls:
            if crystal_type == 'Si':
                a = 2*5.4298/np.dot(i,i)**0.5
            elif crystal_type == 'Ge':
                a = 2*5.658/np.dot(i,i)**0.5
            else:
                print('Crystal type not supported. Please enter Si or Ge.')
                exit(1)
            b = 12398.42/a
            if -1 < b/(e) < 1: 
                bragg = 180*np.arcsin(b/(e))/np.pi
            else:
                bragg = -1.0
            ghkl = [int(i[0]), int(i[1]), int(i[2])]
            for j in xtal_hkls:
                g0 = [int(j[0]), int(j[1]), int(j[2])]
                alpha = 180*get_angle(i,j)/np.pi
                if crystal_angle_range[0]<(bragg+alpha)<crystal_angle_range[1]:
                    best_hkl.append([energy_str, ghkl, bragg, g0, alpha, bragg+alpha])
                else:
                    pass
        if len(best_hkl) == 0:
            print('There are no options with specified conditions for ' + energy_str + 'eV')
            continue
        best_hkl_sorted = sorted(best_hkl, key=lambda x: (x[0], -x[2], x[1], x[4]), reverse = False)
        df = pd.DataFrame(best_hkl_sorted)
        df.columns = ['E (eV)', '$G_{hkl}$', r'$\theta_B$ (deg)', '$G_{0}$', r'$\alpha$ (deg)', '$\theta_M$ (deg)']
        G_hkl = str(i[0])+str(i[1])+str(i[2])
        thetaB = str(bragg.round(2))
        if return_dict == False:
            display(Latex('Table for target energy of ' + energy_str + ' eV'))
            display(df.round(2))
        df.columns = ['E (eV)', 'G_hkl', 'Bragg (deg)', 'G_0', 'alpha (deg)', 'Mech (deg)']
        df_dict[str(e)] = df
        if save_to_excel == True:
            file = file_path + file_name + '.xlsx'
            if flag == True:
                writer = pd.ExcelWriter(file, engine = 'xlsxwriter')
                flag = False
            else:
                writer = pd.ExcelWriter(file, engine = 'openpyxl', mode = 'a')
            df.to_excel(writer, sheet_name =  energy_str + 'eV')
            writer.close()
    if return_dict == False:
        print('---------------------------------------------')
    
    if return_dict == True:
        return df_dict
    
def optimal_sbcas(hkl_max_index, sbca_max_index, mechanical_angle_range, 
                  bragg_angle_range, target_energy, energy_range, crystal_type, max_N):
    # Generate data frame of both Si and Ge crystal
    Si_table = sbca_selection(hkl_max_index, sbca_max_index, mechanical_angle_range, bragg_angle_range,
                              target_energy, energy_range, 'Si', False,'', '', return_dict=True)
    for key in Si_table:
        Si_table[key]['G_hkl'] = [['Si'] + Ghkl for Ghkl in Si_table[key]['G_hkl']]
        Si_table[key]['G_0'] = [['Si'] + G0 for G0 in Si_table[key]['G_0']]
        
    Ge_table = sbca_selection(hkl_max_index, sbca_max_index, mechanical_angle_range, bragg_angle_range,
                              target_energy, energy_range, 'Ge', False,'', '', return_dict=True)
    for key in Ge_table:
        Ge_table[key]['G_hkl'] = [['Ge'] + Ghkl for Ghkl in Ge_table[key]['G_hkl']]
        Ge_table[key]['G_0'] = [['Ge'] + G0 for G0 in Ge_table[key]['G_0']]

    table = {}
    if crystal_type == 'Si_n_Ge':
        table = Si_table
        for key in Ge_table.keys():
            if key in table.keys():
                table[key] = pd.concat([table[key], Ge_table[key]], ignore_index=True)
            else:
                table[key] = Ge_table[key]
    elif crystal_type == 'Si':
        table = Si_table
    elif crystal_type == 'Ge':
        table = Ge_table
    else:
        print('Invalid crystal type entered!')
        
    # Calculate angle distance
    for key in table.keys():
        table[key]['ang_dist'] = np.sqrt(np.square(85-table[key]['Bragg (deg)']) + np.square(table[key]['alpha (deg)']))
        
    # make a table loging what energy each G0 can do
    all_G0 = []
    for key in table:
        all_G0.extend(table[key]['G_0'])

    unique_G0 = set(map(tuple, all_G0))
    unique_G0 = [list(lst) for lst in unique_G0]
    df = pd.DataFrame()
    df['G_0'] = unique_G0

    for key in table:
        FoundG0 = []
        for G0 in df['G_0']:
            if G0 in list(table[key]['G_0']):
                FoundG0.append(1)
            else:
                FoundG0.append(0)
        df[str(key)] = FoundG0
    
    # adding up angle dsitance for each G0
    ave_Rs = []
    for G0_ind, target_G0 in enumerate(df['G_0']):
        sum_R = 0
        for key in table:
            # for a given metal, find the ind in the original list where the reflection uses a certain G0
            indices = []
            for i, G0 in enumerate(list(table[key]['G_0'])):
                if G0 == target_G0:
                    indices.append(i)
            if indices != []:
                # summing up the minimun R for each metal for a given G0
                sum_R = sum_R + min([table[key]['ang_dist'][i] for i in indices])
        ave_R = sum_R/np.sum(df.iloc[G0_ind, 1:])
        ave_Rs.append(ave_R)
    df['ave R'] = ave_Rs
    
    # find the optimal set of SBCA that spans the given energies
    if max_N > len(target_energy):
        print('Maximun number of SBCAs entered is larger than the number of energies selected.')
    else:
        numbers = list(range(len(df['G_0'])))
        result = []
        for N in list(range(max_N, 0, -1)):
            print('---------------------------------------------')
            display(Latex('Optimizing: Number of SBCA = '+ str(N) + '...'))
            combinations_list = list(combinations(numbers, N))
            #loop over each combo to see if it covers all metals
            best_R = 1000
            best_combo = np.empty((1, N))
            for combo in combinations_list:
                flag = np.zeros(df.shape[1]-2)
                ave_R = 0
                for i in combo:
                    flag = flag + df.iloc[i, 1:-1]
                    ave_R = ave_R + df.iloc[i, -1]
                if min(flag)==0:
                    continue
                else:
                    ave_R = ave_R/N
                    if ave_R < best_R:
                        best_R = ave_R
                        best_combo = combo
            display(Latex('When using '+str(N)+' SBCAs:'+'\n'))
            G0s = []
            for G0 in [df['G_0'][i] for i in best_combo]:
                G0s.append(G0[0]+'['+str(G0[1])+', '+str(G0[2])+', '+str(G0[3])+']')
            display(Latex('G0 = '+', '.join(G0s)))
            display(Latex('Average Angle Distance = '+str(round(best_R, 2))))
                    
            result.append([N, best_combo, best_R])
    
    return None
    
def bar_chart(crystal_hkl, crystal_type, index_max, min_thetaB_list, 
              max_alpha, max_bragg, energy_range, elements, 
              fig_dim, vert_guide, save_file, fig_path, fig_name, fig_dpi, fig_format):
    
    num_lists = int(len(min_thetaB_list))
    reflections = [[] for i in range(num_lists)]
    hkl_list = get_allHKL(index_max)
    hkl_noreordered_list = remove_reordering(hkl_list)
    
    for i in range(len(min_thetaB_list)):
        
        list_hkl_refl = []
        
        for j in hkl_noreordered_list:          
            possible_hkl = Reflection(j, crystal_type)
            alpha = possible_hkl.alpha(crystal_hkl)
            e_low = possible_hkl.get_energy(max_bragg)
            e_high = possible_hkl.get_energy(min_thetaB_list[i])
            if (alpha < max_alpha
                and energy_range[0] < e_low < energy_range[1]):
                list_hkl_refl.append([j, alpha, e_low, e_high, '['+str(j[0])+' '+str(j[1])+' '+str(j[2])+']'])
            else:
                pass
        sort_list_hkl_refl = sorted(list_hkl_refl, key=lambda x: x[2])
        reflections[i] = [k for k in sort_list_hkl_refl]
        
    colors = iter(cm.rainbow(np.linspace(1, 0.1, len(min_thetaB_list))))

    bars = []
    lab = []
    
    for i in min_thetaB_list:
        if min_thetaB_list.index(i) < len(min_thetaB_list)-1:
            string = str(i)+'-'+str(min_thetaB_list[min_thetaB_list.index(i)+1])
            lab.append(string)
        else:
            string = str(i)+'-'+str(max_bragg)
            lab.append(string)

    for i in range(len(min_thetaB_list)):

        verts = []

        for j in range(len(reflections[i])):
            v = [(reflections[i][j][2]/1000,j+0.4),
                 (reflections[i][j][2]/1000,j-0.4),
                 (reflections[i][j][3]/1000,j-0.4),
                 (reflections[i][j][3]/1000,j+0.4)]
            verts.append(v)

        bars.append(PolyCollection(verts, color = next(colors), label = lab[i]))
        
    labels = []
    for i in range(len(reflections[0])):
        temp = reflections[0][i][4]
        labels.append(temp)
        
    lab_alpha = []
    for i in range(len(reflections[0])):
        temp = str(reflections[0][i][1].round(2))
        lab_alpha.append(temp)
    
    fig, ax = plt.subplots()

    for i in range(len(min_thetaB_list)):
        ax.add_collection(bars[i])
        ax.autoscale()

    plt.legend(bbox_to_anchor = (0, 1), loc = 'upper left', fontsize = 20, title_fontsize = 15, title = "Minimum\nBragg Angle")

    emission = []

    if elements == None:
        pass
    else:
        for i in elements:
            temp_emis = xr.xray_lines(i)
            ky = list(temp_emis.keys())
            for j in ky:
                em = temp_emis[j][0]
                if energy_range[0] < em < int(reflections[0][-1][3])+100:
                    emission.append(em/1000)
                else: 
                    pass

    for i in emission:
        plt.axvline(x = i, linewidth = 1, color = 'grey', zorder=0)
    
    if vert_guide == None or []:
        pass
    else:
        for i in vert_guide:
            plt.axvline(x = i/1000, linewidth = 1, color = 'red', zorder=0)
    
    ax.set_yticks(range(len(reflections[0])))
    ax.set_yticklabels(labels, fontsize = 20)
    ax.tick_params(axis='y', which='major', width = 3, length = 5)
    
    ax2 = ax.twinx()
    ax2.set_ylabel(r'$\alpha$ (deg)', fontsize = 30)
    ax2.set_ylim(ax.get_ylim())
    ax2.set_yticks(range(len(reflections[0])))
    ax2.set_yticklabels(lab_alpha, fontsize = 20)
    ax2.tick_params(axis='y', which='major', width = 3, length = 5)
    
    ax.set_xticks(np.arange(0,30,step = 1))
    ax.set_xticks(np.arange(0.5,29.5,step = 1), minor = True)
    ax.set_xlabel('Energy (keV)', fontsize = 30)
    ax.tick_params(axis='x', which='major', width = 3, length = 5, direction = "in", labelsize=25)
    ax.tick_params(axis='x', which='minor', width = 2, length = 3, direction = "in")
    
    a = str(crystal_hkl[0])
    b = str(crystal_hkl[1])
    c = str(crystal_hkl[2])
    crystal = crystal_type + '(' + a + b + c +')'
    
    fig.set_size_inches(fig_dim[0], fig_dim[1])

    plt.xlim([energy_range[0]/1000, int(reflections[0][-1][3])/1000+0.1]) 
    plt.title('Asymmetric ' + crystal_type + ' ('+str(crystal_hkl[0])
              +str(crystal_hkl[1])+str(crystal_hkl[2])+')', fontsize = 40)
    if save_file == True:
        file = fig_path + fig_name
        plt.savefig(file, dpi = fig_dpi, format = fig_format)