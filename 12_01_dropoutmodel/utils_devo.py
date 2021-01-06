'''
Goals for this devo script:
Don't bin at all but do incorporate smoothing.

Worked out wraparound/smoothing combinations. Now for cleanup into utils.py in this folder.
'''

import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Interaction should be primarily with these functions 

def make_df(fnames, time_ahead=30, 
            xlimit=None,ylimit=None, 
            time_steps=10):
    # time_steps is how many steps ahead each sample is taken. E.g. 1 would mean a continuous sliding window
    # Makes a dataframe where each row is a data point containing 
    # - time
    # - current observation
    # - action
    # - next observation
    # - sum of rewards from current : current + time_ahead

    def filter_by_loc(traj,xlimit=1e6,ylimit=0):
        # Quick script to remove wrong points
        locs = np.array(traj['loc'])
        loc = []
        rew = []
        obs = []
        act = []
        t = []
        for i in range(locs.shape[0]):
            if locs[i,0]!=0 and locs[i,1]!=0:
                if locs[i,0]<xlimit and locs[i,1]>ylimit:
                    loc.append(locs[i,:])
                    rew.append(traj['reward'][i])
                    obs.append(traj['obs'][i])
                    t.append(traj['t'][i])
                    act.append(traj['action'][i])
        return np.array(loc),np.array(rew),(np.array(obs)*180).astype(int),np.array(act),np.array(t)
    
    nfiles = len(fnames)
    if xlimit is None:
        xlimit = np.zeros(nfiles) + 1e6
    if ylimit is None:
        ylimit = np.zeros(nfiles)
    
    df = pd.DataFrame(columns=['obs_b','obs_h','action','next_obs_b','next_obs_h','reward','loc'])
    
    # Loops through all points and adds to dataframe
    for f_i,fname in enumerate(fnames):
        with open(fname,'rb') as f:
            traj = pickle.load(f)
        loc,rew,obs,act,t_all = filter_by_loc(traj,xlimit=xlimit[f_i],ylimit=ylimit[f_i])

        for i in np.arange(3, len(t_all)-time_ahead, time_steps):
            df = df.append({
                't':t_all[i],
                'obs_b':obs[i][0],
                'obs_h':obs[i][1],
                'action':act[i],
                'prev_actions':sum(act[i-3:i]), # THIS WAS ADDED TO TRY TO FIND CONTINUOUS ACTIONS
                'next_obs_b':obs[i+1][0],
                'next_obs_h':obs[i+1][1],
                'reward':sum(rew[i+1:i+1+time_ahead]),
                'loc':loc[i]
                }, ignore_index=True)

    return df

def make_dist_dict(traj,smoothenpars=None):
    # Makes a dictionary of distributions using trajectory statistics.
    # smoothenpars is a dictionary of form {smooth_par: .05, iters: 30}. If none, then no smoothing.

    def interp2(mat, wraparound=True):
        # Returns a matrix that's been through the linear interpolation function.
        # [12,12,2] with dimensions [body,head,mu/sig] 
        
        m_ints = np.zeros(mat.shape)
        for i in range(2):
            m_ints[:,:,i] = lin_interp_mat(mat[:,:,i],wraparound=wraparound)
        return np.squeeze(m_ints)

    traj_on = traj.query('action==1 & prev_actions==3')
    traj_off = traj.query('action==0 & prev_actions==0')

    r_on_mat, r_on_counts = make_stat_mats(traj_on,'reward')
    b_on_mat, b_on_counts = make_stat_mats(traj_on,'next_obs_b')
    h_on_mat, h_on_counts = make_stat_mats(traj_on,'next_obs_h')
    r_on_mat = interp2(r_on_mat)
    b_on_mat = interp2(b_on_mat)
    h_on_mat = interp2(h_on_mat)
    
    r_off_mat, r_off_counts = make_stat_mats(traj_off,'reward')
    b_off_mat, b_off_counts = make_stat_mats(traj_off,'next_obs_b')
    h_off_mat, h_off_counts = make_stat_mats(traj_off,'next_obs_h')
    r_off_mat = interp2(r_off_mat)
    b_off_mat = interp2(b_off_mat)
    h_off_mat = interp2(h_off_mat)

    # Smoothing function block
    if smoothenpars is None:
        # Set some defaults
        smoothenpars = {'smooth_par': 0.01, 'iters': 0}
    mats = [r_on_mat, r_off_mat]
    mat_counts = [r_on_counts, r_off_counts]
    # Smooths all matrices
    for i in range(len(mats)):
        for j in range(2):
            if len(mats[i].shape)<3:
                mats[i] = np.expand_dims(mats[i],axis=1)
                mat_counts[i] = np.expand_dims(mat_counts[i],axis=1)
            mats[i][:,:,j] = smoothen(mats[i][:,:,j], mat_counts[i],
                                        smooth_par=smoothenpars['smooth_par'],
                                        iters=smoothenpars['iters'])
        
    dist_dict = {
        'body_on': b_on_mat,
        'body_off': b_off_mat,
        'head_on': h_on_mat,
        'head_off': h_off_mat,
        'reward_on': r_on_mat,
        'reward_off': r_off_mat,
    }

    return dist_dict


# End of interaction functions

'''
Utilities used in make_dist_dict()
'''

def get_neighbors(mat,i):
    # Makes array of four neighbors around mat[index]
    # index is a pair
    return np.array([mat[i[0],i[1]-1], mat[i[0],i[1]+1], mat[i[0]-1,i[1]], mat[i[0]+1,i[1]]])

def get_diags(mat,i):
    return np.array([mat[i[0]-1,i[1]-1], mat[i[0]-1,i[1]+1], mat[i[0]+1,i[1]-1], mat[i[0]+1,i[1]+1]])

def make_wraparound(mat,wraparound=False):
    # Expands matrix for wraparound interpolation
    mat_new = np.zeros((np.array(mat.shape)+2)) + np.nan
    mat_new[1:-1,1:-1] = mat

    if wraparound:
        # diagonals
        mat_new[0,0] = wrap_correct(mat[-1,-1], ref=mat[0,0])
        mat_new[0,-1] = wrap_correct(mat[-1,0], ref=mat[0,-1])
        mat_new[-1,0] = wrap_correct(mat[0,-1], ref=mat[-1,0])
        mat_new[-1,-1] = wrap_correct(mat[0,0], ref=mat[-1,-1])
        # adjacents
        mat_new[0,1:-1] = wrap_correct(mat[-1,:], ref=mat[0,:])
        mat_new[-1,1:-1] = wrap_correct(mat[0,:], ref=mat[-1,:])
        mat_new[1:-1,0] = wrap_correct(mat[:,-1], ref=mat[:,0])
        mat_new[1:-1,-1] = wrap_correct(mat[:,0], ref=mat[:,-1])
    return mat_new

def lin_interp_mat(mat, wraparound=False):
    # Fills in NaNs in matrix by linear interpolation. 
    # Only considers nearest neighbors (no diagonals).
    # Fills in NaNs from most neighbors to least neighbors.
    # wraparound extends matrix in all four directions. Haven't really gotten this to work with edge cases yet.

    def set_range(mat):
        mat[mat<-180] += 360
        mat[mat>=180] -= 360
        return mat

    mat = make_wraparound(mat, wraparound=wraparound)

    # Find nans in relevant matrix section
    nan_inds = np.argwhere(np.isnan(mat[1:-1,1:-1])) + 1
        # add 1 because need index for extended matrix
    
    neighbor_lim = 3
    while nan_inds.size>0:
        candidates = 0
        for ind in nan_inds:
            neighbors = get_neighbors(mat,ind)
            if sum(~np.isnan(neighbors)) >= neighbor_lim:
                mat[ind[0],ind[1]] = np.mean(wrap_correct(neighbors[~np.isnan(neighbors)], ref=min(neighbors)))
                candidates+=1
        if candidates==0:
            neighbor_lim-=1
        nan_inds = np.argwhere(np.isnan(mat[1:-1,1:-1])) + 1

    return set_range(mat[1:-1,1:-1])

def make_stat_mats(traj,newkey):

    def get_stats_angs(df,obs,newkey):
        # gets mean and std for the newkey df values that match obs in oldkey, centered on obs.
        # As in, series will first be translated to [obs-180,obs+180].
        # Keeping the convention of keep the floor, remove the ceiling when rounding.
        
        # Remove points where HT orientation switched
        backwards = obs[0]-180
        if backwards < -180:
            backwards += 360
        
        series = df.query('obs_b=='+str(obs[0])+'& obs_h=='+str(obs[1])+
                        '& next_obs_b!='+str(backwards))[newkey].to_numpy()
        
        if newkey=='next_obs_h':
            series[series<obs[1]-180] += 360
            series[series>=obs[1]+180] -= 360
        elif newkey=='next_obs_b':
            series[series<obs[0]-180] += 360
            series[series>=obs[0]+180] -= 360     
        
        # if there was only one sample, make up a distribution anyway.
        if series.size == 0:
            sermean,sererr = np.nan,np.nan
        else:
            if np.std(series)==0:
                sererr = np.nan # Leaving it up to interpolation
            else:
                sererr = np.std(series) #/ np.sqrt(series.size)

            sermean = np.mean(series)
            if sermean<-180:
                sermean += 360
            elif sermean>=180:
                sermean -= 360
                
        return sermean,sererr,series.size

    stat_mats = np.zeros((12,12,2)) + np.nan 
    count_mat = np.zeros((12,12))
    for i,theta_b in enumerate(np.arange(-180,180,30)):
        for j,theta_h in enumerate(np.arange(-180,180,30)):
            sermean,sererr,count_mat[i,j] = get_stats_angs(traj,[theta_b,theta_h],newkey)
            stat_mats[i,j,:] = sermean,sererr
    return stat_mats, count_mat

def wrap_correct(arr,ref=0):
    # Takes an array of angles and translates to +/-180 around ref.
    # ref should stay zero for del(body angle). It should be the previous angle otherwise.
    if hasattr(arr,"__len__"):
        if hasattr(ref,"__len__"):
            for i in range(len(arr)):
                arr[i] = wrap_correct(arr[i],ref[i])
        else:
            arr[arr<ref-180] += 360
            arr[arr>ref+180] -= 360
    else: 
        if arr<ref-180:
            arr+=360
        elif arr>ref+180:
            arr-=360
            
    return arr

def smoothen(matrix,counts,smooth_par=.05,iters=30,wraparound=True,diagonals=True): 
    # For the reward matrices. 
    # matrix is in form [12,12]
    # counts is [12,12].
    # Will start with a simple linear weighting/smoothing. 
    
    # So the shapes start out right before looping 
    matrix = make_wraparound(matrix, wraparound=True)
    counts = make_wraparound(counts, wraparound=True)
    
    for it in range(iters):
        matrix = make_wraparound(matrix[1:-1,1:-1], wraparound=True)
        tempmat = np.copy(matrix) # Now tempmat and matrix are the same extended size
        rows,cols = np.array(matrix.shape)-2 

        # Loops through each matrix element and weights changes by counts
        for i in np.arange(rows)+1:
            for j in np.arange(cols)+1:
                neighs = np.append(get_neighbors(matrix,(i,j)), matrix[i,j])
                neigh_counts = np.append(get_neighbors(counts,(i,j)), counts[i,j])
                del_sm = np.sum(np.multiply(neigh_counts, neighs))
                if diagonals:
                    # Diagonal entries (scaled by 1/sqrt(2))
                    neighs_d = np.append(get_diags(matrix,(i,j)), matrix[i,j])
                    neighs_counts_d = np.append(get_diags(counts,(i,j)), counts[i,j])
                    del_sm_d = (np.sum(np.multiply(neighs_counts_d, neighs_d)))/np.sqrt(2)
                    Z = np.sum(neigh_counts) + np.sum(neighs_counts_d)/np.sqrt(2)
                else:
                    del_sm_d = 0
                    Z = np.sum(neigh_counts)

                tempmat[i,j] = tempmat[i,j] + smooth_par*(del_sm/Z+del_sm_d/Z - tempmat[i,j])
                
        # After tempmat is updated, set reference matrix to be the same
        # This way updates within one iteration don't get included in the same iteration
        matrix = np.copy(tempmat)
    
    return matrix[1:-1,1:-1]

'''
End of make_dist_dict() utils
'''