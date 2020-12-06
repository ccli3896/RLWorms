'''
DEVO CHANGES:
Fix issue: interpolates between variances instead of standard deviations.
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

def make_dist_dict(traj,bin_z=3):
    # Makes a dictionary of distributions using trajectory statistics.
    def interp2(mat, wraparound=True):
        # Returns a matrix that's been through the linear interpolation function.
        # [12,12,2] with dimensions [body,head,mu/sig^2] 
        
        m_ints = np.zeros(mat.shape)
        for i in range(2):
            m_ints[:,:,i] = lin_interp_mat(mat[:,:,i],wraparound=wraparound)
        return np.squeeze(m_ints)

    traj_on = traj.query('action==1 & prev_actions==3')
    traj_off = traj.query('action==0 & prev_actions==0')

    r_on_mat = make_stat_mats(traj_on,'reward')
    b_on_mat,bin_on = make_obs_mat(traj_on,'b',bin_z=bin_z)
    h_on_mat,bin_on = make_obs_mat(traj_on,'h',bin_z=bin_z)

    r_off_mat = make_stat_mats(traj_off,'reward')
    b_off_mat,bin_off = make_obs_mat(traj_off,'b',bin_z=bin_z)
    h_off_mat,bin_off = make_obs_mat(traj_off,'h',bin_z=bin_z)
        
    dist_dict = {
        'obs_on_bins': bin_on,
        'obs_off_bins': bin_off,
        'body_on': b_on_mat,
        'body_off': b_off_mat,
        'head_on': h_on_mat,
        'head_off': h_off_mat,
        'reward_on': interp2(r_on_mat),
        'reward_off': interp2(r_off_mat),
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
        mat_new[0,0] = mat[-1,-1]
        mat_new[0,-1] = mat[-1,0]
        mat_new[-1,0] = mat[0,-1]
        mat_new[-1,-1] = mat[0,0]
        # adjacents
        mat_new[0,1:-1] = mat[-1,:] 
        mat_new[-1,1:-1] = mat[0,:] 
        mat_new[1:-1,0] = mat[:,-1] 
        mat_new[1:-1,-1] = mat[:,0] 
    return mat_new

def lin_interp_mat(mat, wraparound=True): 
    # Fills in NaNs in matrix by linear interpolation. 
    # Only considers nearest neighbors (no diagonals).
    # Fills in NaNs from most neighbors to least neighbors.
    # wraparound extends matrix in all four directions. Haven't really gotten this to work with edge cases yet.

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
                mat[ind[0],ind[1]] = np.mean(neighbors[~np.isnan(neighbors)])
                candidates+=1
        if candidates==0:
            neighbor_lim-=1
        nan_inds = np.argwhere(np.isnan(mat[1:-1,1:-1])) + 1

    return mat[1:-1,1:-1]

def make_stat_mats(traj,newkey):

    def get_stats_angs(df,obs,newkey):
        # gets mean and var for the newkey df values that match obs in oldkey, centered on obs.
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
            sermean,servar = np.nan,np.nan
        else:
            if np.var(series)==0:
                servar = np.nan # Leaving it up to interpolation
            else:
                servar = np.var(series) #/ np.sqrt(series.size)

            sermean = np.mean(series)
            if sermean<-180:
                sermean += 360
            elif sermean>=180:
                sermean -= 360
                
        return sermean,servar

    stat_mats = np.zeros((12,12,2)) + np.nan 
    for i,theta_b in enumerate(np.arange(-180,180,30)):
        for j,theta_h in enumerate(np.arange(-180,180,30)):
            stat_mats[i,j,:] = get_stats_angs(traj,[theta_b,theta_h],newkey)
    return stat_mats

'''
End of make_dist_dict() utils
'''

def wrap_correct(arr,ref=0):
    # Takes an array of angles and translates to +/-180 around ref.
    # ref should stay zero for del(body angle). It should be the previous angle otherwise.
    if isinstance(arr,(int,float)):
        if arr<ref-180:
            arr+=360
        elif arr>ref+180:
            arr-=360
    else:
        arr[arr<ref-180] += 360
        arr[arr>ref+180] -= 360
    return arr

def make_obs_mat(traj, ang_key, bin_z=3):
    # Makes an array [theta_h,2] where second axis is mu/sig. Standard deviation.
    # Result is mean for CHANGE in body angle, which is why it's not an axis in the output array.
    # ang_key is 'b' for body angle, 'h' for head angle. 
    
    # BINNING: keeps one vector of indices and one vector of values. 
    # Index vector will be as long as original no. of bins. 
    
    def get_stats_obs_b(traj, theta_h):
        # Returns mean and var for a given previous head angle. 
        # Returns array[mean, var], # of points
        # Bins all past body angles together.
        
        # Make change in body angle array
        # Remove points where HT orientation switched
        series = traj.query('obs_h=='+str(theta_h))
        series_del = series['obs_b'].to_numpy() - series['next_obs_b'].to_numpy()
        series_del = wrap_correct(series_del[np.abs(series_del)!=180])
        
        if len(series_del)>1:
            safe_var = np.var(series_del, ddof=1) # ddof=1?
            safe_mu = np.mean(series_del)
        else:
            if len(series_del)==0:
                safe_mu = 0
            else:
                safe_mu = np.mean(series_del)
            safe_var = 0
        return np.array([safe_mu, safe_var]), len(series_del)
    
    def get_stats_obs_h(traj, theta_h):
        # Returns mean and var for a given previous head angle. 
        # Returns array[mean, var], # of points
        # Bins all past body angles together.
        
        # Make change in body angle array
        # Remove points where HT orientation switched
        series = traj.query('obs_h=='+str(theta_h))
        series_del = series['obs_b'].to_numpy() - series['next_obs_b'].to_numpy()
        series_h = series['next_obs_h'].to_numpy()
        series_h = wrap_correct(series_h[np.abs(series_del)!=180], theta_h)
        
        if len(series_del)>1:
            safe_var = np.var(series_h, ddof=1) 
            safe_mu = np.mean(series_h)
        else:
            if len(series_h)==0:
                safe_mu = 0
            else:
                safe_mu = np.mean(series_h)
            safe_var = 0
        
        return np.array([safe_mu,safe_var]), len(series_h)
    
    def join_bins(to_join, bin_inds, stat_mat_binned, counts):
        # Returns modified bin_inds, stat_mat_binned, and counts where to_join[1] has been joined to
        # to_join[0]. 
        # Replaces to_join[1] entries in stat_mat_binned and counts with nans. 
        
        def check_bins(b_inds):
            # Makes sure bins point directly to bin index
            check_vec = []
            for i in range(len(b_inds)):
                if b_inds[i]!=i:
                    check_vec.append(i)
            for ch in check_vec:
                if b_inds[b_inds[ch]] != b_inds[ch]:
                    b_inds[ch] = b_inds[b_inds[ch]]
            return b_inds
        
        n0,n1 = counts[to_join[0]], counts[to_join[1]]
        mu0,s0 = stat_mat_binned[to_join[0],:]
        mu1,s1 = stat_mat_binned[to_join[1],:]
        mu1 = wrap_correct(mu1,ref=mu0)
        
        mu_new = (n0*mu0 + n1*mu1)/(n0+n1)
        s_new = np.sqrt(((n0-1)*s0**2 + (n1-1)*s1**2)/(n0+n1-1)  +  (n0*n1*(mu0-mu1)**2)/((n0+n1)*(n0+n1-1)))
        stat_mat_binned[bin_inds[to_join[1]],:] = np.zeros(2)+np.nan
        stat_mat_binned[bin_inds[to_join[0]],:] = np.array([mu_new, s_new])
        
        counts[to_join[0]] += counts[to_join[1]]
        counts[to_join[1]] = np.nan
        
        # Check that every map goes directly to the bin index
        bin_inds[to_join[1]] = bin_inds[to_join[0]]
        bin_inds = check_bins(bin_inds)                
        
        return bin_inds, stat_mat_binned, counts
        
    def get_bin_neighbors(bin_inds,b):
        # Finds the indices of neighbors of the bth bin
        neighs = [b-1,b+1]
        while bin_inds[neighs[0]]==bin_inds[b]:
            neighs[0]-=1
        while bin_inds[neighs[1]]==bin_inds[b]:
            neighs[1]+=1
            if neighs[1]>=len(bin_inds):
                neighs[1] -= len(bin_inds)
        return neighs
    
    
    stat_mat = np.zeros((12,2)) + np.nan
    count = np.zeros(12)
    for i, theta_h in enumerate(np.arange(-180,180,30)):
        if ang_key == 'b':
            stat_mat[i,:], count[i] = get_stats_obs_b(traj, theta_h)
        elif ang_key == 'h':
            stat_mat[i,:], count[i] = get_stats_obs_h(traj, theta_h)
        else:
            raise KeyError('Invalid angle key')
        
    # Binning process. Joins if one bin is more than bin_z standard deviations below average.
    num_angs = 12
    bin_inds = np.arange(12)
    stat_mat_binned = stat_mat
    mu_count,std_count = np.mean(count),np.std(count)
    z_counts = (count-mu_count)/std_count
    
    while len(bin_inds[z_counts<-bin_z])>0:
        for b in bin_inds[z_counts<-bin_z]:
            if np.isnan(count[b]):
                continue

            # Otherwise, find minimum zscore from nearby bins
            neighs = get_bin_neighbors(bin_inds,b)
            less_neigh = neighs[np.argmin([z_counts[bin_inds[neighs[0]]], z_counts[bin_inds[neighs[1]]]])]
            to_join = [b, less_neigh]
            bin_inds, stat_mat_binned, count = join_bins(to_join, bin_inds, stat_mat_binned, count)
            z_counts[less_neigh] = 0
            z_counts = (count-mu_count)/std_count

    return stat_mat_binned, bin_inds