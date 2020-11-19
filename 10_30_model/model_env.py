import pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def make_df(traj, time_ahead=30, 
            xlimit=1e6,ylimit=0, 
            time_steps=10):
    # time_steps is how many steps ahead each sample is taken. E.g. 1 would mean a continuous sliding window
    # Makes a dataframe where each row is a data point containing 
    # - time
    # - current observation
    # - action
    # - next observation
    # - sum of rewards from current : current + time_ahead

    def filter_by_loc(traj,xlimit=1430,ylimit=0):
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
    
    loc,rew,obs,act,t_all = filter_by_loc(traj,xlimit=xlimit,ylimit=ylimit)
    df = pd.DataFrame(columns=['obs_b','obs_h','action','next_obs_b','next_obs_h','reward','loc'])
    
    # Loops through all points and adds to dataframe
    for i in np.arange(0, len(t_all)-time_ahead, time_steps):
        df = df.append({
            't':t_all[i],
            'obs_b':obs[i][0],
            'obs_h':obs[i][1],
            'action':act[i],
            'next_obs_b':obs[i+1][0],
            'next_obs_h':obs[i+1][1],
            'reward':sum(rew[i+1:i+1+time_ahead]),
            'loc':loc[i]
            }, ignore_index=True)

    return df

def lin_interp_mat(mat, wraparound=False):
    # Fills in NaNs in matrix by linear interpolation. 
    # Only considers nearest neighbors (no diagonals).
    # Fills in NaNs from most neighbors to least neighbors.
    # wraparound extends matrix in all four directions. Haven't really gotten this to work with edge cases yet.

    def set_range(mat):
        mat[mat<-180] += 360
        mat[mat>=180] -= 360
        return mat

    def get_neighbors(mat,i):
        # Makes array of four neighbors around mat[index]
        # index is a pair
        return np.array([mat[i[0],i[1]-1], mat[i[0],i[1]+1], mat[i[0]-1,i[1]], mat[i[0]+1,i[1]]])

    def make_wraparound(mat,wraparound=False):
        # Expands matrix for wraparound interpolation
        
        mat_new = np.zeros((np.array(mat.shape)+2)) + np.nan
        mat_new[1:-1,1:-1] = mat

        if wraparound:
            mat_new[0,1:-1] = mat[-1,:] 
            mat_new[-1,1:-1] = mat[0,:] 
            mat_new[1:-1,0] = mat[:,-1] 
            mat_new[1:-1,-1] = mat[:,0] 
        return mat_new

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

def make_stat_mats(traj,newkey)
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
            sermean,stderr = np.nan,np.nan
        else:
            if np.std(series)==0:
                stderr = np.nan # Leaving it up to interpolation
            else:
                stderr = np.std(series) #/np.sqrt(series.size)

            sermean = np.mean(series)
            if sermean<-180:
                sermean += 360
            elif sermean>=180:
                sermean -= 360
                
        return sermean,stderr

    stat_mats = np.zeros((12,12,2)) + np.nan 
    for i,theta_b in enumerate(np.arange(-180,180,30)):
        for j,theta_h in enumerate(np.arange(-180,180,30)):
            stat_mats[i,j,:] = get_stats_angs(traj,[theta_b,theta_h],newkey)
    return stat_mats

def make_dist_dict(traj):
    # Makes a dictionary of distributions using trajectory statistics.
    def interp2(mat, wraparound=False):
        # Returns a matrix that's been through the linear interpolation function.
        # [12,12,2] with dimensions [body,head,mu/sig] 
        
        m_ints = np.zeros(mat.shape)
        for i in range(2):
            m_ints[:,:,i] = lin_interp_mat(mat[:,:,i],wraparound=wraparound)
        return np.squeeze(m_ints)

    traj_on = traj.query('action==1')
    traj_off = traj.query('action==0')

    for i,theta_b in enumerate(np.arange(-180,180,30)):
        for j,theta_h in enumerate(np.arange(-180,180,30)):
            r_on_mat = make_stat_mats(traj_on,'reward')
            b_on_mat = make_stat_mats(traj_on,'next_obs_b')
            h_on_mat = make_stat_mats(traj_on,'next_obs_h')
            
            r_off_mat = make_stat_mats(traj_off,'reward')
            b_off_mat = make_stat_mats(traj_off,'next_obs_b')
            h_off_mat = make_stat_mats(traj_off,'next_obs_h')
        
    dist_dict = {
        'body_on': interp2(b_on_mat),
        'body_off': interp2(b_off_mat),
        'head_on': interp2(h_on_mat),
        'head_off': interp2(h_off_mat),
        'reward_on': interp2(r_on_mat),
        'reward_off': interp2(r_off_mat),
    }