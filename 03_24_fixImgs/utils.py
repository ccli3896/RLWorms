import pickle
import numpy as np
import pandas as pd 
from improc import *

# Interaction should be mainly with these functions

def make_df(fnames, 
    old_frame=None, 
    reward_ahead=10, 
    timestep_gap=1, 
    prev_act_window=3, 
    jump_limit=100,
    ):

    '''
    Takes a file and turns it into a trajectory dataframe.
    Can add to old data.
    Inputs:
                old_frame: old df
             reward_ahead: how many steps ahead to sum reward, for each table entry
             timestep_gap: how data are sampled (e.g. =5 means only every fifth datapoint is kept)
          prev_act_window: how many steps to look AHEAD to make sure all actions are 'on' or 'off' # naming is poor #TODO
               jump_limit: data are processed to remove faulty points where worm loc has jumped really far.
                           This is the maximum jump distance allowed before points are tossed.
                     disc: discretization of angles

    Output:
        dataframe object with keys:
            't', 'obs_b', 'obs_h', 'prev_actions', 'next_obs_b', 'next_obs_h', 'reward', 'loc'
    '''
    def add_ind_to_df(traj,df,i, reward_ahead, prev_act_window):
        # Assumes data for angle observations go from -1 to 1. 
        # This was because of use w/ normalized box wrapper with actual worm env
        ANG_BOUND = 180
        return df.append({
            't'           : traj['t'][i],
            'obs_b'       : int(traj['obs'][i][0]*ANG_BOUND),
            'obs_h'       : int(traj['obs'][i][1]*ANG_BOUND),
            'prev_actions': sum(traj['action'][i+1:i+prev_act_window+1]), # Note does not include current action
            'next_obs_b'  : int(traj['obs'][i+1][0]*ANG_BOUND),
            'next_obs_h'  : int(traj['obs'][i+1][1]*ANG_BOUND),
            'reward'      : sum(traj['reward'][i:i+reward_ahead]),
            'loc'         : traj['loc'][i],
        }, ignore_index=True)

    if old_frame is None:
        df = pd.DataFrame(columns = ['t', 
            'obs_b', 'obs_h', 'prev_actions', 
            'next_obs_b', 'next_obs_h', 'reward', 'loc'])
    else:
        df = old_frame

    # For every file, loop through and remove problem points.
    for fname in fnames:
        newf = True
        with open(fname, 'rb') as f:
            traj = pickle.load(f)

        for i in np.arange(0,len(traj['t'])-np.max([reward_ahead,prev_act_window+1]),timestep_gap):
            # For every timestep, check if the jump is reasonable and add to dataframe.
            if newf:
                if sum(traj['loc'][i])!=0:
                    df = add_ind_to_df(traj,df,i,reward_ahead,prev_act_window)
                    newf = False
            elif np.sqrt(np.sum(np.square(df['loc'].iloc[-1]-traj['loc'][i]))) < jump_limit:
                df = add_ind_to_df(traj,df,i,reward_ahead,prev_act_window)

    return df


def make_dist_dict2(df, sm_pars=None,
    prev_act_window=3,
    lp_frac=None):
    # This version doesn't smooth reward matrices. First subtracts them and then smooths the resulting matrix.
    # Stores difference in r_on and adds variances; sets r_off to 0's. 
    # Makes a dictionary of distributions using trajectory statistics.
    # sm_pars is a dict of form {'lambda': .05, 'iters': 30}
    #     If None, then no smoothing.
    # Lp_frac: [0,1]. Models find a number to subtract from light-on matrices, in order for
    #  this fraction of observations to remain above their corresponding light-off spots.

    traj_on = df.query('prev_actions=='+str(prev_act_window))
    traj_off = df.query('prev_actions==0')

    r_on, b_on, h_on, count_on = make_stat_mats(traj_on)
    r_off, b_off, h_off, count_off = make_stat_mats(traj_off)

    all_mats = [r_on,b_on,h_on,r_off,b_off,h_off]
    counts = [count_on,count_off]
    counts_lp = counts[0]+counts[1]

    for i in [1,2,4,5]:
        for j in range(2):
            if j==1:
                ang_par = False 
            else:
                ang_par = True
            all_mats[i][:,:,j] = lin_interp_mat(all_mats[i][:,:,j], ang_par)
            
            if sm_pars is not None:
                all_mats[i][:,:,j] = smoothen(all_mats[i][:,:,j], 
                                                counts[i//3], ang_par, 
                                                smooth_par=sm_pars['lambda'], iters=sm_pars['iters'])
                
    # Turns r_on and r_off matrices into their differences stored in r_on. (r_off is now set to 0)
    r_on[:,:,0] = r_on[:,:,0]-r_off[:,:,0]
    r_on[:,:,1] = r_on[:,:,1]+r_off[:,:,1]
    for i in range(2):
        r_on[:,:,i] = lin_interp_mat(r_on[:,:,i], False)
    if sm_pars is not None:
        r_on[:,:,0] = smoothen(r_on[:,:,0], counts_lp, False, smooth_par=sm_pars['lambda']*5, iters=sm_pars['iters'])
    r_off = np.zeros(r_off.shape)
    
    # This block is for light penalty implementation. Only applied to r_on.
    if lp_frac is None:
        light_penalty = 0
    else:
        r_diffs = r_on[:,:,0]-r_off[:,:,0]
        r_diffs_sorti = np.unravel_index(np.argsort(-r_diffs,axis=None), r_on[:,:,0].shape) # Subtract means and gets sorted indices.
        r_diffs_sorted = r_diffs[r_diffs_sorti]
        #counts_lp = np.ones((12,12)) ############
        count_lim = np.sum(counts_lp)*lp_frac
        cs = np.cumsum(counts_lp[r_diffs_sorti]) < count_lim
        cutoff_ind = [i for i,x in enumerate(cs) if not x][0]
        light_penalty = r_diffs_sorted[cutoff_ind]
        print(f'Penalty {light_penalty}')

    dist_dict = {
        'body_on': b_on,
        'body_off': b_off,
        'head_on': h_on,
        'head_off': h_off,
        'reward_on': r_on - light_penalty,
        'reward_off': r_off,
    }    
    return dist_dict, counts_lp

def add_to_traj(trajectory,info):
    # appends each key in info to the corresponding key in trajectory.
    # If trajectory is empty, returns trajectory as copy of info but with each element as list
    # so it can be appended to in the future.

    if trajectory:
        for k in info.keys():
            trajectory[k].append(info[k])
    else:
        for k in info.keys():
            trajectory[k] = [info[k]]

def make_stat_mats(df):
    # Inner func does most of the work querying for each obs.
    # Returns everything at once: 
    #   r_mat[12,12,2], b_mat[12,12,2], h_mat[12,12,2], counts[12,12].

    def get_stats_angs(df, obs):
        # Gets mean and var of df values that match obs, centered on obs
        # Returns r_stats, b_stats, h_stats, count. The first three are tuples [mu,var].

        df_d = dict(zip(df.columns,range(len(df.columns))))
        series = df.query('obs_b=='+str(obs[0])+'& obs_h=='+str(obs[1])).copy()
        series.iloc[:,df_d['next_obs_b']] = wrap_correct(series['next_obs_b'].to_numpy(),ref=series['obs_b'].to_numpy(),buffer=180)
        series.iloc[:,df_d['next_obs_h']] = wrap_correct(series['next_obs_h'].to_numpy(),ref=series['obs_h'].to_numpy(),buffer=180)

        # Handles case for one sample (initialize)
        r_sts,b_sts,h_sts,count = [np.nan,np.nan],[np.nan,np.nan],[np.nan,np.nan],0
        if series.size > 0:    
            r_sts[0],b_sts[0],h_sts[0] = wrap_correct(series['reward'].mean(),buffer=None), \
                                            wrap_correct(series['next_obs_b'].mean(),buffer=180), \
                                            wrap_correct(series['next_obs_h'].mean(),buffer=180)
            if series.size > 1:
                r_sts[1],b_sts[1],h_sts[1] = series['reward'].var(), \
                                                series['next_obs_b'].var(), \
                                                series['next_obs_h'].var()
        return r_sts,b_sts,h_sts,len(series)


    r_mat = np.zeros((12,12,2)) + np.nan 
    b_mat = np.zeros((12,12,2)) + np.nan 
    h_mat = np.zeros((12,12,2)) + np.nan 
    counts = np.zeros((12,12))

    for i,theta_b in enumerate(np.arange(-180,180,30)):
        for j,theta_h in enumerate(np.arange(-180,180,30)):
            r_sts,b_sts,h_sts,counts[i,j] = get_stats_angs(df,[theta_b,theta_h])
            r_mat[i,j,:] = r_sts
            b_mat[i,j,:] = b_sts
            h_mat[i,j,:] = h_sts

    return r_mat, b_mat, h_mat, counts

'''
Matrix regularizers: interpolation and smoothing
'''
def lin_interp_mat(mat,ang,wraparound=True):
    # Fills in NaNs in matrix by linear interpolation. 
    # ang is a boolean (True if data are for angles)
    # Only considers nearest neighbors (no diagonals).
    # Fills in NaNs from most neighbors to least neighbors.
    # wraparound extends matrix in all four directions. 
    if ang:
        buffer=180
    else:
        buffer=1e6

    mat = make_wraparound(mat,ang,wraparound=wraparound)

    # Find nans in relevant matrix section
    nan_inds = np.argwhere(np.isnan(mat[1:-1,1:-1])) + 1
        # add 1 because need index for extended matrix
    
    neighbor_lim = 3
    while nan_inds.size>0: 
        candidates = 0
        for ind in nan_inds:
            neighbors = get_neighbors(mat,ind)
            if sum(~np.isnan(neighbors)) >= neighbor_lim:
                mat[ind[0],ind[1]] = np.mean(wrap_correct(neighbors[~np.isnan(neighbors)], ref=min(neighbors), buffer=buffer))
                candidates+=1
        if candidates==0:
            neighbor_lim-=1
        nan_inds = np.argwhere(np.isnan(mat[1:-1,1:-1])) + 1

    return wrap_correct(mat[1:-1,1:-1],buffer=buffer)

def smoothen(matrix,counts,ang,smooth_par=.05,iters=30,wraparound=True,diagonals=True): 
    # matrix is in form [12,12]
    # counts is [12,12].
    # ang is bool, True if angle matrix
    # Will start with a simple weighted average between nearest neighbors and diagonals.
    #
    # Derivation for procedure in CL's nb page 81
    
    if np.array_equal(matrix,counts):
        count_version = True
    else:
        count_version = False
        
    if not np.all(counts):
        counts+=1
    # So the shapes start out right before looping 
    matrix = make_wraparound(matrix, ang, wraparound=True)
    counts = make_wraparound(counts, False, wraparound=True)
    if ang:
        buffer=180
    else:
        buffer=None
    
    for it in range(iters):
        matrix = make_wraparound(matrix[1:-1,1:-1], ang, wraparound=True)
        tempmat = np.copy(matrix) # Now tempmat and matrix are the same extended size
        rows,cols = np.array(matrix.shape)-2 

        # Loops through each matrix element and weights changes by counts
        for i in np.arange(rows)+1:
            for j in np.arange(cols)+1:
                neighs = wrap_correct(get_neighbors(matrix,(i,j)), ref=matrix[i,j], buffer=buffer)
                neigh_counts = get_neighbors(counts,(i,j))
                prod_nn = np.sum(np.multiply(neigh_counts, neighs))
                c_nn = np.sum(neigh_counts)
                
                if diagonals:
                    # Diagonal entries (scaled by 1/sqrt(2))
                    neighs_d = wrap_correct(get_diags(matrix,(i,j)), ref=matrix[i,j], buffer=buffer)
                    neigh_counts_d = get_diags(counts,(i,j))
                    prod_d = np.sum(np.multiply(neigh_counts_d, neighs_d))
                    c_d = np.sum(neigh_counts_d)
                    
                    mu = (prod_nn/c_nn + prod_d/(np.sqrt(2)*c_d)) / (1+1/np.sqrt(2))
                    alpha = 1/(1+(counts[i,j]*(1+1/np.sqrt(2)))/(c_nn/4+c_d/(4*np.sqrt(2))))
                    
                else:
                    mu = prod_nn/c_nn
                    alpha = 1/(1+(4*counts[i,j]/c_nn))
                    
                tempmat[i,j] = tempmat[i,j] + smooth_par*alpha*(mu - tempmat[i,j])
            
                
        # After tempmat is updated, set reference matrix to be the same
        # This way updates within one iteration don't get included in the same iteration
        matrix = np.copy(tempmat)
        if not count_version:
            counts = smoothen(counts,counts,False,smooth_par=smooth_par,iters=1)
    
    return wrap_correct(matrix[1:-1,1:-1], buffer=buffer)


'''
Small funcs and utils
'''
def wrap_correct(arr_orig,ref=0,buffer=180):
    # Takes angles and translates them to +/-buffer around ref.
    # For things like std, use large buffer so it doesn't change
    # If both arrays, send each element through this function.
    if buffer is None:
        return arr_orig
    
    if hasattr(arr_orig,"__len__"):
        arr = arr_orig.copy()
        if hasattr(ref,"__len__"):
            for i in range(len(arr)):
                arr[i] = wrap_correct(arr[i],ref=ref[i],buffer=buffer)
        # If only arr is an array
        else:
            arr[arr<ref-buffer]+=buffer*2
            arr[arr>=ref+buffer]-=buffer*2
            if len(arr[arr<ref-buffer])>0 or len(arr[arr>=ref+buffer])>0:
                arr = wrap_correct(arr,ref=ref,buffer=buffer)
    else:
        arr = arr_orig
        if arr<ref-buffer:
            arr+=buffer*2
            if arr<ref-buffer:
                arr = wrap_correct(arr,ref=ref,buffer=buffer)
        elif arr>=ref+buffer:
            arr-=buffer*2
            if arr>=ref+buffer:
                arr = wrap_correct(arr,ref=ref,buffer=buffer)
    return arr
        
def make_wraparound(mat,ang,wraparound=True):
    # Expands matrix for wraparound interpolation
    # If matrix is angle values, set ang=True.
    mat_new = np.zeros((np.array(mat.shape)+2)) + np.nan
    mat_new[1:-1,1:-1] = mat
    if ang:
        buffer=180
    else:
        buffer=None

    if wraparound:
        # diagonals
        mat_new[0,0] = wrap_correct(mat[-1,-1], ref=mat[0,0], buffer=buffer)
        mat_new[0,-1] = wrap_correct(mat[-1,0], ref=mat[0,-1], buffer=buffer)
        mat_new[-1,0] = wrap_correct(mat[0,-1], ref=mat[-1,0], buffer=buffer)
        mat_new[-1,-1] = wrap_correct(mat[0,0], ref=mat[-1,-1], buffer=buffer)
        # adjacents
        mat_new[0,1:-1] = wrap_correct(mat[-1,:], ref=mat[0,:], buffer=buffer)
        mat_new[-1,1:-1] = wrap_correct(mat[0,:], ref=mat[-1,:], buffer=buffer)
        mat_new[1:-1,0] = wrap_correct(mat[:,-1], ref=mat[:,0], buffer=buffer)
        mat_new[1:-1,-1] = wrap_correct(mat[:,0], ref=mat[:,-1], buffer=buffer)
    return mat_new

def get_neighbors(mat,i):
    # Makes array of four neighbors around mat[index]
    # index is a pair
    return np.array([mat[i[0],i[1]-1], mat[i[0],i[1]+1], mat[i[0]-1,i[1]], mat[i[0]+1,i[1]]])

def get_diags(mat,i):
    return np.array([mat[i[0]-1,i[1]-1], mat[i[0]-1,i[1]+1], mat[i[0]+1,i[1]-1], mat[i[0]+1,i[1]+1]])

def myround(x, base=30):
    return base * round(x/base)