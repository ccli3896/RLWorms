'''
Some utilities that I'm not sure where to put otherwise
'''

import numpy as np
import torch
from torch.utils.data import Dataset

def worm_bound(a):
    if hasattr(a,'__len__'):
        a = np.array(a)
        a = np.where(a<-180, a+360, a)
        a = np.where(a>=180, a-360, a)
        return a

    else:
        if a<-180: return a+360
        elif a>=180: return a-360
        else: return a

def ang_between(a,b):
	# Both angles must be worm-bounded in [-180,180)
    diff = np.abs(a-b)
    if diff > 180:
        diff = np.abs(diff-360)
    return diff

def projection(body_angle, mvt_angle, speed):
	# Gets speed in direction of the body angle.
	ang = ang_between(body_angle,mvt_angle)
	return speed*np.cos(ang*np.pi/180)

def class_error(guesses, true_value):
    # Returns specificity (true positive/total yes) and sensitivity (true negative/total no).
    guesses = guesses.astype(bool)
    true_value = true_value.astype(bool)
    TP = sum(np.bitwise_and(guesses,true_value))
    allyes = np.sum(true_value)
    TN = sum(~np.bitwise_or(guesses, true_value))
    allno = np.sum(~true_value)

    specificity = TP/allyes
    sensitivity = TN/allno

    return specificity, sensitivity

class MemoryDataset(Dataset):
  '''
  Makes an input [state,action] and output [next state] pair for each memory.
  '''
  def __init__(self, memories):
    self.memories = memories

  def __len__(self):
    return len(self.memories)

  def __getitem__(self,idx):
    if torch.is_tensor(idx):
      idx = idx.tolist()

    state_action = torch.tensor(np.append(self.memories[idx][0],self.memories[idx][1])).type(torch.float)
    next_state = torch.tensor(self.memories[idx][-1]).type(torch.float)

    return (state_action, next_state)