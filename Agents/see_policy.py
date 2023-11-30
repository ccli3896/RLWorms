import numpy as np
import matplotlib.pyplot as plt
import SAC as sac
import torch
import pickle
from torch import nn
from copy import deepcopy
import argparse

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def check_ens_xy(actor, loc, speed=0, frames=15, disc=36, shift_relative=True):
    '''Takes an actor and returns a matrix of the probability of light ON at the given location.
    Axes are y: body angle relative to target
             x: head angle relative to body angle
    '''

    # Set up axes: from -180 to 180 in steps of 10 (len 36)
    bodies = np.arange(-180,180,10)
    heads = np.arange(-180,180,10)

    choices = np.zeros((disc,disc))
    standard = np.zeros((disc,disc))

    # Shift body angles so it's relative to target
    if shift_relative:
        shift = np.arctan2(loc[1],-loc[0]) *180/np.pi
        if shift<0: shift+=360 # Runs from 0 to 360 now
        shift_ind = int(shift/360 * disc) # This gives how many indices to move to the right.

        bodies = np.roll(bodies, shift_ind)
    

    # Make a batch of states consisting of all the body/head angle combinations
    # Convert to radians    
    states = []
    for b_i, b in enumerate(bodies):
        for h_i, h in enumerate(heads):
            b_ang = b*np.pi/180
            h_ang = h*np.pi/180
            # Make head angle wrap around to stay within [-pi, pi]
            h_ang += b_ang
            if h_ang < -np.pi: 
                h_ang += 2*np.pi
            elif h_ang > np.pi:
                h_ang -= 2*np.pi

            # Makes an input state that is the same angles and location repeated for all frames
            state = torch.repeat_interleave(torch.tensor([np.sin(b_ang),np.cos(b_ang),np.sin(h_ang),np.cos(h_ang),*loc]),frames)
            states.append((state).view(1,-1).type(torch.float))
  
    states = torch.cat(states, dim=0)

    # Send batch of states through actor and return probabilities
    return actor(states)


def load_ens(line_path, agent_list, suffix):
  frames = 15
  nrns = 64

  polnet = sac.DiscretePolicy(frames*6, 2, 64, dropout=0.0)
  actors = [sac.DiscretePolicy(frames*6,2,64,dropout=0.0) for i in range(len(agent_list))]

  for i,i_agent in enumerate(agent_list):
    actor_path = f'{line_path}_s{i_agent}_{suffix}_.zip'
    actors[i].load_state_dict(torch.load(actor_path, map_location=torch.device('cpu')))
  return actors


def main():
  parser = argparse.ArgumentParser(description='Visualize agent policies.')
  parser.add_argument('line', type=int, help='Animal line')
  parser.add_argument('agent_id', type=str, help='Trained agent ID')
    # Can input one int for one agent, comma-separated lists of agents (e.g. '1,2,3'), or hyphen-separated numbers (e.g. '3-5', inclusive)
  parser.add_argument('epoch', type=int, help='Training epoch to inspect')

  args = parser.parse_args()


  # Make a parser for agent list; converts input into string of agent ID numbers
  if ',' in args.agent_id:
    agent_list = args.agent_id.split(',')
    agent_list = [int(x) for x in agent_list]
  elif '-' in args.agent_id:
    agent_list = args.agent_id.split('-')
    agent_list = np.arange(int(agent_list[0]), int(agent_list[1])+1)
  else:
    agent_list = [int(args.agent_id)]

  # Load agent(s)
  ens = load_ens(f'./Agents_last four lines/Agents {args.line}/sac_actor_L{args.line}', agent_list, f'e{args.epoch}')

  # Set up image
  fig,ax = plt.subplots(9,9)
  fig.set_size_inches((20,20))

  # Get policy for each (x,y) location around the target. Roughly normalized to plate size
  pols_array = []
  for x_i,x in enumerate(np.linspace(-1,1,9)):
      for y_i,y in enumerate(np.linspace(-1,1,9)):
          print(x_i,x,y_i,y)
          pols = []
          for i in range(len(agent_list)):

              pol = check_ens_xy(ens[i], (x,y))
              pol = pol[1].detach().numpy()
              pols.append(pol[:,1])

          pols = np.array(pols)

          pols_array.append(np.mean(np.vstack(pols), axis=0).reshape(36,36))
          im = ax[x_i,y_i].imshow(pols_array[-1], vmin=.45,vmax=.55,cmap='magma')

  fig.subplots_adjust(right=0.85)
  cbar_ax = fig.add_axes([0.9, 0.15, 0.02, 0.7])
  fig.colorbar(im, cax=cbar_ax)

  plt.savefig(f'./Policy_L{args.line}_{args.agent_id}.jpg')
  plt.show()

if __name__=='__main__':
  main()
