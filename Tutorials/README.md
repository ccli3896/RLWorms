# Tutorials

This is copied from repo README. 

# Installation guide
A conda environment with the required dependencies for agent training can be built from `./Tutorials/rlwormsdemo_environment.yml`. The time required is the time to install Pytorch and its dependencies.

Here we list two tutorials that can be run from the repo alone (without hardware installation or animals).
## Processing images
Clone the repo and run the following to process a set of 500 images (at 3 fps, so 2.78 minutes) into tracks. 

      python ./Tutorials/check_data_tutorial.py

This sample dataset is from an evaluation episode with trained agents on line 301. Images are stored in `./Tutorials/Worm image data/imgs/`. Track data is a dictionary saved in a `.pkl` file. Two JPG images are saved: one colored by time and the other colored by agent actions. Actions are read from `./Tutorials/Worm image data/actions2021-11-02_13-40-29.pkl`. Raw image datasets are very large so have not been uploaded in their entirety. They are available upon request. 

## Agent training and visualization
Clone the repo and run the following to train two agents on the same dataset, collected from animals with channelrhodopsin expressed in AWC(ON):
      
      python ./Tutorials/train_agents_main.py 0 301
      python ./Tutorials/train_agents_main.py 1 301

This will create a new folder named 'models' that contains saved soft actor-critic agents for line 301. The first input is a label number (ID) for the agent and the second is the training line. The second input can be modified for the desired line listed in the `Training data/` folder. You can try any of the following lines corresponding to the genotype in Table 1 of the manuscript:



| Line | Effect      | Expression                                      | Code Label |
|------|-------------|-------------------------------------------------|------------|
| 1    | Excitatory  | AIY                                             | 281        |
| 2    | Excitatory  | AWC(ON), [ASI]*                                 | 301        |
| 3    | Inhibitory  | SIA; SIB; RIC; AVA; RMD; AIY; AVK; BAG          | 352        |
| 4    | Inhibitory  | All neurons                                     | 446        |
| 5    | Excitatory  | Cholinergic ventral cord motor neurons          | 336        |
| 6    | Excitatory  | IL1; PQR                                        | 437        |
| WT   | N/A         | None                                            | 73         |
* Weak or unstable expression, see manuscript

On a 2020 MacBook Pro with an Apple M1 chip running Ventura 13.4.1, it takes roughly 40 min to train one agent for twenty epochs.

For agents used in the paper, we trained ensembles of 20 independent agents for at least 20 epochs on each line. Ensemble policies were visually inspected to determine stopping criteria: if the policy was 1. non-trivial; i.e. not an "always on" or "always off" policy, and 2. symmetric about the origin (as it should be, given random translations and rotations of data during training) then training was stopped and the ensemble used for evaluation episodes. 

One can visually inspect trained agent policies. To see an agent's policy at epoch 19, for instance, run:

        python ./Tutorials/see_policy.py 301 0 19
        python ./Tutorials/see_policy.py 301 1 19

To see multiple agents' policies, enter IDs either comma-separated or with a hyphen. E.g.

        python ./Tutorials/see_policy.py 301 0-1 19
OR

        python ./Tutorials/see_policy.py 301 0,1 19

These commands will save JPG images of the policy for an "ensemble" of two agents as a demo, averaging their policies to form the ensemble.