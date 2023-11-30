A conda environment with the required dependencies for agent training can be built from `./Training scripts/rlworms-env.yml`. The time required is the time to install Pytorch and its dependencies.

# Demo
Clone the repo and run the following to train two agents on the same dataset:
      
      python ./Tutorials/train_agents_main.py 0 301
      python ./Tutorials/train_agents_main.py 1 301

This will create a new folder named 'models' that contains saved soft actor-critic agents for line 301. The first input is a label number (ID) for the agent and the second is the training line. The second input can be modified for the desired line listed in the `Training data/` folder. You can try any of the following lines corresponding to the genotype in Table 1 of the manuscript:



| Line | Expression                                      | Code label|
|------|--------------------------------------------------|----------------------------------------|
| 1    | AIY                                              | 281                                    |
| 2    | AWC(ON), [ASI]*                                  | 301                                    |
| 3    | SIA; SIB; RIC; AVA; RMD; AIY; AVK; BAG           | 352                                    |
| 4    | All neurons                                      | 446                                    |
| 5    | Cholinergic ventral cord motor neurons           | 336                                    |
| 6    | IL1; PQR                                         | 437                                    |
| WT   | None                                            | 73 |


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

# Instructions for use
Scripts that interact with the animals are in the `Evaluation scripts/` folder.

1. We used `collect.py` to collect training data by running the command:

        python collect.py 20 [camera_id] --randomrate=0.1 --lightrate=3

      where the first input is the number of minutes of data collected for that session. Data are saved as images. We had two rigs so camera ID was always 1 or 2.

2. To process image data, `check_data.py` was executed on the output folder from `collect.py` or `eval.py`.

        python check_data.py [folder label] [number of images in folder] [comma-separated (x,y) coordinates of target] [camera_id]

      This script saves images of the animal tracks and a `.pkl` file containing animal coordinates on the plate, head angle, and body angle. These files can be compiled with other tracks to train agents as in the demo.

3. To run evaluation episodes as in Figures 2-3, 5-6, one can run `eval.py` after agents have been trained.

        python eval.py [camera_id] [animal line number] [target coordinates] --eptime=600

   The `--eptime` input is in seconds and denotes length of the evaluation episode.
