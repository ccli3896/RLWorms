# RLWorms
This repository contains code and data used to set up the nematode C elegans as an RL environment. To accompany paper [Improving animal behaviors through a neural interface with deep reinforcement learning](https://www.biorxiv.org/content/10.1101/2022.09.19.508590v2.article-metrics) by Li, Kreiman, and Ramanathan (2023). 

## Introduction
Guiding or improving animal behavior directly through the nervous system has been a common goal for neuroscience and robotics researchers alike [1-3]. Previous works in brain interfaces and animal robotics have attempted to use direct interventions to affect behavior on a variety of tasks, relying on manual specification for stimulation frequencies, locations, dynamics, and patterns [4–21]. A central difficulty with these approaches is that manual tuning has limited applicability, as it relies on knowledge of the neural circuits or mechanisms involved. For direct neural stimulation, effective patterns can vary depending on which neurons are targeted and on the animal itself [22,23], and thus, even though technologies for precise neuronal modulation exist [24,25], there still lies the challenge of how to design and choose an algorithm that can systematically and automatically learn strategies to activate a set of neurons to improve a particular behavior [26–30]. 

<img width="300" alt="Screenshot 2023-12-13 at 9 34 25 PM" src="https://github.com/ccli3896/RLWorms/assets/33879208/ddd465d3-4136-446c-9beb-4a38ca4842bc"> <img width="400" alt="Screenshot 2023-12-13 at 9 48 23 PM" src="https://github.com/ccli3896/RLWorms/assets/33879208/8934d13d-9449-4192-bf9d-c1324d39afe6">



Here we addressed this challenge using deep reinforcement learning (RL). In an RL setting, an agent collects rewards through interactions with its environment. We present a flexible framework that can, given only a reward signal, observations, and a set of relevant actions, learn different ways of achieving a goal behavior that adapt to the chosen interface. We tested our ideas on the nematode C. elegans, interfacing an RL agent with its nervous system using optogenetic tools [24,27]. This animal has small and accessible nervous system and yet still possesses a rich behavioral repertoire [46]. In a natural setting, C. elegans must navigate variable environments to avoid danger or find targets like food. Therefore, we aimed to build an RL agent that could learn how to interface with neurons to assist C. elegans in target-finding and food search. We tested the agent by connecting it to different sets of neurons with distinct roles in behavior. The agents could not only couple with different sets of neurons to perform a target-finding task, but could also generalize the task to improve food search across novel environments in a zero-shot fashion, that is, without any prior training. 

# Contents of repository
            1. Agents: Trained agents used in evaluations, separated by genetic line.
            2. Animal scripts: Code used to interact with animals, including collecting training data and evaluating agents on live animals.
            3. Basic evaluation data: Agents were trained and tested on each of 6 optogenetically modified animal lines, labeled as in the table below (Agent Training and Visualization). Control and experimental tracks; Figures 2-4.
            4. Cross evaluation data: Animal track data used in Figure 5.
            5. Foodsearch data: Tracking data used in Figure 6B-G.
            6. Obstacle data: Tracking data used in Figure 6I-K.
            7. Training data: Datasets used to train agents. Concatenations of 20 min episodes of randomly flashing light data, with animals switched out at the end of every episode. See Figure 1 and Methods in manuscript for details.
            8. Training scripts: Code to train soft actor-critic agents on animal data.
            9. Tutorials: Folder containing tutorials that do not require animals.

### Video demos
In the two videos below, the blue frame represents agent decisions and the red circle represents a virtual target. Videos are 8x speed and animals are roughly 1 mm in length. Plates are 4 cm in diameter.

#### Random actions
https://github.com/ccli3896/RLWorms/assets/33879208/bc5858d8-7dbe-4766-a7da-616ecec953f1

#### After training
https://github.com/ccli3896/RLWorms/assets/33879208/5d229fe1-37bb-485c-b822-879f566f575a

# System requirements
Training can be completed on any machine with Pytorch (tested on torch==2.0.1). The demo has been tested on a 2020 MacBook Pro with an Apple M1 chip running Ventura 13.4.1. For the manuscript, the computations were run on the FASRC Cannon cluster supported by the FAS Division of Science Research Computing Group at Harvard University. GPU types available to us are in [this list.](https://docs.rc.fas.harvard.edu/kb/running-jobs/#Using_GPUs)
Training completed in under an hour with these resources and a memory pool of 10gb for all cores during an array of training jobs of 20-30 agents.

For the hardware setup in the manuscript (Figure 1), we used an Edmund Optics 5012 LE Monochrome USB 3.0 camera or a ThorLabs DCC1545M with [pypyueye](https://github.com/galaunay/pypyueye.git). 
Lights for optogenetic illumination were Kessil PR160L LEDs at wavelengths of 467 nm for blue and 525 nm for green. LEDs were controlled by National Instruments DAQmx devices with the [nidaqmx library](https://nidaqmx-python.readthedocs.io/en/latest/).

Due to hardware compatibility issues, data collection and evaluation on live animals must be completed on a Windows machine (all live animal data collected using Windows 10 and 11).

# Installation guide
A conda environment with the required dependencies for agent training can be built from `./Tutorials/rlwormsdemo_environment.yml`. The time required is the time to install Pytorch and its dependencies.

# Tutorials
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

# Instructions for use with animals
Scripts that interact with the animals are in the `Evaluation scripts/` folder. See manuscript for experimental setup and details.

1. We used `collect.py` to collect training data by running the command:

        python collect.py 20 [camera_id] --randomrate=0.1 --lightrate=3

      where the first input is the number of minutes of data collected for that session. Data are saved as images. We had two rigs so camera ID was always 1 or 2.

2. To process image data, `check_data.py` was executed on the output folder from `collect.py` or `eval.py`.

        python check_data.py [folder label] [number of images in folder] [comma-separated (x,y) coordinates of target] [camera_id]

      This script saves images of the animal tracks and a `.pkl` file containing animal coordinates on the plate, head angle, and body angle. These files can be compiled with other tracks to train agents as in the demo.

3. To run evaluation episodes as in Figures 2-3, 5-6, one can run `eval.py` after agents have been trained.

        python eval.py [camera_id] [animal line number] [target coordinates] --eptime=600

   The `--eptime` input is in seconds and denotes length of the evaluation episode.
