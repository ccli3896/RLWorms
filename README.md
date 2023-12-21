# RLWorms
This repository contains code and data used to set up the nematode C elegans as an RL environment. To accompany paper [Discovering neural policies to drive behavior by integrating deep reinforcement learning agents with biological neural networks](https://www.biorxiv.org/content/10.1101/2022.09.19.508590v2.article-metrics) by Li, Kreiman, and Ramanathan (2023). 

# Contents
1. Introduction
2. Repository folders and descriptions
3. Video demos
4. System requirements and installation
5. Tutorials
6. Instructions for use with animals
7. References

# Introduction
Guiding or improving animal behavior directly through the nervous system has been a common goal for neuroscience and robotics researchers alike [1-3]. Previous works in brain interfaces and animal robotics have attempted to use direct interventions to affect behavior on a variety of tasks, relying on manual specification for stimulation frequencies, locations, dynamics, and patterns [4–21]. A central difficulty with these approaches is that manual tuning has limited applicability, as it relies on knowledge of the neural circuits or mechanisms involved. For direct neural stimulation, effective patterns can vary depending on which neurons are targeted and on the animal itself [22,23], and thus, even though technologies for precise neuronal modulation exist [24,25], there still lies the challenge of how to design and choose an algorithm that can systematically and automatically learn strategies to activate a set of neurons to improve a particular behavior [26–30]. 

<p align="center">
<img width="300" alt="Screenshot 2023-12-13 at 9 34 25 PM" src="https://github.com/ccli3896/RLWorms/assets/33879208/ddd465d3-4136-446c-9beb-4a38ca4842bc"> <img width="400" alt="Screenshot 2023-12-13 at 9 48 23 PM" src="https://github.com/ccli3896/RLWorms/assets/33879208/8934d13d-9449-4192-bf9d-c1324d39afe6">
</p>


Here we addressed this challenge using deep reinforcement learning (RL). In an RL setting, an agent collects rewards through interactions with its environment. We present a flexible framework that can, given only a reward signal, observations, and a set of relevant actions, learn different ways of achieving a goal behavior that adapt to the chosen interface. We tested our ideas on the nematode C. elegans, interfacing an RL agent with its nervous system using optogenetic tools [24,27]. This animal has small and accessible nervous system and yet still possesses a rich behavioral repertoire [31]. In a natural setting, C. elegans must navigate variable environments to avoid danger or find targets like food. Therefore, we aimed to build an RL agent that could learn how to interface with neurons to assist C. elegans in target-finding and food search. We tested the agent by connecting it to different sets of neurons with distinct roles in behavior. The agents could not only couple with different sets of neurons to perform a target-finding task, but could also generalize the task to improve food search across novel environments in a zero-shot fashion, that is, without any prior training. 

# Repository
- Agents
  - Trained agents used in evaluations, separated by genetic line.
- Animal scripts
  - Code used to interact with animals, including collecting training data and evaluating agents on live animals.
- Basic evaluation data
  - Agents were trained and tested on each of 6 optogenetically modified animal lines, labeled as in the table below (Agent Training and Visualization).
  - Control and experimental tracks; Figures 2-4.
- Cross evaluation data
  - Animal track data used in Figure 5.
- Foodsearch data
  - Tracking data used in Figure 6B-G.
- Obstacle data
  - Tracking data used in Figure 6I-K.
- Training data
  - Datasets used to train agents.
  - Concatenations of 20 min episodes of randomly flashing light data, with animals switched out at the end of every episode.
  - See Figure 1 and Methods in the manuscript for details.
- Training scripts
  - Code to train soft actor-critic agents on animal data.
- Tutorials
  - Folder containing tutorials that do not require animals.


# Video demos
In the two videos below, the blue frame represents agent decisions and the red circle represents a virtual target. Videos are 8x speed and animals are roughly 1 mm in length. Plates are 4 cm in diameter.

## Random actions
https://github.com/ccli3896/RLWorms/assets/33879208/bc5858d8-7dbe-4766-a7da-616ecec953f1

## After training
https://github.com/ccli3896/RLWorms/assets/33879208/5d229fe1-37bb-485c-b822-879f566f575a

# System requirements and installation
Training can be completed on any machine with Pytorch (tested on torch==2.0.1). The demo has been tested on a 2020 MacBook Pro with an Apple M1 chip running Ventura 13.4.1. For the manuscript, the computations were run on the FASRC Cannon cluster supported by the FAS Division of Science Research Computing Group at Harvard University. GPU types available to us are in [this list.](https://docs.rc.fas.harvard.edu/kb/running-jobs/#Using_GPUs)
Training completed in under an hour with these resources and a memory pool of 10gb for all cores during an array of training jobs of 20-30 agents.

For the hardware setup in the manuscript (Figure 1), we used an Edmund Optics 5012 LE Monochrome USB 3.0 camera or a ThorLabs DCC1545M with [pypyueye](https://github.com/galaunay/pypyueye.git). 
Lights for optogenetic illumination were Kessil PR160L LEDs at wavelengths of 467 nm for blue and 525 nm for green. LEDs were controlled by National Instruments DAQmx devices with the [nidaqmx library](https://nidaqmx-python.readthedocs.io/en/latest/).

Due to hardware compatibility issues, data collection and evaluation on live animals must be completed on a Windows machine (all live animal data collected using Windows 10 and 11).

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

4. To run evaluation episodes as in Figures 2-3, 5-6, one can run `eval.py` after agents have been trained.

        python eval.py [camera_id] [animal line number] [target coordinates] --eptime=600

   The `--eptime` input is in seconds and denotes length of the evaluation episode.

# References
1.	Romano, D., Donati, E., Benelli, G. & Stefanini, C. A review on animal–robot interaction: from bio-hybrid organisms to mixed societies. Biol. Cybern. 113, 201–225 (2019).
2.	Tankus, A., Fried, I. & Shoham, S. Cognitive-motor brain–machine interfaces. J. Physiol. Paris 108, 38–44 (2014).
3.	Bostrom, N. & Sandberg, A. Cognitive Enhancement: Methods, Ethics, Regulatory Challenges. Sci. Eng. Ethics 15, 311–341 (2009).
4.	Afraz, S.-R., Kiani, R. & Esteky, H. Microstimulation of inferotemporal cortex influences face categorization. Nature 442, 692–695 (2006).
5.	Bonizzato, M. & Martinez, M. An intracortical neuroprosthesis immediately alleviates walking deficits and improves recovery of leg control after spinal cord injury. Sci. Transl. Med. 13, eabb4422 (2021).
6.	Enriquez-Geppert, S., Huster, R. J. & Herrmann, C. S. Boosting brain functions: Improving executive functions with behavioral training, neurostimulation, and neurofeedback. Int. J. Psychophysiol. 88, 1–16 (2013).
7.	Iturrate, I., Pereira, M. & Millán, J. del R. Closed-loop electrical neurostimulation: Challenges and opportunities. Curr. Opin. Biomed. Eng. 8, 28–37 (2018).
8.	Lafer-Sousa, R. et al. Behavioral detectability of optogenetic stimulation of inferior temporal cortex varies with the size of concurrently viewed objects. Curr. Res. Neurobiol. 4, 100063 (2023).
9.	Lu, Y. et al. Optogenetically induced spatiotemporal gamma oscillations and neuronal spiking activity in primate motor cortex. J. Neurophysiol. 113, 3574–3587 (2015).
10.	Salzman, D., C., Britten, K. H. & Newsome, W. T. Cortical microstimulation influences perceptual judgements of motion direction. Nature 346, 174–177 (1990).
11.	Schild, L. C. & Glauser, D. A. Dual Color Neural Activation and Behavior Control with Chrimson and CoChR in Caenorhabditis elegans. Genetics 200, 1029–1034 (2015).
12.	Xu, J. et al. Thalamic Stimulation Improves Postictal Cortical Arousal and Behavior. J. Neurosci. 40, 7343–7354 (2020).
13.	Park, S.-G. et al. Medial preoptic circuit induces hunting-like actions to target objects and prey. Nat. Neurosci. 21, 364–372 (2018).
14.	Yang, J., Huai, R., Wang, H., Lv, C. & Su, X. A robo-pigeon based on an innovative multi-mode telestimulation system. Biomed. Mater. Eng. 26 Suppl 1, S357-363 (2015).
15.	Holzer, R. & Shimoyama, I. Locomotion control of a bio-robotic system via electric stimulation. in Proceedings of the 1997 IEEE/RSJ International Conference on Intelligent Robot and Systems. Innovative Robotics for Real-World Applications. IROS ’97 vol. 3 1514–1519 vol.3 (1997).
16.	Talwar, S. K. et al. Rat navigation guided by remote control. Nature 417, 37–38 (2002).
17.	Sato, H. et al. A cyborg beetle: Insect flight control through an implantable, tetherless microsystem. in 2008 IEEE 21st International Conference on Micro Electro Mechanical Systems 164–167 (2008). doi:10.1109/MEMSYS.2008.4443618.
18.	Peckham, P. H. & Knutson, J. S. Functional electrical stimulation for neuromuscular applications. Annu. Rev. Biomed. Eng. 7, 327–360 (2005).
19.	Kashin, S. M., Feldman, A. G. & Orlovsky, G. N. Locomotion of fish evoked by electrical stimulation of the brain. Brain Res. 82, 41–47 (1974).
20.	Hinterwirth, A. J. et al. Wireless Stimulation of Antennal Muscles in Freely Flying Hawkmoths Leads to Flight Path Changes. PLOS ONE 7, e52725 (2012).
21.	Sanchez, C. J. et al. Locomotion control of hybrid cockroach robots. J. R. Soc. Interface 12, 20141363 (2015).
22.	Bergmann, E., Gofman, X., Kavushansky, A. & Kahn, I. Individual variability in functional connectivity architecture of the mouse brain. Commun. Biol. 3, 1–10 (2020).
23.	Mueller, S. et al. Individual Variability in Functional Connectivity Architecture of the Human Brain. Neuron 77, 586–595 (2013).
24.	Husson, S. J., Gottschalk, A. & Leifer, A. M. Optogenetic manipulation of neural activity in C. elegans: from synapse to circuits and behaviour. Biol. Cell 105, 235–250 (2013).
25.	Nagel, G. et al. Channelrhodopsin-2, a directly light-gated cation-selective membrane channel. PNAS 100, 13940–13945 (2003).
26.	Kocabas, A., Shen, C.-H., Guo, Z. V. & Ramanathan, S. Controlling interneuron activity in Caenorhabditis elegans to evoke chemotactic behaviour. Nature 490, 273–277 (2012).
27.	Leifer, A. M., Fang-Yen, C., Gershow, M., Alkema, M. J. & Samuel, A. D. T. Optogenetic manipulation of neural activity in freely moving Caenorhabditis elegans. Nat. Methods 8, 147–152 (2011).
28.	Wen, Q. et al. Proprioceptive Coupling within Motor Neurons Drives C. elegans Forward Locomotion. Neuron 76, 750–761 (2012).
29.	Hernandez-Nunez, L. et al. Reverse-correlation analysis of navigation dynamics in Drosophila larva using optogenetics. eLife 4, e06225 (2015).
30.	Donnelly, J. L. et al. Monoaminergic Orchestration of Motor Programs in a Complex C. elegans Behavior. PLOS Biol. 11, (2013).
31.	Haarnoja, T. et al. Soft actor-critic algorithms and applications. ArXiv Prepr. ArXiv181205905 (2018).
