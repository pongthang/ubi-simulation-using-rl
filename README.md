# Universal Basic Income Simulation Using RL

## Introduction:

With the recent advancements in the field of AI, there is an ongoing debate of AIs replacing
humans in most of the jobs that exist today. With this situation, it will become important for
the government to support unemployed people by providing them with a universal basic
income(UBI) .
The project aimed at simulating an economy to study the effect of the introduction of AI
agents in a world.Our target is to find how agents will react with the introduction of AI agents
and to find the optimal value of UBI income that the government should provide to sustain all
the people in the economy.

** for deeper understanding for our project , you can check [Readme_from_ai_economist](/Readme_from_ai_economist/README.md) folder and our report [Univeral Basic Income Simulation Using RL doc b19147 b19231 b19011.pdf](Univeral%20Basic%20Income%20Simulation%20Using%20RL%20doc%20b19147%20b19231%20b19011.pdf). 


## Getting started:

### Installing from Source:

1. Clone this repository to your local machine:

  ```
   git clone https://github.com/pongthang/ubi-simulation-using-rl.git
   ```

   2. Create a new conda environment (named "ubi-simulation-using-rl" below - replace with anything else) and activate it

  ```pyfunctiontypecomment
   conda create --name ubi-simulation-using-rl python=3.7 --yes
   conda activate ubi-simulation-using-rl
   ```

3. Either

   a) Edit the PYTHONPATH to include the ubi-simulation-using-rl directory
  ```
   export PYTHONPATH=<local path to ubi-simulation-using-rl>:$PYTHONPATH
   ```

   OR

   b) Install as an editable Python package
  ```pyfunctiontypecomment
   cd ubi-simulation-using-rl
   pip3 install -e .
   ```
   ### Testing your Install

To test your installation, try running:

```
conda activate ubi-simulation-using-rl
python3 -c "import ai_economist"
```

## About the simulation:
There will be 4 agents. And one agent who will act as governor. 4 agents will earn money. Governor agent will collect taxes and distribute as UBI.
## How to train the RL agent

There are two steps:
* First you need to train the agents are trained without governor in order to explore what they can do - earning money by collecting woods, building house etc. This simulation part is called "Free Market".

```
conda activate ubi-simulation-using-rl
cd ubi-simulation-using-rl/tutorials/rllib/
python3 training_script.py --run-dir phase1
```
Here phase1 directory contains the configuration files and a folder called "ckpts" where the trained model checkpoints are being saved. 

* Then the we use the last checkpoint of the model in the next simulation - 4 agents and 1 governor where taxes are introduced.
Corresponding configuration of simulation are written in phase2 folder.

```
conda activate ubi-simulation-using-rl
cd ubi-simulation-using-rl/tutorials/rllib/
python3 training_script.py --run-dir phase2
```

## Results:

Check the results by plotting. Use the jupyter notebook 
[plot_your_results.ipynb](/tutorials/plot_your_results.ipynb) in the "tutorials" folder to see the output of the RL model.
