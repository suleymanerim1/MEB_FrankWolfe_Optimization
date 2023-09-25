# Minimum Enclosing Ball and Anomaly Detection using variants of Frank-Wolfe

## Table of contents
* [General Info](#General-Info)
* [Group Members](#Group-Members)
* [Project Structure](#Project-Structure)
* [Setup](#Setup)
* [Run](#Run)

## General Info
This is a final project for Optimization for Data Science course in University of Padua.

### Task
Minimum Enclosing Ball and Anomaly Detection
- analyze in depth the papers and the theoretical results
- implement the Away-Step Frank Wolfe [Lacoste-Julien et al]
- implement PFW algorithm described in [Tsuji et al., Algorithm 1]
- implement the first algorithm in the paper [Yildirim, 2008].
- choose two datasets and use the MEB defined over them to find anomalies (new points that are out of the MEB)

## Group Members:
- Dejan Dichoski marticola number : 2071886
- Marija Cveevska marticola number : 2072060
- Suleyman Erim marticola number : 2072058

## Project Structure
```bash
├── configurations
│   ├── experiment1_Uniform
│   ├── experiment2_Gaussian
│   ├── experiment6_CustomerChurn
│   ├── experiment9_BreastCancer
├── datasets
│   ├── CustomerChurn
│   └── BreastCancer
├── experiment_results (for each configuration)
│   ├── graphs
│   ├── output.yaml
│   └── log files
├── papers
│   ├── Lacoste_Julien et al.
│   ├── Tsuji et al.
│   ├── Yıldırım
├── runs (1 dummy run output added)
├── src (source codes)
│   ├── FrankWolfeVariants.py
│   ├── data_generation.py
│   ├── execution.py
│   ├── logger.py
│   ├── plotting.py
│   ├── utils.py
│   └── partials/template
├── main.py
├── setup.py
├── requirements.txt
├── README.md
└── .gitignore
```

## Setup
```
$ conda update conda -y
$ conda create -p venv python==3.10 -y
$ conda activate venv/
```
Go to  requirements.txt file and uncomment symbol (#) before "-e ."
Then,
```
$ pip install -r requirements.txt
```

## Run
Choose a configuration file(.yaml) to run an experiment
configs/"experiment.yaml"
```
$ python main.py --cfg "exp0_Default.yaml"
```
Results will be saved in runs/exp0_Defaulft folder


