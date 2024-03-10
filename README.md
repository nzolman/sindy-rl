# SINDy-RL
This repository houses the code associated with the paper **"SINDy-RL: Interpretable and Efficient Model-Based Reinforcement Learning"** $\textcolor{red}{<\text{insert URL}>}$
.
 While we do provide some amount of documentation, the intent for this repository is for research code illustrating the ideas and results of the paper; it is not expected to be used as a standalone pacakge, nor officially maintained. 
 However, the code is still being updated and cleaned to make it easier to read and use. Please check back for additional updates and use github for any discussions, inquiries, and/or issues.

# Installation
Note: This repository has only been tested on linux machines (Ubuntu). For issues with Mac/Windows, please leave a Git Issue. 

## Docker
The main repo can be installed below using pip, however, you may find it easier to run the Hydrogym example using a pre-built Docker container. 
While Hydrogym now offers containers for use, this project forked an earlier version of Hydrogym and the code in this repository is only guaranteed to work with that version. 
Because of this, you can access a pre-built version that includes the forked Hydrogym at $\textcolor{red}{<\text{insert URL}>}$. 


## Pip
For local installations, you can simply run 

```
$ git clone https://github.com/nzolman/sindy-rl.git
$ cd sindy_rl
$ pip install -r requirements.txt
$ pip install -e .
```


# Documentation
There is no official built documentation; however, you may find the quick start tutorials useful under 
$\textcolor{red}{<\text{insert path}>}$. 

# Accessing Data/Results
Due to the size of the benchmarks (20 trials per experiment), only indvidual checkpoints are available with the repository. You can find them under $\textcolor{red}{<\text{insert path}>}$. 
