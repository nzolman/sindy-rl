# SINDy-RL
This repository houses the code associated with the paper [**"SINDy-RL: Interpretable and Efficient Model-Based Reinforcement Learning"**](https://arxiv.org/abs/2403.09110). A video abstract of the work [can be found here](https://youtu.be/7Q8oNNsZGcA).
 While we do provide some amount of documentation, the intent for this repository is for research code illustrating the ideas and results of the paper; it is not expected to be used as a standalone pacakge, nor officially maintained. 
 However, the code is still being updated and cleaned to make it easier to read and use. Please check back for additional updates, and please use Github for any discussions, inquiries, and/or issues.

# Installation
This repository has only been tested on linux machines (Ubuntu). For issues with Mac/Windows, please leave a Git Issue. While there is no reason to suspect that this code will not work with other versions of Python 3.x, it has only been tested with Python=3.9.13, Python=3.10.6, and Python==3.10.12. For just using the algorithms with `gymnasium`, `dm_control`, or other easy-installation environments, you should be able to simply clone this repository and install from the requirements file. See "Local Installation" below. However, if you seek to use the fluids examples with Hydrogym, it is recommended you use a Docker container. 

**NOTE**: Between the original version on arXiv and February 2025, Hydrogym went through a major refactor. To use with the refactored version, we recommend you use the devcontainer (e.g. with VS Code). 
The original arXiv version of the paper uses the Docker container instructions below. To use the new Hydrogym repository (or the fork at https://github.com/nzolman/hydrogym/), follow the devcontainer instructions below.

**NOTE**: `ray` has undergone several changes since the original onset of this project. This has only been tested with `ray==2.6.3` with `ray.rllib` and `ray.tune`. There is no expectation this code will be updated to accommodate later versions of ray. 

## Local Install: `pip`
For local installations, you can simply run 

```
$ git clone https://github.com/nzolman/sindy-rl.git
$ cd sindy_rl
$ pip install -r requirements.txt
$ pip install -e .
```

## Devcontainer: newer Hydrogym versions.
Devcontainers are a convenient way to use docker containers that mount local file systems for persistent code changes. VSCode has a [great tutorial](https://code.visualstudio.com/docs/devcontainers/tutorial) on getting started with using devcontainers with their application. The devcontainer file is under `.devcontainer/.devcontainer.json`. 

First, clone this repository: 

```
$ git clone https://github.com/nzolman/sindy-rl.git
$ cd sindy_rl
```

You will need a compatible version of Hydrogym then need to clone Hydrogym. For convenience, I have a fork with the pinball environment here:  https://github.com/nzolman/hydrogym/. Simply run

```
$ git clone https://github.com/nzolman/hydrogym/
```

NOTE: At some point, Hydrogym started to use `git lfs` for downloading the meshes; for ease of installation, you may want to install `git lfs` first. Though, in principle, this can also be done after the devcontainer has been built. 

Now you can build the devcontainer, which will pull an image of hydrogym and install the requirements into the container. Follow the [tutorial](https://code.visualstudio.com/docs/devcontainers/tutorial) for how to do this. If you run into issues, you might need to manually install some python packages---[see this Hydrogym issue](https://github.com/dynamicslab/hydrogym/issues/198).


Once you're inside the container and you've installed any packages, you'll want to navigate to the root directory of the `sindy-rl/` repository and install it. Simply run the following as in the local install.

```
$ cd sindy_rl
$ pip install -r requirements.txt
$ pip install -e .
```


## Docker: Original arXiv version
The Cylinder example from the paper used a previous version of Hydrogym forked from a much earlier version of the code. Because of this, you can access a pre-built version that includes the forked Hydrogym at [https://hub.docker.com/r/nfzolman/sindy_rl](https://hub.docker.com/r/nfzolman/sindy_rl). 

If you don't have familiarity with Docker, a great staring guide can be found [here](https://www.datacamp.com/tutorial/docker-for-data-science-introduction). After installing Docker, the shortest path to running the code in this project is to pull the docker image by running:

```
$ docker pull nfzolman/sindy_rl:arxiv
```

If you've already pulled the `sindy_rl` repo, then you can run:

```
docker run  -u 0 \
            -v /local/path/to/sindy_rl:/home/firedrake/sindy-rl:z \
            -p 8888:8888 \
            -it sindy_rl:arxiv \
            /bin/bash
```

This will mount your local version of `sindy_rl` into the container so you can use its environment. After mounting and the container is running, you'll need to navigate to `/home/firedrake/sindy_rl` and run `pip install -e .`. This will add `sindy_rl` as a package to the docker container's python environment.

### Jupyterlab
The container exposes port 8888 and `jupyterlab` is available inside the container. You can host a jupyter instance using 

```
jupyter-lab --no-browser --allow-root --ip 0.0.0.0
```

which will map to the host at `localhost:8888`.


# Documentation
There is no official built documentation; however, you may find the quick start tutorials useful under `tutorials`. There are planned tutorials for more aspects of the code after the initial release. For a particular tutorial request, please use Github discussions.  

# Usage
The code relies on `rllib==2.6.3` and has not been updated for the new RLlib APIs. To align with RLlib and `ray.tune`'s configuration-driven API, every training experiment is defined by a configuration file. The configurations used in the paper can be found in `sindy_rl/config_templates`. Note, there are hardcoded path references to existing data—particularly for the Cylinder examples; these align with the location inside the docker container. These should be replaced with the appropriate files from `data/` if you are seeking to run these.

After setting up the `sindy_rl` package, the main entry point to the code is `sindy_rl/pbt_dyna.py`, which essentially acts as Algorithm 1 in the paper. Simply change the name/location of the configuration file to be the one you'd like to run; this includes functionality for Population-Based Training (PBT) with Ray Tune by using the keyword `use_pbt: True` in the configuration file. For running MB-MPO or standard baselines, you can find scripts under `sindy_rl/scripts`. By default, all experiments use Ray Tune to launch 20 different trials (identical configuration, but different random seeds/initializations).

When running these scripts, it can take several hours (or even days) to run. However, the ray logger should start producing result outputs after a few minutes. When launched on a computer with fewer CPUs than number of trials, the trials will run serially until completed. 

# Accessing Data/Results
Due to the size of the benchmarks (20 trials per experiment), only individual checkpoints are available. All the data/models can be found in the associated [Hugging Face repository](https://huggingface.co/nzolman/sindy-rl_data) (Note: see [issue](https://github.com/nzolman/sindy-rl/issues/2) for more detail on this choice). Because there are significant binary files, these are zipped up in a tarball and managed with `git lfs`. In order to download the data, you should make sure that you have `git lfs` installed. If you install `git lfs` after cloning, you will need to run `git lfs pull`. 

All of this has been made simple using hooks inside our Makefile (at least for Unix users, e.g. Linux or MacOS). To get the data and untar.gz into a folder named `./data`, simply navigate to the root directory of the repo and run: 

```
# clone data from Hugging Face
make grab_data

# set up all the data by unzipping
make unzip_data
```

If you have issues with this (e.g., because folders exist from an older version of the repo), you can run ```make clean_data```. However, note that this will remove ALL the contents of `./data`! This should be done with extreme caution!!