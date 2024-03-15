# SINDy-RL
This repository houses the code associated with the paper [**"SINDy-RL: Interpretable and Efficient Model-Based Reinforcement Learning"**](https://arxiv.org/abs/2403.09110).
 While we do provide some amount of documentation, the intent for this repository is for research code illustrating the ideas and results of the paper; it is not expected to be used as a standalone pacakge, nor officially maintained. 
 However, the code is still being updated and cleaned to make it easier to read and use. Please check back for additional updates, and please use Github for any discussions, inquiries, and/or issues.

# Installation
Note: This repository has only been tested on linux machines (Ubuntu). For issues with Mac/Windows, please leave a Git Issue. While there is no reason to suspect that this code will not work with other versions of Python 3.x, it has only been tested with Python=3.9.13 and Python=3.10.6. As discussed below, if you seek to use Hydrogym with this version of the code, we recommend you use the Docker container below (which uses Python=3.10.6) as opposed to creating your own installation. Future compatibility with the initial Hydrogym release is expected, but not currently supported. 

## Docker
The main repo can be installed below using pip, however, you may find it easier to run the Hydrogym example using a pre-built Docker container. 
While Hydrogym now offers containers for use, this project forked an earlier version of Hydrogym and the code in this repository is only guaranteed to work with that version. 
Because of this, you can access a pre-built version that includes the forked Hydrogym at [https://hub.docker.com/r/nfzolman/sindy_rl](https://hub.docker.com/r/nfzolman/sindy_rl). 

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


## Local Install: `pip`
For local installations, you can simply run 

```
$ git clone https://github.com/nzolman/sindy-rl.git
$ cd sindy_rl
$ pip install -r requirements.txt
$ pip install -e .
```

# Documentation
There is no official built documentation; however, you may find the quick start tutorials useful under `tutorials`. There are planned tutorials for more aspects of the code after the initial release. For a particular tutorial request, please use Github discussions.  

# Usage
The code relies on `rllib==2.6.3` and has not been updated for the new RLlib APIs. To align with RLlib and `ray.tune`'s configuration-driven API, every training experiment is defined by a configuration file. The configurations used in the paper can be found in `sindy_rl/config_templates`. Note, there are hardcoded path references to existing data—particularly for the Cylinder examples; these align with the location inside the docker container. These should be replaced with the appropriate files from `data/` if you are seeking to run these.

After setting up the `sindy_rl` package, the main entry point to the code is `sindy_rl/pbt_dyna.py`, which essentially acts as Algorithm 1 in the paper. Simply change the name/location of the configuration file to be the one you'd like to run; this includes functionality for Population-Based Training (PBT) with Ray Tune by using the keyword `use_pbt: True` in the configuration file. For running MB-MPO or standard baselines, you can find scripts under `sindy_rl/scripts`. By default, all experiments use Ray Tune to launch 20 different trials (identical configuration, but different random seeds/initializations).

# Accessing Data/Results
Due to the size of the benchmarks (20 trials per experiment), only indvidual checkpoints are available with the repository. These can be found under `data/`. 

Because there are significant binary files, these are zipped up in a tarball and managed with `git lfs`. In order to download the data, you should make sure that you have `git lfs` installed. If you install `git lfs` after cloning, you will need to run `git lfs pull`. 

 To untar.gz them. For unix users (linux, macos), this was made easy with our Makefile. You can simply navigate to the root directory of the repo and run 

```make unzip_data```

