# Interactive behaviour design

[![Build Status](https://travis-ci.com/HumanCompatibleAI/interactive-behaviour-design.svg?branch=master)](https://travis-ci.com/HumanCompatibleAI/interactive-behaviour-design)

This repository contains the code for
[*Towards an IDE for agent design*](https://drive.google.com/file/d/1lYdp5ym5OeL0WpzVBLL1rZYpga_1h6WC/view),
and for Matthew's master's thesis, *Designing agents with user-defined concepts*.

## Cloning

Note that this repository contains submodules.
To make sure your clone includes those submodules, do:

`$ git clone --recurse-submodules https://github.com/HumanCompatibleAI/interactive-behaviour-design`.

## Dependencies

For binary dependencies, see the [Dockerfile](Dockerfile).

Package dependencies are listed in the [Pipfile](Pipfile), but need to be installed
in a particular order. Again, check the [Dockerfile](Dockerfile).

A Docker image set up with the current set of requirements is hosted on the
Amazon Elastic Container Registry, but is currently private because it contains
a MuJoCo key. This container is used for automatic tests with Travis CI and
for runs on AWS Batch (described below). To rebuild the Docker image, see
[`scripts/build_docker.sh`](scripts/build_docker.sh).

## Usage

[`run.py`](run.py) is the main program file. Basic usage is

```
$ python3 run.py <env name>
```

where `<env name>` is e.g.

* `SeaquestDeepMindDense-v0`
* `LunarLanderStatefulStats-v0`
* `FetchPickAndPlace-Repeat1-BinaryGripper-5InitialBlockPos-FixedGoal-GripperBonus-NoET-SlowGripper-NoVanillaRL-PartialObs-v0`

For a full list of command-line options, see [`params.py`](params.py).
See [`wrappers`](wrappers) for a full list of supported environments.

### Web interface

Once finished initialising, `run.py` serves a web app on port 5000. Important pages in this app are:

* `/status` ([`cmd_status.py`](web_app/cmd_status.py)): overall program status report
* `/run_cmd` (`cmd_status.py`): run a command
* `/label_video` ([`labelling.py`](web_app/labelling.py)): label goal states from videos of episodes
* `/demonstrate` ([`demonstrations.py`](web_app/demonstrations.py)): demonstrate episodes using trained primitives
    * `/view_trajectories` ([`view_trajectories.py`](web_app/view_trajectories.py)): examine demonstrated episodes (for debugging)
* `/compare_segments` ([`comparisons.py`](web_app/comparisons.py)): provide DRLHP-style comparisons

Training is controlled through commands given to `/run_cmd`. Commands follow a basic format of `/run_cmd?cmd=cmd_name&args=cmd_args`.
Important commands are listed below.

**Goal state classifier commands**
* `cmd=add_classifier&name=classifier_name`: add a goal state classifier
* `cmd=use_classifier&name=classifier_name`: specified classifier is marked as 'selected';
* `cmd=train&n_epochs=n_epochs`: train the selected classifier

**Policy commands**
* `cmd=add_policy&name=policy_name`: add a (randomly-initialised) policy
* `cmd=training_mode&mode={reward_only,bc_only,reward_plus_bc,no_training}`: set how the current policy is to be trained (default: `reward_only`)
* `cmd=set_reward_source&src={env,classifier,drlhp,none}`: for reward-based training modes, use rewards from the environment,
  from the selected goal state classifier, from the DRLHP reward predictor, or provide zero rewards (default: `none`)
* `cmd=use_policy&name=policy_name`: start training the specific policy

**Reward predictor commands**
* `cmd=start_drlhp_training`: start DRLHP reward predictor training
* `cmd=stop_drlhp_training`: stop training

**Reset state commands**
* `cmd=add_reset_pool&name=pool_name&max_len=max_len`:
  add a named pool of reset states. If `max_len` is specified, old states are dropped
  when the pool exceeds the specified size.
* `cmd=use_reset_pool&from={training,demonstrations}&to={training,demonstrations`:
   connect the specified reset pool to receive states from either training
   of the current policy or from demonstrations, and to supply states similarly.
* `cmd=set_training_reset_mode&mode={use_env_reset,from_state_cache}`: 
  when resetting the training environment, either get a reset state from the environment,
  or take a reset state from the reset pool currently connected to training.
  
For an example of the proper full sequence of commands (after goal state classifiers
have been trained), see [`scripts/train/auto_train_prefs.py`](scripts/train/auto_train_prefs.py).

## Automatic training

To train using demonstrations from an oracle instead of a human, use
[`scripts/train/auto_train_prefs.py`](scripts/train/auto_train_prefs.py).
This will automatically set up a tmux session
running both `run.py` and the [oracle](oracle.py), and coordinate
the various stages of training.

Use [`scripts/train/auto_train_rl.py`](scripts/train/auto_train_rl.py)
to train using vanilla RL with rewards from the environment.

To produce a full set of results using the oracle, run the commands produced by
[`scripts/train/train_all.py`](scripts/train/train_all.py).

### Training on AWS

See [`interactive-behaviour-design-aws`](https://github.com/HumanCompatibleAI/interactive-behaviour-design-aws)
for scripts to train on AWS. For example, to produce a full set of results using AWS:

`$ scripts/train_all.py | aws_run_ibd_cmds.py`

## Tests

To run tests:

`$ python -u -m unittest discover -v -p '*_test.py'`

## Bugs and TODOs

* Implement monitoring for what proportion of segments from each episode are
  labelling when training in DRLHP mode.
* Success rate for the demonstration environment (`env_demo/success_rate`) is
  broken.
