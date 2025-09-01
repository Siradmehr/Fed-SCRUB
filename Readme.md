# How to run
first create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows use `venv\Scripts\activate`
```
then install the required packages:
```bash 
pip install -r requirements.txt
```

Then you need to make two directory:
```bash
mkdir data
mkdir checkpoints
```
For different scenrios You need to run with different env folder.
export the EXP_ENV_DIR variable to point to the correct env folder.
```bash
export EXP_ENV_DIR=envs  # For Linux/Mac
set EXP_ENV_DIR=envs  # For Windows
```
In the env folder, you can find different scenarios.
In .env you have to set these parameters:
```
NAME=fed_fscrub
RESUME=
DATASET=mnist
DATAROOT=data/mnist
MODEL=LeNet5
SEED=17
CONFIG_ID=2
CONFIG_NUMBER=2
FORGET_CLASS={1:0.5}
CLIENT_ID_TO_FORGET=0
Client_ID_TO_EXIT=0
UNLEARNING_CASE="CONFUSE"
NUM_CLASSES=10
LOCAL_EPOCHS=2
MIN_EPOCHS=2
MAX_EPOCHS=2
NUM_ROUNDS=30
MIN_CLIENTS=4
NUM_SUPERNODES=4
CLIENT_RESOURCES_NUM_CPUS=1
CLIENT_RESOURCES_NUM_GPUS=0.2
STARTING_PHASE="PRETRAIN"
TEACHER="INIT"
LAST_MAX_STEPS=5
LR_DECAY_EPOCHS=10,50,100
LR_ROUND=10,20
```
DATASET and DaTAROOT: dataset name and path. Supported datasets are mnist, cifar10, cifar100, tinyimagenet.

MODEL: model name. Supported models are LeNet5 for mnist, ResNet18 for cifar10 and cifar100, ResNet18 for tinyimagenet.

FORGET_CLASS a dictionary of classes to forget and their corresponding weights. For example, {1:0.5, 2:0.5} means to forget class 1 and class 2 with equal weights.

CLIENT_ID_TO_FORGET client id to forget the specified amount set in FORGET_CLASS.

Client_ID_TO_EXIT When a client wants to remove its whole dataset.

UNLEARNING_CASE available options are "CONFUSE", "BACKDOOR", "NONE". -> applied method when pretraining.




MIN_EPOCHS, MAX_EPOCHS: number of each client round of training epochs for Max phase and Min phase.

LOCAL_EPOCHS: number of each client round of training epochs for Pretrain phase.

NUM_ROUNDS: number of communication rounds.

NUM_SUPERNODES: number of supernodes. (all clients)

MIN_CLIENTS: number of clients selected in each round.

CLIENT_RESOURCES_NUM_CPUS: number of cpus allocated for each client.

CLIENT_RESOURCES_NUM_GPUS: number of gpus allocated for each client.

STARTING_PHASE: the starting phase of the training. Options are "PRETRAIN", "MAX", "MIN".

LAST_MAX_STEPS: number of last rounds to perform Max phase.


```commandline
LR_DECAY_EPOCHS=10,50,100
LR_ROUND=10,20
```
currently disabled.

In .env.training file You would see 
```commandline

MOMENTUM=0.9
WEIGHT_DECAY=0.0005
LR=0.5
LR_DECAY_RATE=0.1
SGDA_WEIGHT_DECAY=0.1
SGDA_MOMENTUM=0.9
DEVICE=cuda:0
LOSSCLS=KL
LOSSCLS_ARGS=0
LOSSKD=KL
LOSSKD_ARGS=0
LOSSDIV=KL
LOSSDIV_ARGS=0
GAMMA=1.0
ALPHA=0.5
BETA=0.0
SMOOTHING=0.5
KD_T=2.0
DISTILL=kd
RETRAIN_BATCH=32
FORGET_BATCH=32
VAL_BATCH=32
TEST_BATCH=32
```

Important parameters are:
BATCH sizes for training, forgetting, validation, and testing.
LOSSCLS, LOSSKD, and LOSSDIV: loss functions KL AND JS.




