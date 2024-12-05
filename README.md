# 1. Few shot segmentation 🫁
This data challenge focuses on developing a highly accurate model for automatically segmenting anatomical structures and tumors in high-resolution 3D CT scans. The task is unique as the model must generalize based on shape recognition rather than prior knowledge of specific organs or tumors, despite incomplete labeling of anatomical structures. Adding to the complexity, the dataset includes both partially labeled and unlabeled data, necessitating a combination of off-topic learning, few-shot learning, and semi-supervised learning. We explored various approaches, including self-supervised methods for MaskRCNN (e.g., DiNOv2, MAE, SAM) and semi-supervised strategies like pseudo-labeling.

## College de France data challenge
We are currently 2nd out 20 on private ranking: [public ranking](https://challengedata.ens.fr/participants/challenges/150/ranking/public).

## Report with Some Insights ✨  
Check out our **report**: [here](https://drive.google.com/file/d/1MphvWSCc__vAgsqBbba5LtitUX6HSckc/view).

 
# 2. Setup 
We create our own Kaggle cluster leveraging the amazing work of [mva pepites](https://github.com/balthazarneveu/mva_pepites) !

## How to use
- custom your model/dataloader/training in /code folder (the configuration variable 'cfg' corresponds to CFG_EXPERIMENTS configuration.py)
- run the terminal command `python remote_training.py --user user0 --branch your_branch_name` and replace the given options
Note: by running this command, you will execute the function `run_experiment` in code/train.py for each configuration given in `CFG_EXPERIMENT` of configuration.py file.
- You can also add the option `--nowandb` to avoir logs in wandb (usefull for local tests)

## Get started 🚀
- copy this repo
- Create accounts on Kaggle and weight and biases, get there API tokens (https://wandb.ai/authorize and https://www.kaggle.com/settings)
- create a git token from your github accouunt: Settings->Developper settings->Fine-grained personal tokens->generate new tokens (give the token name you want, select only this repository, give Read-only acces for Contents in Repository permission, click on Generate token and copy the git_token)
- fill the __kaggle_login file with your kaggle API tokens and username (you can custom how to use them by replacing 'user0' key)
- In configuration.py, enter your git username, git repository (corresponding to this repository), add your dataset (here it's 'hugorbrt/few-shot-segmentation') and give a name of the notebook you will run on kaggle (and never change it after!)

## First time 
- Go to kaggle and check your notifications to access your notebook.
- Edit notebook manually
- Allow internet requires your permission (internet is required to clone the git), a verified kaggle account is required
- Allow Kaggle secrets to access wandb (https://www.kaggle.com/discussions/product-feedback/114053) by associating the `wandb_api_key` label to your wandb API key value.
- Add another Kaggle secret by associating the `git_token_raidium` label to your git token value.
- Quick save your notebook.
- Now run the remote training script again, this should execute.
