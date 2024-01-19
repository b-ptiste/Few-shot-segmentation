# Setup

- copy this repo
- Create accounts on Kaggle and weight and biases, get there API tokens (https://wandb.ai/authorize and https://www.kaggle.com/settings)
- create a git token from your github accouunt: Settings->Developper settings->Fine-grained personal tokens->generate new tokens (give the token name you want, select only this repository, give Read-only acces for Contents in Repository permission, click on Generate token and copy the git_token)
- fill the __kaggle_login file with your kaggle API tokens and username (you can custom how to use them by replacing 'user0' key)
- In configuration.py, enter your git username, git repository (corresponding to this repository), add your dataset (here it's 'hugorbrt/nlplsv3') and give a name of the notebook you will run on kaggle (and never change it after!)

# How to use
- custom your model/dataloader/training in /code folder (the configuration variable 'cfg' corresponds to CFG_EXPERIMENTS configuration.py)
- run the terminal command `python remote_training.py --user user0 --branch your_branch_name` and replace the given options
Note: by running this command, you will execute the function `run_experiment` in code/train.py for each configuration given in `CFG_EXPERIMENT` of configuration.py file.
- You can also add the option `--nowandb` to avoir logs in wandb (usefull for local tests)

# First time: 

- Go to kaggle and check your notifications to access your notebook.
- Edit notebook manually
- allow internet requires your permission (internet is required to clone the git), a verified kaggle account is required
- Allow Kaggle secrets to access wandb (https://www.kaggle.com/discussions/product-feedback/114053) by associating the `wandb_api_key` label to your wandb API key value.
- Add another Kaggle secret by associating the `git_token` label to your git token value.
- Quick save your notebook.
- Now run the remote training script again, this should execute.

# Few shot segmentation