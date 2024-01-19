NB_ID = "notebook-few-shot-segmentation"  # This will be the name which appears on Kaggle.
GIT_USER = "b-ptiste"  # Your git user name
GIT_REPO = "Few-shot-segmentation"  # Your current git repo
# Keep free unless you need to acess kaggle datasets. You'll need to modify the remote_training_template.ipynb.
KAGGLE_DATASET_LIST = ['hugorbrt/few-shot-segmentation']
CFG_EXPERIMENTS = {
    0:{  
        'who': GIT_USER,
        'name_exp': "test_exp",
        'scheduler': 'CosineAnnealingLR',
        'nb_epochs': 15,
        'batch_size': 24,
        'learning_rate': 2e-5,
        'model_name': 'distilbert-base-cased',
        'num_node_features': 300,
        'nout':  768,
        'nhid': 500,
        'graph_hidden_channels': 300,
        'heads': 30,
        'comment': '',
    },
}