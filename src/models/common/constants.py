# instantiate the game environment parameters
train_env_reset_config = {
    "tower-seed": 99,  # fix floor generation seed to remove generalization
    "visual-theme": 0,  # default theme to remove generalization while training
    "agent-perspective": 0,
    "dense-reward": 1,
    "starting-floor": 0,
    "default-theme": 3
}

eval_env_reset_config = {
        "tower-seed": 99,
        "visual-theme": 0,
        "agent-perspective": 0,
        "dense-reward": 1
}

train_env_reset_config_industrial = {
    "tower-seed": 99,  # fix floor generation seed to remove generalization
    "visual-theme": 0,  # default theme to remove generalization while training
    "agent-perspective": 0,
    "dense-reward": 1,
    "default-theme": 3
}

eval_env_reset_config_industrial = {
    "tower-seed": 99,  # fix floor generation seed to remove generalization
    "visual-theme": 0,  # default theme to remove generalization while training
    "agent-perspective": 0,
    "dense-reward": 1,
    "default-theme": 3
}
