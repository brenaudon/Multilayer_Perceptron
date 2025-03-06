"""
This script parses and validates configuration files for training neural networks.

It ensures the integrity of layer definitions, learning rate schedules, and optimizers.

Dependencies:
    - json
"""

import json

def validate_layers(config):
    """
    Validate the part describing layers of a neural network configuration file.

    @param config: The configuration dictionary.
    @type  config: dict
    """
    # Extract keys that match "layerX"
    layer_keys = [key for key in config.keys() if key.startswith("layer")]

    # Convert them to integers and sort them
    layer_numbers = sorted(int(key[5:]) for key in layer_keys)

    if len(layer_numbers) < 2:
        raise ValueError("At least two hidden layers are required.")

    if layer_numbers[0] != 1:
        raise ValueError("First hidden layer must be layer1.")

    # Check for missing numbers in sequence (layer1, layer2, ... must be consecutive)
    for i in range(len(layer_numbers) - 1):
        if layer_numbers[i + 1] != layer_numbers[i] + 1:
            raise ValueError(f"Layer sequence is broken: missing layer {layer_numbers[i] + 1}")

    # Check for missing keys in each layer
    for i in layer_numbers:
        if config.get(f'layer{i}').get('nb_neurons') is None:
            print(f"Warning: 'nb_neurons' not found in configuration file for layer{i}. Using default number of neurons: 24.")
        if config.get(f'layer{i}').get('activation') is None:
            print(f"Warning: 'activation' not found in configuration file for layer{i}. Using default activation function 'sigmoid'.")
        if config.get(f'layer{i}').get('initialization') is None:
            print(f"Warning: 'initialization' not found in configuration file for layer{i}. Using default initialization function 'random_normal'.")


def warning_schedule_params(config):
    """
    Warn the user that default values are used if learning rate schedule parameters are missing.

    @param config: The configuration dictionary.
    @type  config: dict
    """

    schedule_function = config.get('schedule')
    schedule_params = config.get('schedule_params')
    if schedule_function:
        if schedule_params is None or (schedule_params.get('initial_learning_rate') is None and config.get('learning_rate') is None):
            print("Warning: 'initial_learning_rate' not found in configuration file for learning rate schedule. Using default value of 0.01.")
        match schedule_function:
            case 'step':
                if schedule_params is None or schedule_params.get('drop_factor') is None:
                    print("Warning: 'drop_factor' not found in configuration file for step decay learning rate schedule. Using default value of 0.1.")
                if schedule_params is None or schedule_params.get('epochs_drop') is None:
                    print("Warning: 'epochs_drop' not found in configuration file for step decay learning rate schedule. Using default value of 50.")
            case 'exponential':
                if schedule_params is None or schedule_params.get('decay_rate') is None:
                    print("Warning: 'decay_rate' not found in configuration file for exponential decay learning rate schedule. Using default value of 0.01.")
            case 'time_based':
                if schedule_params is None or schedule_params.get('decay_rate') is None:
                    print("Warning: 'decay_rate' not found in configuration file for time based decay learning rate schedule. Using default value of 0.01.")
            case 'cosine':
                if config.get('epochs') is None:
                    print("Warning: 'epochs' not found in configuration file. Using default value of 200.")
            case _:
                raise ValueError(f"Unknown learning rate schedule function: {schedule_function}")
    elif config.get('learning_rate') is None:
        print("Warning: 'learning_rate' not found in configuration file. Using default value of 0.002.")

def warning_optimizer_params(config):
    """
    Warn the user that default values are used if optimizer parameters are missing.

    @param config: The configuration dictionary.
    @type  config: dict
    """
    optimizer_name = config.get('optimization')
    optimizer_params = config.get('optimizer_params')
    if optimizer_name:
        match optimizer_name:
            case 'gradient_descent':
                pass
            case 'adagrad':
                pass
            case 'momentum':
                if optimizer_params is None or optimizer_params.get('beta') is None:
                    print("Warning: 'beta' not found in configuration file for Momentum optimizer. Using default value of 0.9.")
            case 'rmsprop':
                if optimizer_params is None or optimizer_params.get('beta') is None:
                    print("Warning: 'beta' not found in configuration file for RMSprop optimizer. Using default value of 0.9.")
            case 'adam':
                if optimizer_params is None or optimizer_params.get('beta1') is None:
                    print("Warning: 'beta1' not found in configuration file for Adam optimizer. Using default value of 0.9.")
                if optimizer_params is None or optimizer_params.get('beta2') is None:
                    print("Warning: 'beta2' not found in configuration file for Adam optimizer. Using default value of 0.999.")
            case 'adamw':
                if optimizer_params is None or optimizer_params.get('beta1') is None:
                    print("Warning: 'beta1' not found in configuration file for AdamW optimizer. Using default value of 0.9.")
                if optimizer_params is None or optimizer_params.get('beta2') is None:
                    print("Warning: 'beta2' not found in configuration file for AdamW optimizer. Using default value of 0.999.")
                if optimizer_params is None or optimizer_params.get('weight_decay') is None:
                    print("Warning: 'weight_decay' not found in configuration file for AdamW optimizer. Using default value of 0.01.")
            case _:
                raise ValueError(f"Unknown optimizer function: {optimizer_name}")
    else:
        print("Warning: 'optimization' not found in configuration file. Using default optimization function 'gradient_descent'.")

def parse_config(config_file):
    """
    Parse and validate a neural network configuration file.

    @param config_file: The path to the JSON configuration file.
    @type  config_file: str

    @return: A dictionary containing the validated configuration parameters.
    @rtype:  dict
    """
    with open(config_file, 'r') as f:
        config = json.load(f)
        validate_layers(config)
        warning_schedule_params(config)
        warning_optimizer_params(config)
        if config.get('batch_size') is None:
            print("Warning: 'batch_size' not found in configuration file. Using default value of 8.")
        if config.get('epochs') is None:
            print("Warning: 'epochs' not found in configuration file. Using default value of 200.")
        if config.get('model_name') is None:
            print("Warning: 'model_name' not found in configuration file. Using config file name as default.")
            config['model_name'] = config_file.split('.')[0]

    return config