import json


def serialize_experiment(filepath, info):
    """Save experiment info as a json file

    Parameters
    ----------
    filename : str
        experiment info filename

    info : dict
        experiment info
    """
    with open(filepath, "w") as info_file:
        json.dump(info, info_file)


def update_experiment_info(filepath, key, value):
    """Update JSON experiment info

    Parameters
    ----------
    filepath : str
        experiment info filepath
    key : str
        info dict key
    value : whatever
        value to be updated
    """
    with open(filepath, "r") as info_file:
        info = json.load(info_file)

    info[key] = value

    with open(filepath, "w") as info_file:
        json.dump(info, info_file)
    

