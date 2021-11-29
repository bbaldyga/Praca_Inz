def load_data_Strawberry():
    """ Method to load data as nested dataframe form Strawberry tsfile

    Returns:
        [list]: [list with nested datasets and numpay arrays]
    """
    from sktime.utils.data_io import load_from_tsfile_to_dataframe
    train_x, train_y = load_from_tsfile_to_dataframe(
        "data/Strawberry/Strawberry_TRAIN.ts"
    )
    test_x, test_y = load_from_tsfile_to_dataframe(
        "data/Strawberry/Strawberry_TEST.ts"
    )
    return [train_x, train_y], [test_x, test_y]


def load_data_Earthquakes():
    """ Method to load data as nested dataframe form Earthquakes tsfile

    Returns:
        [list]: [list with nested datasets and numpay arrays]
    """
    from sktime.utils.data_io import load_from_tsfile_to_dataframe
    train_x, train_y = load_from_tsfile_to_dataframe(
        "data/Earthquakes/Earthquakes_TRAIN.ts"
    )
    test_x, test_y = load_from_tsfile_to_dataframe(
        "data/Earthquakes/Earthquakes_TEST.ts"
    )
    return [train_x, train_y], [test_x, test_y]


def load_data_FordA():
    """ Method to load data as nested dataframe form FordA tsfile

    Returns:
        [list]: [list with nested datasets and numpay arrays]
    """
    from sktime.utils.data_io import load_from_tsfile_to_dataframe
    train_x, train_y = load_from_tsfile_to_dataframe(
        "data/FordA/FordA_TRAIN.ts"
    )
    test_x, test_y = load_from_tsfile_to_dataframe(
        "data/FordA/FordA_TEST.ts"
    )
    return [train_x, train_y], [test_x, test_y]


def load_data_Meat():
    """ Method to load data as nested dataframe form Meat tsfile

    Returns:
        [list]: [list with nested datasets and numpay arrays]
    """
    from sktime.utils.data_io import load_from_tsfile_to_dataframe
    train_x, train_y = load_from_tsfile_to_dataframe(
        "data/Meat/Meat_TRAIN.ts"
    )
    test_x, test_y = load_from_tsfile_to_dataframe(
        "data/Meat/Meat_TEST.ts"
    )
    return [train_x, train_y], [test_x, test_y]

def load_data_Crop():
    """ Method to load data as nested dataframe form Crop tsfile

    Returns:
        [list]: [list with nested datasets and numpay arrays]
    """
    from sktime.utils.data_io import load_from_tsfile_to_dataframe
    train_x, train_y = load_from_tsfile_to_dataframe(
        "data/Crop/Crop_TRAIN.ts"
    )
    test_x, test_y = load_from_tsfile_to_dataframe(
        "data/Crop/Crop_TEST.ts"
    )
    return [train_x, train_y], [test_x, test_y]
