def KNClassification(train_x,train_y, test_x, test_y):
    """ Classification useing K Nearest Neighbors classifier form scilearn

    Args:
        train_x (pandas DataType nested): nested datatype of train time series 
        train_y (numpy array): array of classes in train series 
        test_x (pandas DataType nested): nested datatype of test time series 
        test_y (numpy array): array of classes in test series

    Returns:
        list: return four element list with predicts, score, amount guessed predicts and time in which function run
    """    ''''''
    from sklearn.neighbors import KNeighborsClassifier
    from sktime.datatypes._panel._convert import from_nested_to_2d_array
    from sklearn.metrics import accuracy_score
    import time
    clf = KNeighborsClassifier()
    train_x = from_nested_to_2d_array(train_x)
    test_x = from_nested_to_2d_array(test_x)
    start_time = time.time()
    clf.fit(train_x ,train_y.ravel())
    predict = clf.predict(test_x)
    score = accuracy_score(test_y.ravel(), predict)
    score_amount = accuracy_score(test_y.ravel(), predict, normalize=False)
    result_time = time.time() - start_time
    return [predict,score, score_amount, result_time]

def TSSClasifier(train_x,train_y, test_x, test_y):
    """Method that classify datasets using time series classifier form scilearn

    Args:
        train_x (pandas DataType nested): nested datatype of train time series 
        train_y (numpy array): array of classes in train series 
        test_x (pandas DataType nested): nested datatype of test time series 
        test_y (numpy array): array of classes in test series

    Returns:
        list: return four element list with predicts, score, amount guessed predicts and time in which function run
    """    ''''''
    from sklearn.ensemble import RandomForestClassifier
    from sktime.datatypes._panel._convert import from_nested_to_2d_array
    from sklearn.metrics import accuracy_score
    import time
    clf = RandomForestClassifier()
    train_x = from_nested_to_2d_array(train_x)
    test_x = from_nested_to_2d_array(test_x)
    start_time = time.time()
    clf.fit(train_x ,train_y.ravel())
    predict = clf.predict(test_x)
    score = accuracy_score(test_y.ravel(), predict)
    score_amount = accuracy_score(test_y.ravel(), predict, normalize=False)
    result_time = time.time() - start_time
    return [predict,score, score_amount, result_time]

def RTSSClasifier(train_x,train_y, test_x, test_y):
    """Method that classify datasets using composable time series classifier form skTime

    Args:
        train_x (pandas DataType nested): nested datatype of train time series 
        train_y (numpy array): array of classes in train series 
        test_x (pandas DataType nested): nested datatype of test time series 
        test_y (numpy array): array of classes in test series

    Returns:
        list: return four element list with predicts, score, amount guessed predicts and time in which function run
    """    ''''''
    from sktime.classification.compose import ComposableTimeSeriesForestClassifier
    from sklearn.metrics import accuracy_score
    import time
    clf = ComposableTimeSeriesForestClassifier()
    start_time = time.time()
    clf.fit(train_x ,train_y)
    predict = clf.predict(test_x)
    score = accuracy_score(test_y, predict)
    score_amount = accuracy_score(test_y.ravel(), predict, normalize=False)
    result_time = time.time() - start_time
    return [predict,score, score_amount, result_time]

def BOSSClasifier(train_x,train_y, test_x, test_y):
    """Method that classify datasets using bag of SFA-Sybmols in vector space from pyts library

    Args:
        train_x (pandas DataType nested): nested datatype of train time series 
        train_y (numpy array): array of classes in train series 
        test_x (pandas DataType nested): nested datatype of test time series 
        test_y (numpy array): array of classes in test series

    Returns:
        list: return four element list with predicts, score, amount guessed predicts and time in which function run
    """    ''''''
    from pyts.classification import BOSSVS
    from sktime.datatypes._panel._convert import from_nested_to_2d_array
    from sklearn.metrics import accuracy_score
    import time
    clf = BOSSVS(n_bins=15, strategy='normal')
    train_x = from_nested_to_2d_array(train_x)
    test_x = from_nested_to_2d_array(test_x)
    start_time = time.time()
    clf.fit(train_x ,train_y.ravel())
    predict = clf.predict(test_x)
    score = accuracy_score(test_y.ravel(), predict)
    score_amount = accuracy_score(test_y.ravel(), predict, normalize=False)
    result_time = time.time() - start_time
    return [predict,score, score_amount, result_time]

def SAXClasifier(train_x,train_y, test_x, test_y):
    """Method that classify datasets using SAX and vector space model algorithm from pyts library

    Args:
        train_x (pandas DataType nested): nested datatype of train time series 
        train_y (numpy array): array of classes in train series 
        test_x (pandas DataType nested): nested datatype of test time series 
        test_y (numpy array): array of classes in test series

    Returns:
        list: return four element list with predicts, score, amount guessed predicts and time in which function run
    """    ''''''
    from pyts.classification import SAXVSM
    from sktime.datatypes._panel._convert import from_nested_to_2d_array
    from sklearn.metrics import accuracy_score
    import time
    import numpy as np
    clf = SAXVSM(strategy='normal')
    train_x = from_nested_to_2d_array(train_x)
    test_x = from_nested_to_2d_array(test_x)
    start_time = time.time()
    clf.fit(train_x ,train_y.ravel())
    predict = clf.predict(test_x)
    score = accuracy_score(test_y.ravel(), predict)
    score_amount = accuracy_score(test_y.ravel(), predict, normalize=False)
    result_time = time.time() - start_time
    return [predict,score, score_amount, result_time]
