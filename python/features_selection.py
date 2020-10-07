def features_selection(train, test, x_column, y_column, threshold=1, random_state=None):
    from itertools import combinations
    import multiprocessing
    import functools

    total = len(x_column)
    predicts = []

    for i in range(total, threshold-1, -1):
        print(f'>>>\r{round((1-((i-threshold)/total)) * 100, 1)}% complete', end='')
        cb = sum([list(map(list, combinations(x_column, y))) for y in range(i, i + 1)], [])

        with multiprocessing.Pool(multiprocessing.cpu_count()) as pool:
            pd = pool.map(functools.partial(_selection, train, test, x_column, y_column, random_state), cb)
            pd = np.array(pd, dtype=object)
            pool.close()
            pool.join()

        if i == total:
            x_column = x_column[np.argsort(pd[:,2][0])[::-1]]

        index_max = np.argmax(pd[:,0])
        index_del = [i for i, item in enumerate(x_column) if item not in pd[index_max][1]]

        predicts.append(pd[index_max])
        x_column = np.delete(x_column, index_del)

    predicts = np.array(predicts, dtype=object)
    predicts = predicts[predicts[:,0].argsort()][::-1]
    index_max = np.argmax(predicts[:,0])
    return (predicts, index_max)

def _selection(*args):
    train, test, x_column, y_column, random_state, features = args

    train = resample(train, x_column, y_column, SMOTE(sampling_strategy=0.75, random_state=random_state))
    x_train, y_train, x_test, y_test = prepare_data(train, test, x_column, y_column, random_state=random_state)

    clf = RandomForestClassifier(n_estimators=100, criterion='entropy', class_weight='balanced', random_state=random_state)
    clf.fit(x_train, np.squeeze(y_train))

    cr = classification_report(y_test, clf.predict(x_test), output_dict=True, zero_division=True)
    return [cr['macro avg']['f1-score'], features, clf.feature_importances_]


f_selected, index_max = features_selection(train, test, x_column, y_column, threshold=12, random_state=SEED)

print(f'\n>>> Max f1-score: {f_selected[index_max,0]}, {f_selected[index_max,1]}')
print(f'\n>>> Attempts:\n{f_selected}')
