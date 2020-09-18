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
    x_train, y_train, x_test, y_test = prepare_data(train, test, y_column, x_column, random_state=seed)

    clf = RandomForestClassifier(n_estimators=75, criterion='entropy', random_state=random_state)
    clf.fit(x_train, np.squeeze(y_train))

    cr = classification_report(y_test, clf.predict(x_test), output_dict=True, zero_division=True)
    return [cr['macro avg']['f1-score'], features, clf.feature_importances_]


f_selected, index_max = features_selection(train, test, x_column, y_column, threshold=12, random_state=seed)

print(f'\n>>> Max f1-score: {f_selected[index_max,0]}, {f_selected[index_max,1]}')
print(f'\n>>> Attempts:\n{f_selected}')



####################


from sklearn.model_selection import train_test_split
import tensorflow as tf

def create_model():
    model = tf.keras.models.Sequential(name='cubricks')
    model.add(tf.keras.layers.Input(shape=x_train.shape[1]))

    model.add(tf.keras.layers.Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l=0.001)))
    model.add(tf.keras.layers.BatchNormalization(renorm=True))
    # model.add(tf.keras.layers.Dropout(rate=0.1))

    model.add(tf.keras.layers.Dense(512, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l=0.001)))
    model.add(tf.keras.layers.BatchNormalization(renorm=True))
    # model.add(tf.keras.layers.Dropout(rate=0.1))

    model.add(tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(l=0.001)))
    model.add(tf.keras.layers.BatchNormalization(renorm=True))
    # model.add(tf.keras.layers.Dropout(rate=0.1))

    model.add(tf.keras.layers.Dense(2, activation='softmax'))
    model.summary()
    return model


train, test = split_data_month_window(df, col='DueDate', date='2020-08-01', month_window=12)
x_train, y_train, x_test, y_test = prepare_data(train, test, y_column, x_column, random_state=seed)
x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.1, shuffle=True, random_state=seed, stratify=y_train)

model = create_model()
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
              loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1), metrics=['accuracy'])

model.fit(x_train, tf.keras.utils.to_categorical(y_train),
          validation_data=(x_valid, tf.keras.utils.to_categorical(y_valid)),
          callbacks=[tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=40, restore_best_weights=True)],
          epochs=10000, batch_size=32,
          verbose=1)
          
          
y_predict = classifier_predict(model, x_test, threshold=0.9, network=True)
plot_confuncion_matrix(y_test, y_predict)
