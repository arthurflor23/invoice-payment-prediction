from sklearn.model_selection import train_test_split
import tensorflow as tf

def binary_encoding(df, cols):
    for col in cols:
        bincol = np.array([str('{0:b}'.format(x)) for x in df[col[1]].values])
        header = np.array([f'{col[1]}{i}' for i in range(col[0])])
        newcol = np.zeros((bincol.shape[0], col[0]), dtype=np.int8)

        for i in range(bincol.shape[0]):
            a = np.array(list(bincol[i]), dtype=np.int8)
            newcol[i][col[0] - len(a):] = a

        df2 = pd.DataFrame(newcol, columns=header)
        df.reset_index(drop=True, inplace=True)
        df = pd.concat([df, df2], axis=1)
        df.drop(columns=[col[1]], inplace=True)
    return df

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
