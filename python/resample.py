def resample(df, x_column, y_column, func):
    dtypes = df[x_column].dtypes.to_dict()
    dtypes.update(df[y_column].dtypes.to_dict())    
    x, y = df[x_column].values, df[y_column].values

    try:
        x, y = func.fit_resample(x, y)
        y = np.expand_dims(y, axis=1)
    except:
        pass

    xy = np.concatenate((x, y), axis=1)
    data = pd.DataFrame(xy, columns=np.concatenate((x_column, y_column)))
    data = data.astype(dtypes)
    return data
