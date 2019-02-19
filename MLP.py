import joblib
import numpy as np
from sklearn.neural_network import MLPRegressor


if __name__ == "__main__":

    flag_train = False

    X = np.load("input.npy")
    Y = np.load("target.npy")

    # normalize data
    norm_params = np.zeros((6, 2))
    for i_feature in range(X.shape[1]):
        feature_mean = np.mean(X[:, i_feature])
        norm_params[i_feature, 0] = feature_mean
        X[:, i_feature] -= feature_mean
        feature_std = np.std(X[:, i_feature])
        norm_params[i_feature, 1] = feature_std
        X[:, i_feature] /= feature_std

    num_dp = len(X)
    train_size = int(0.7*num_dp)

    X_train = X[:train_size]
    Y_train = Y[:train_size]
    X_test = X[train_size:]
    Y_test = Y[train_size:]

    mlp = MLPRegressor(hidden_layer_sizes=(50, 25, 15,),
                       activation='relu',
                       solver='adam',
                       alpha=0.0001,
                       batch_size=64,
                       learning_rate_init=0.001,
                       learning_rate='adaptive',
                       power_t=0.5,
                       max_iter=200,
                       verbose=4,
                       shuffle=True,
                       early_stopping=False,
                       tol=1e-4,
                       n_iter_no_change=10,
                       random_state=21)
    if flag_train:
        mlp.fit(X_train, Y_train)
        joblib.dump(mlp, "mlp.p")
    else:
        mlp = joblib.load("mlp.p")

    score = mlp.score(X_test, Y_test)
    print("Modellaccuracy:", score)

    with open("input.txt") as f:
        data = f.readlines()
        v_0 = float(data[0].split(":")[-1])
        phi = float(data[1].split(":")[-1])
        m = float(data[2].split(":")[-1])
        d = float(data[3].split(":")[-1])
        g = float(data[4].split(":")[-1])
        H = float(data[5].split(":")[-1])

    X_val = [[v_0, phi, m, d, g, H]]
    X_val -= norm_params[:, 0]
    X_val /= norm_params[:, 1]
    pred = mlp.predict(X_val)
    print("Vorhergesagte Zielposition: {0:.2f} [m]".format(pred[0]))