import numpy as np
import statsmodels.api as sm
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import mean_squared_error
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso


def create_toeplitz_matrix(n, r):
    matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            matrix[i, j] = r ** abs(i - j)
    return matrix


def simulate_data(n=1000, n_training=10000):
    alpha_0, alpha_3, beta_1, beta_6 = 1, 1, 1, 4
    beta3 = 0
    beta_X_to_D = np.array([0.5, 0.6, 0.7])
    beta_X_to_Y = np.array([0.8, 0.9, 1.0])

    # Generate Z
    nc = 40  # predictor numbers
    rc = 0.8  # covariance hyper parameter
    cov_matrix = create_toeplitz_matrix(nc, rc)
    Z = np.random.multivariate_normal(np.zeros(nc), cov_matrix, size=n)
    Z_test = np.random.multivariate_normal(np.zeros(nc), cov_matrix, size=n)
    Z_training = np.random.multivariate_normal(np.zeros(nc), cov_matrix, size=n_training)

    # Generate training parameter Z
    beta_Z = np.zeros(nc)
    beta_Z[:18] = 0.05
    beta_Z[19:20] = 0
    beta_Z[21:38] = 0.025
    beta_Z[39:40] = 0

    # Generate Test Parameter Z
    beta_Z_training = beta_Z.copy()
    random_index_1 = np.random.randint(0, 20)
    random_index_11 = np.random.randint(0, 20)
    random_index_2 = np.random.randint(21, 40)
    random_index_22 = np.random.randint(21, 40)
    beta_Z_training[random_index_1] -= 0.05
    beta_Z_training[random_index_11] -= 0.05
    beta_Z_training[random_index_2] -=  0.025
    beta_Z_training[random_index_22] -=  0.025
    random_index_111 = np.random.randint(0, 20)
    random_index_1111 = np.random.randint(0, 20)
    random_index_222 = np.random.randint(21, 40)
    random_index_2222 = np.random.randint(21, 40)
    beta_Z_training[random_index_111] -= 0.05
    beta_Z_training[random_index_1111] -= 0.05
    beta_Z_training[random_index_222] -= 0.025
    beta_Z_training[random_index_2222] -= 0.025
    random_index_11111 = np.random.randint(0, 20)
    random_index_111111 = np.random.randint(0, 20)
    random_index_22222 = np.random.randint(21, 40)
    random_index_222222 = np.random.randint(21, 40)
    beta_Z_training[random_index_11111] -= 0.05
    beta_Z_training[random_index_111111] -= 0.05
    beta_Z_training[random_index_22222] -= 0.025
    beta_Z_training[random_index_222222] -= 0.025
    random_index_1111111 = np.random.randint(0, 20)
    random_index_11111111 = np.random.randint(0, 20)
    random_index_2222222 = np.random.randint(21, 40)
    random_index_22222222 = np.random.randint(21, 40)
    beta_Z_training[random_index_1111111] -= 0.05
    beta_Z_training[random_index_11111111] -= 0.05
    beta_Z_training[random_index_2222222] -= 0.025
    beta_Z_training[random_index_22222222] -= 0.025

    # Generate X
    X = np.random.multivariate_normal(np.zeros(3), np.eye(3), n)
    X_training = np.random.multivariate_normal(np.zeros(3), np.eye(3), n_training)
    X_test = np.random.multivariate_normal(np.zeros(3), np.eye(3), n)

    # Generate C
    C = np.random.normal(0, 1, n)
    C_training = np.random.normal(0, 1, n_training)
    C_test = np.random.normal(0, 1, n)

    # Generate Error Term
    u = np.random.normal(0, 1, n)
    u_training = np.random.normal(0, 1, n_training)
    u_test = np.random.normal(0, 1, n)

    # Generate D
    D = alpha_0 + np.dot(Z[:, :20], beta_Z[:20]) + np.dot(Z[:, 21:], beta_Z[21:]) + np.dot(X,beta_X_to_D) + alpha_3 * C + u
    D_test = alpha_0 + np.dot(Z_test[:, :20], beta_Z[:20]) + np.dot(Z_test[:, 21:], beta_Z[21:])+ np.dot(X_test,
                                                                                                           beta_X_to_D) + alpha_3 * C_test  + u_test
    D_training = alpha_0 + np.dot(Z_training[:, :20], beta_Z_training[:20]) + np.dot(Z_training[:, 21:], beta_Z_training[21:]) + np.dot(X_training,beta_X_to_D) + alpha_3 * C_training  + u_training

    # Generate Y
    v = np.random.normal(0, 1, n)
    Y = beta3 + beta_1 * D + np.dot(X, beta_X_to_Y)  + beta_6 * C + v
    v_training = np.random.normal(0, 1, n_training)
    Y_training = beta3 + beta_1 * D_training + np.dot(X_training, beta_X_to_Y) + beta_6 * C_training + v_training
    v_test = np.random.normal(0, 1, n)
    Y_test = beta3 + beta_1 * D_test + np.dot(X_test, beta_X_to_Y) + beta_6 * C_test + v_test
    return (Z, X, sm.add_constant(np.column_stack((Z, X))),
            Z_training, X_training, sm.add_constant(np.column_stack((Z_training, X_training))),
            Z_test, X_test, sm.add_constant(np.column_stack((Z_test, X_test))),
            D, D_training, D_test, Y, Y_training, Y_test)


def simulate_data2(n=1000, n_training=10000):
    alpha_0, alpha_3, beta_1, beta_6 = 1, 1, 1, 4
    beta3 = 0
    beta_X_to_D = np.array([0.5, 0.6, 0.7])
    beta_X_to_Y = np.array([0.8, 0.9, 1.0])

    # Generate Z
    nc = 40  # predictor numbers
    rc = 0.8  # covariance hyper parameter
    cov_matrix = create_toeplitz_matrix(nc, rc)
    Z = np.random.multivariate_normal(np.zeros(nc), cov_matrix, size=n)
    Z_test = np.random.multivariate_normal(np.zeros(nc), cov_matrix, size=n)
    Z_training = np.random.multivariate_normal(np.zeros(nc), cov_matrix, size=n_training)

    # Generate training parameter Z
    beta_Z = np.zeros(nc)
    beta_Z[:18] = 0.1
    beta_Z[19:20] = 0
    beta_Z[21:38] = 0.05
    beta_Z[39:40] = 0


    # Generate Test Parameter Z
    beta_Z_training = beta_Z.copy()
    random_index_1 = np.random.randint(0, 20)
    random_index_11 = np.random.randint(0, 20)
    random_index_2 = np.random.randint(21, 40)
    random_index_22 = np.random.randint(21, 40)
    beta_Z_training[random_index_1] -= 0.1
    beta_Z_training[random_index_11] -= 0.1
    beta_Z_training[random_index_2] -= 0.05
    beta_Z_training[random_index_22] -= 0.05
    random_index_111 = np.random.randint(0, 20)
    random_index_1111 = np.random.randint(0, 20)
    random_index_222 = np.random.randint(21, 40)
    random_index_2222 = np.random.randint(21, 40)
    beta_Z_training[random_index_111] -= 0.1
    beta_Z_training[random_index_1111] -= 0.1
    beta_Z_training[random_index_222] -= 0.05
    beta_Z_training[random_index_2222] -= 0.05
    random_index_11111 = np.random.randint(0, 20)
    random_index_111111 = np.random.randint(0, 20)
    random_index_22222 = np.random.randint(21, 40)
    random_index_222222 = np.random.randint(21, 40)
    beta_Z_training[random_index_11111] -= 0.1
    beta_Z_training[random_index_111111] -= 0.1
    beta_Z_training[random_index_22222] -= 0.05
    beta_Z_training[random_index_222222] -= 0.05

    # Generate X
    X = np.random.multivariate_normal(np.zeros(3), np.eye(3), n)
    X_training = np.random.multivariate_normal(np.zeros(3), np.eye(3), n_training)
    X_test = np.random.multivariate_normal(np.zeros(3), np.eye(3), n)

    # Generate C
    C = np.random.normal(0, 1, n)
    C_training = np.random.normal(0, 1, n_training)
    C_test = np.random.normal(0, 1, n)

    # Generate Error Term
    u = np.random.normal(0, 1, n)
    u_training = np.random.normal(0, 1, n_training)
    u_test = np.random.normal(0, 1, n)

    # Generate D
    D = alpha_0 + np.dot(Z[:, :20], beta_Z[:20]) + np.dot(Z[:, 21:], beta_Z[21:]) + np.dot(X,beta_X_to_D) + alpha_3 * C + u
    D_test = alpha_0 + np.dot(Z_test[:, :20], beta_Z[:20]) + np.dot(Z_test[:, 21:], beta_Z[21:])+ np.dot(X_test,
                                                                                                           beta_X_to_D) + alpha_3 * C_test  + u_test
    D_training = alpha_0 + np.dot(Z_training[:, :20], beta_Z_training[:20]) + np.dot(Z_training[:, 21:], beta_Z_training[21:]) + np.dot(X_training,beta_X_to_D) + alpha_3 * C_training  + u_training

    # Generate Y
    v = np.random.normal(0, 1, n)
    Y = beta3 + beta_1 * D + np.dot(X, beta_X_to_Y)  + beta_6 * C + v
    v_training = np.random.normal(0, 1, n_training)
    Y_training = beta3 + beta_1 * D_training + np.dot(X_training, beta_X_to_Y) + beta_6 * C_training + v_training
    v_test = np.random.normal(0, 1, n)
    Y_test = beta3 + beta_1 * D_test + np.dot(X_test, beta_X_to_Y) + beta_6 * C_test + v_test
    return (Z, X, sm.add_constant(np.column_stack((Z, X))),
            Z_training, X_training, sm.add_constant(np.column_stack((Z_training, X_training))),
            Z_test, X_test, sm.add_constant(np.column_stack((Z_test, X_test))),
            D, D_training, D_test, Y, Y_training, Y_test)



def simulate_data3(n=100, n_training=10000):
    alpha_0, alpha_3, beta_1, beta_6 = 1, 1, 1, 4
    beta3 = 0
    beta_X_to_D = np.array([0.5, 0.6, 0.7])
    beta_X_to_Y = np.array([0.8, 0.9, 1.0])

    # Generate Z
    nc = 40  # predictor numbers
    rc = 0.8  # covariance hyper parameter
    cov_matrix = create_toeplitz_matrix(nc, rc)
    Z = np.random.multivariate_normal(np.zeros(nc), cov_matrix, size=n)
    Z_test = np.random.multivariate_normal(np.zeros(nc), cov_matrix, size=n)
    Z_training = np.random.multivariate_normal(np.zeros(nc), cov_matrix, size=n_training)

    # Generate training parameter Z
    beta_Z = np.zeros(nc)
    beta_Z[:18] = 0.1
    beta_Z[19:20] = 0
    beta_Z[21:38] = 0.05
    beta_Z[39:40] = 0


    # Generate Test Parameter Z
    beta_Z_training = beta_Z.copy()
    random_index_1 = np.random.randint(0, 20)
    random_index_11 = np.random.randint(0, 20)
    random_index_2 = np.random.randint(21, 40)
    random_index_22 = np.random.randint(21, 40)
    beta_Z_training[random_index_1] -= 0.1
    beta_Z_training[random_index_11] -= 0.1
    beta_Z_training[random_index_2] -= 0.05
    beta_Z_training[random_index_22] -= 0.05
    random_index_111 = np.random.randint(0, 20)
    random_index_1111 = np.random.randint(0, 20)
    random_index_222 = np.random.randint(21, 40)
    random_index_2222 = np.random.randint(21, 40)
    beta_Z_training[random_index_111] -= 0.1
    beta_Z_training[random_index_1111] -= 0.1
    beta_Z_training[random_index_222] -= 0.05
    beta_Z_training[random_index_2222] -= 0.05

    # Generate X
    X = np.random.multivariate_normal(np.zeros(3), np.eye(3), n)
    X_training = np.random.multivariate_normal(np.zeros(3), np.eye(3), n_training)
    X_test = np.random.multivariate_normal(np.zeros(3), np.eye(3), n)

    # Generate C
    C = np.random.normal(0, 1, n)
    C_training = np.random.normal(0, 1, n_training)
    C_test = np.random.normal(0, 1, n)

    # Generate Error Term
    u = np.random.normal(0, 1, n)
    u_training = np.random.normal(0, 1, n_training)
    u_test = np.random.normal(0, 1, n)

    # Generate D
    D = alpha_0 + np.dot(Z[:, :20], beta_Z[:20]) + np.dot(Z[:, 21:], beta_Z[21:]) + np.dot(X,beta_X_to_D) + alpha_3 * C + u
    D_test = alpha_0 + np.dot(Z_test[:, :20], beta_Z[:20]) + np.dot(Z_test[:, 21:], beta_Z[21:])+ np.dot(X_test,
                                                                                                           beta_X_to_D) + alpha_3 * C_test  + u_test
    D_training = alpha_0 + np.dot(Z_training[:, :20], beta_Z_training[:20]) + np.dot(Z_training[:, 21:], beta_Z_training[21:]) + np.dot(X_training,beta_X_to_D) + alpha_3 * C_training  + u_training

    # Generate Y
    v = np.random.normal(0, 1, n)
    Y = beta3 + beta_1 * D + np.dot(X, beta_X_to_Y)  + beta_6 * C + v
    v_training = np.random.normal(0, 1, n_training)
    Y_training = beta3 + beta_1 * D_training + np.dot(X_training, beta_X_to_Y) + beta_6 * C_training + v_training
    v_test = np.random.normal(0, 1, n)
    Y_test = beta3 + beta_1 * D_test + np.dot(X_test, beta_X_to_Y) + beta_6 * C_test + v_test
    return (Z, X, sm.add_constant(np.column_stack((Z, X))),
            Z_training, X_training, sm.add_constant(np.column_stack((Z_training, X_training))),
            Z_test, X_test, sm.add_constant(np.column_stack((Z_test, X_test))),
            D, D_training, D_test, Y, Y_training, Y_test)


def simulate_data4(n=1000, n_training=10000):
    alpha_0, alpha_3, beta_1, beta_6 = 1, 1, 1, 4
    beta3 = 0
    beta_X_to_D = np.array([0.5, 0.6, 0.7])
    beta_X_to_Y = np.array([0.8, 0.9, 1.0])

    # Generate Z
    nc = 40  # predictor numbers
    rc = 0.8  # covariance hyper parameter
    cov_matrix = create_toeplitz_matrix(nc, rc)
    Z = np.random.multivariate_normal(np.zeros(nc), cov_matrix, size=n)
    Z_test = np.random.multivariate_normal(np.zeros(nc), cov_matrix, size=n)
    Z_training = np.random.multivariate_normal(np.zeros(nc), cov_matrix, size=n_training)

    # Generate training parameter Z
    beta_Z = np.zeros(nc)
    beta_Z[:18] = 0.1
    beta_Z[19:20] = 0
    beta_Z[21:38] = 0.05
    beta_Z[39:40] = 0

    # Generate Test Parameter Z
    beta_Z_training = beta_Z.copy()
    random_index_1 = np.random.randint(0, 20)
    random_index_11 = np.random.randint(0, 20)
    random_index_2 = np.random.randint(21, 40)
    random_index_22 = np.random.randint(21, 40)
    beta_Z_training[random_index_1] -= 0.1
    beta_Z_training[random_index_11] -= 0.1
    beta_Z_training[random_index_2] -= 0.05
    beta_Z_training[random_index_22] -= 0.05

    # Generate X
    X = np.random.multivariate_normal(np.zeros(3), np.eye(3), n)
    X_training = np.random.multivariate_normal(np.zeros(3), np.eye(3), n_training)
    X_test = np.random.multivariate_normal(np.zeros(3), np.eye(3), n)

    # Generate C
    C = np.random.normal(0, 1, n)
    C_training = np.random.normal(0, 1, n_training)
    C_test = np.random.normal(0, 1, n)

    # Generate Error Term
    u = np.random.normal(0, 1, n)
    u_training = np.random.normal(0, 1, n_training)
    u_test = np.random.normal(0, 1, n)

    # Generate D
    D = alpha_0 + np.dot(Z[:, :20], beta_Z[:20]) + np.dot(Z[:, 21:], beta_Z[21:]) + np.dot(X,beta_X_to_D) + alpha_3 * C + u
    D_test = alpha_0 + np.dot(Z_test[:, :20], beta_Z[:20]) + np.dot(Z_test[:, 21:], beta_Z[21:])+ np.dot(X_test,
                                                                                                           beta_X_to_D) + alpha_3 * C_test  + u_test
    D_training = alpha_0 + np.dot(Z_training[:, :20], beta_Z_training[:20]) + np.dot(Z_training[:, 21:], beta_Z_training[21:]) + np.dot(X_training,beta_X_to_D) + alpha_3 * C_training  + u_training

    # Generate Y
    v = np.random.normal(0, 1, n)
    Y = beta3 + beta_1 * D + np.dot(X, beta_X_to_Y)  + beta_6 * C + v
    v_training = np.random.normal(0, 1, n_training)
    Y_training = beta3 + beta_1 * D_training + np.dot(X_training, beta_X_to_Y) + beta_6 * C_training + v_training
    v_test = np.random.normal(0, 1, n)
    Y_test = beta3 + beta_1 * D_test + np.dot(X_test, beta_X_to_Y) + beta_6 * C_test + v_test
    return (Z, X, sm.add_constant(np.column_stack((Z, X))),
            Z_training, X_training, sm.add_constant(np.column_stack((Z_training, X_training))),
            Z_test, X_test, sm.add_constant(np.column_stack((Z_test, X_test))),
            D, D_training, D_test, Y, Y_training, Y_test)


def simulate_data5(n=1000, n_training=10000):
    alpha_0, alpha_3, beta_1, beta_6 = 1, 1, 1, 4
    beta3 = 0
    beta_X_to_D = np.array([0.5, 0.6, 0.7])
    beta_X_to_Y = np.array([0.8, 0.9, 1.0])

    # Generate Z
    nc = 40  # predictor numbers
    rc = 0.8  # covariance hyper parameter
    cov_matrix = create_toeplitz_matrix(nc, rc)
    Z = np.random.multivariate_normal(np.zeros(nc), cov_matrix, size=n)
    Z_test = np.random.multivariate_normal(np.zeros(nc), cov_matrix, size=n)
    Z_training = np.random.multivariate_normal(np.zeros(nc), cov_matrix, size=n_training)

    # Generate training parameter Z
    beta_Z = np.zeros(nc)
    beta_Z[:18] = 0.1
    beta_Z[19:20] = 0
    beta_Z[21:38] = 0.05
    beta_Z[39:40] = 0


    # Generate Test Parameter Z
    beta_Z_training = beta_Z.copy()

    # Generate X
    X = np.random.multivariate_normal(np.zeros(3), np.eye(3), n)
    X_training = np.random.multivariate_normal(np.zeros(3), np.eye(3), n_training)
    X_test = np.random.multivariate_normal(np.zeros(3), np.eye(3), n)

    # Generate C
    C = np.random.normal(0, 1, n)
    C_training = np.random.normal(0, 1, n_training)
    C_test = np.random.normal(0, 1, n)

    # Generate Error Term
    u = np.random.normal(0, 1, n)
    u_training = np.random.normal(0, 1, n_training)
    u_test = np.random.normal(0, 1, n)

    # Generate D
    D = alpha_0 + np.dot(Z[:, :20], beta_Z[:20]) + np.dot(Z[:, 21:], beta_Z[21:]) + np.dot(X,beta_X_to_D) + alpha_3 * C + u
    D_test = alpha_0 + np.dot(Z_test[:, :20], beta_Z[:20]) + np.dot(Z_test[:, 21:], beta_Z[21:])+ np.dot(X_test,
                                                                                                           beta_X_to_D) + alpha_3 * C_test  + u_test
    D_training = alpha_0 + np.dot(Z_training[:, :20], beta_Z_training[:20]) + np.dot(Z_training[:, 21:], beta_Z_training[21:]) + np.dot(X_training,beta_X_to_D) + alpha_3 * C_training  + u_training

    # Generate Y
    v = np.random.normal(0, 1, n)
    Y = beta3 + beta_1 * D + np.dot(X, beta_X_to_Y)  + beta_6 * C + v
    v_training = np.random.normal(0, 1, n_training)
    Y_training = beta3 + beta_1 * D_training + np.dot(X_training, beta_X_to_Y) + beta_6 * C_training + v_training
    v_test = np.random.normal(0, 1, n)
    Y_test = beta3 + beta_1 * D_test + np.dot(X_test, beta_X_to_Y) + beta_6 * C_test + v_test
    return (Z, X, sm.add_constant(np.column_stack((Z, X))),
            Z_training, X_training, sm.add_constant(np.column_stack((Z_training, X_training))),
            Z_test, X_test, sm.add_constant(np.column_stack((Z_test, X_test))),
            D, D_training, D_test, Y, Y_training, Y_test)


# Define the neural network model
def build_nn(input_dim):
    model = Sequential()
    model.add(Dense(16, input_dim=input_dim, activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(4, activation='relu'))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

def train_models(X_train, D_train):
    # OLS
    model_ols = sm.OLS(D_train, X_train).fit()
    model_lasso = Lasso(alpha=0.1)
    model_lasso.fit(X_train, D_train)
    # Neural Network
    model_nn = build_nn(X_train.shape[1])
    model_nn.fit(X_train, D_train, epochs=10, batch_size=10)

    return model_ols, model_nn,model_lasso


def target_direct(x0,y0):
    input_dim = x0.shape[1]
    output_dim = 1  # Assuming the residual is a single value prediction

    # Use the refactored function to define the model architecture for residual neural network
    model_target = build_residual_nn(input_dim, output_dim)

    # Train the residual neural network on (X^(0), R^(m->0)) pairs
    model_target.fit(x0, y0, epochs=10, batch_size=10)
    return model_target

def evaluate_models(model_ols, model_nn,model_lasso_initial,model_residual,model_oollss,model_lassoo_oollss, X, D, Y,model_nn_target):
    # Split the data
    X_C, X, D_C, D, Y_C, Y = train_test_split(X, D, Y, test_size=0.5)

    D_hat_ols = model_ols.predict(X)
    D_hat_nn_main = model_nn.predict(X).flatten()
    D_hat_lasso_main = model_lasso_initial.predict(X).flatten()
    D_hat_nn_residual = model_residual.predict(X).flatten()
    D_hat_oollss_residual =model_oollss.predict(X)
    D_hat_lassoo_residual = model_lassoo_oollss.predict(X)
    D_hat_nn = D_hat_nn_main + D_hat_nn_residual
    D_hat_nn_oollss=D_hat_nn_main+D_hat_oollss_residual
    D_hat_lassoo_oollss = D_hat_lasso_main + D_hat_lassoo_residual
    D_hat_model_target=model_nn_target.predict(X).flatten()

    alpha_hat_ols = LinearRegression().fit(D_hat_ols.reshape(-1, 1), D).coef_[0]
    alpha_hat_nn_main = LinearRegression().fit(D_hat_nn_main.reshape(-1, 1), D).coef_[0]
    alpha_hat_nn = LinearRegression().fit(D_hat_nn.reshape(-1, 1), D).coef_[0]
    alpha_hat_lassoo_oollss = LinearRegression().fit(D_hat_lassoo_oollss.reshape(-1, 1), D).coef_[0]
    alpha_hat_model_target = LinearRegression().fit(D_hat_model_target.reshape(-1, 1), D).coef_[0]

    D_hat_hat_ols = model_ols.predict(X_C)
    D_hat_hat_nn_main = model_nn.predict(X_C).flatten()
    D_hat_hat_nn_residual = model_residual.predict(X_C).flatten()
    D_hat_hat_oollss_residual =model_oollss.predict(X_C)
    D_hat_hat_nn = D_hat_hat_nn_main+D_hat_hat_nn_residual
    D_hat_hat_lassoo_oollss = D_hat_hat_nn_main+D_hat_hat_oollss_residual
    D_hat_hat_model_target = model_nn_target.predict(X_C).flatten()

    # Use alphas and D_hat_hat for OLS regression of Y on alpha*D
    def perform_direct_ols_regression(Y, D_hat, alpha):
        X_ols = sm.add_constant(alpha * D_hat)
        model_direct = sm.OLS(Y, X_ols).fit()
        return model_direct.params[1]

    beta1_direct_ols = perform_direct_ols_regression(Y_C, D_hat_hat_ols, alpha_hat_ols)
    beta1_direct_nn_main = perform_direct_ols_regression(Y_C, D_hat_hat_nn_main, alpha_hat_nn_main)
    beta1_direct_nn = perform_direct_ols_regression(Y_C, D_hat_hat_nn, alpha_hat_nn)
    beta1_direct_lassoo_oollss = perform_direct_ols_regression(Y_C, D_hat_hat_lassoo_oollss, alpha_hat_lassoo_oollss)
    beta1_direct_model_target = perform_direct_ols_regression(Y_C, D_hat_hat_model_target, alpha_hat_model_target)

    # Compile results
    results = {
        "beta1_direct_ols": beta1_direct_ols,
        "beta1_direct_nn_main": beta1_direct_nn_main,
        "beta1_direct_nn": beta1_direct_nn,
        "beta1_direct_lassoo_oollss": beta1_direct_lassoo_oollss,
        "beta1_direct_model_target": beta1_direct_model_target,

        # Compute the absolute error from the truth (assuming truth is 1)
        "error_beta1_direct_ols": abs(1 - beta1_direct_ols),
        "error_beta1_direct_nn_main": abs(1 - beta1_direct_nn_main),
        "error_beta1_direct_nn": abs(1 - beta1_direct_nn),
        "error_beta1_direct_lassoo_oollss": abs(1 - beta1_direct_lassoo_oollss),
        "error_beta1_direct_model_target": abs(1 - beta1_direct_model_target),

        # Compute MSE between D_hat_hat and true D
        "mse_D_hat_hat_ols": mean_squared_error(D_C, D_hat_hat_ols),
        "mse_D_hat_hat_nn_main": mean_squared_error(D_C, D_hat_hat_nn_main),
        "mse_D_hat_hat_nn": mean_squared_error(D_C, D_hat_hat_nn),
        "mse_D_hat_hat_lassoo_oollss": mean_squared_error(D_C, D_hat_hat_lassoo_oollss),
        "mse_D_hat_hat_model_target": mean_squared_error(D_C, D_hat_hat_model_target)
    }
    return results


def build_residual_nn(input_dim, output_dim):
    model = Sequential()
    model.add(Dense(16, input_dim=input_dim, activation='relu'))
    model.add(Dense(4, activation='relu'))
    model.add(Dense(output_dim))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

def compute_residuals(model_nn_initial,model_lasso_initial,x0,y0):
    residuals = []
    for i in range(len(x0)):
        X_i_0 = x0[i].reshape(1, -1)
        Y_i_0 = y0[i]
        # Printing is generally for debugging, consider removing for performance if not needed
#        print(x0[i])
#        print(X_i_0)
#        print(Y_i_0)
        residual = Y_i_0 - model_nn_initial.predict(X_i_0).flatten()
#        print(residual)
        residuals.append(residual)

    residuals = np.array(residuals)
    input_dim = x0.shape[1]
    output_dim = 1  # Assuming the residual is a single value prediction

    # Use the refactored function to define the model architecture for residual neural network
    model_residual = build_residual_nn(input_dim, output_dim)

    # Train the residual neural network on (X^(0), R^(m->0)) pairs
    model_residual.fit(x0, residuals, epochs=10, batch_size=10)

    # Define the Lasso regression model
    # You can specify the alpha parameter here; alpha is the regularization strength
    model_oollss = Lasso(alpha=0.1)

    # Fit the Lasso model to your data
    model_oollss.fit(x0, residuals)
    residuals = []
    for i in range(len(x0)):
        X_i_0 = x0[i].reshape(1, -1)
        Y_i_0 = y0[i]
        # Printing is generally for debugging, consider removing for performance if not needed
        #        print(x0[i])
        #        print(X_i_0)
        #        print(Y_i_0)
        residual = Y_i_0 - model_lasso_initial.predict(X_i_0).flatten()
        #        print(residual)
        residuals.append(residual)

    residuals = np.array(residuals)

    # You can specify the alpha parameter here; alpha is the regularization strength
    model_lassoo_oollss = Lasso(alpha=0.1)

    # Fit the Lasso model to your data
    model_lassoo_oollss.fit(x0, residuals)



    return model_nn_initial, model_residual, model_oollss,model_lassoo_oollss


from sklearn.utils import resample
import statsmodels.api as sm

def bootstrap_evaluate(model_ols, model_nn_initial, model_lasso_initial, model_residual, model_oollss, model_lassoo_oollss, Z, D, Y, model_nn_target, n_bootstrap=500):
    bootstrap_results = {
        "beta1_direct_ols": [],
        "beta1_direct_nn_main": [],
        "beta1_direct_nn": [],
        "beta1_direct_lassoo_oollss": [],
        "beta1_direct_model_target": []
    }

    for _ in range(n_bootstrap):
        indices = resample(range(len(Z)), replace=True)
        Z_bs, D_bs, Y_bs = Z[indices], D[indices], Y[indices]
        results_bs = evaluate_models(model_ols, model_nn_initial, model_lasso_initial, model_residual, model_oollss, model_lassoo_oollss, Z_bs, D_bs, Y_bs, model_nn_target)

        for key in bootstrap_results:
            bootstrap_results[key].append(results_bs[key])

    summary = {}
    for key in bootstrap_results:
        estimates = bootstrap_results[key]
        mean_estimate = np.mean(estimates)
        std_dev = np.sqrt(np.var(bootstrap_results[key]))
        lower_bound = mean_estimate - 1.96 * std_dev
        upper_bound = mean_estimate + 1.96 * std_dev
        covers_truth = lower_bound <= 1 <= upper_bound

        summary[key] = {"mean": mean_estimate, "CI": (lower_bound, upper_bound), "covers_truth": covers_truth}

    return summary


if __name__ == "__main__":
    num_iterations = 5
    all_results = []
    all_resultss = {
        "beta1_direct_ols": [],
        "beta1_direct_nn_main": [],
        "beta1_direct_nn": [],
        "beta1_direct_lassoo_oollss": [],
        "beta1_direct_model_target": []
    }
    all_resultsss = []
    for _ in range(num_iterations):
        Z, X, ZX_combined, Z_training, X_training, ZX_training_combined, Z_test, X_test, ZX_test_combine, D, D_training, D_test, Y, Y_training,Y_test = simulate_data()
        from sklearn.linear_model import LinearRegression


        def calculate_iv_r_squared(Z, D):
            model = LinearRegression().fit(Z, D)
            r_squared = model.score(Z, D)
            return r_squared


        # 生成数据
        Z, X, ZX_combined, Z_training, X_training, ZX_training_combined, Z_test, X_test, ZX_test_combine, D, D_training, D_test, Y, Y_training, Y_test = simulate_data()

        # 计算 R² 作为 IV 强度的指标
        r_squared_Z_D = calculate_iv_r_squared(Z, D)
        r_squared_Z_test_D_test = calculate_iv_r_squared(Z_test, D_test)
        r_squared_Z_training_D_training = calculate_iv_r_squared(Z_training, D_training)
        r_squared_ZX_combined_D = calculate_iv_r_squared(ZX_combined, D)
        r_squared_ZX_training_combined_D_training = calculate_iv_r_squared(ZX_training_combined, D_training)
        r_squared_ZX_test_combine_D_test = calculate_iv_r_squared(ZX_test_combine, D_test)

        # 打印 R² 结果
        print("R² of Z and D:", r_squared_Z_D)
        print("R² of Z_test and D_test:", r_squared_Z_test_D_test)
        print("R² of Z_training and D_training:", r_squared_Z_training_D_training)
        print("R² of ZX_combined and D:", r_squared_ZX_combined_D)
        print("R² of ZX_training_combined and D_training:", r_squared_ZX_training_combined_D_training)
        print("R² of ZX_test_combine and D_test:", r_squared_ZX_test_combine_D_test)

        model_ols, model_nn_initial,model_lasso_initial = train_models(Z_training, D_training)
        model_nn_initial, model_residual, model_oollss,model_lassoo_oollss = compute_residuals(model_nn_initial,model_lasso_initial,Z_test,D_test)
        model_nn_target=target_direct(Z_test, D_test)
        results = evaluate_models(model_ols, model_nn_initial, model_lasso_initial, model_residual, model_oollss,model_lassoo_oollss, Z, D, Y,model_nn_target)
        all_results.append(results)
        print(all_results)
        summary = bootstrap_evaluate(model_ols, model_nn_initial, model_lasso_initial, model_residual, model_oollss,model_lassoo_oollss, Z, D, Y,model_nn_target)
        print(summary)
        for key in summary:
            # 确保 key 是一个字符串
            if isinstance(key, str):
                covers_truth = 1 if summary[key]["covers_truth"] else 0
                all_resultss[key].append(covers_truth)
        print(all_resultss)

    # mean
    mean_results = {}
    for key in results.keys():
        mean_results[key] = np.mean([result[key] for result in all_results])


    # variance
    var_results = {}
    for key in results.keys():
        var_results[key] = np.var([result[key] for result in all_results])

    # 将结果写入文件
    with open("Dresults111111.txt", "w") as file:
        file.write("mean：\n")
        for key, value in mean_results.items():
            file.write(f"{key}: {value}\n")

        file.write("\nvar：\n")
        for key, value in var_results.items():
            file.write(f"{key}: {value}\n")
    # 计算 all_resultss 中每个键的平均值
    mean_coverages = {key: np.mean(values) for key, values in all_resultss.items()}

    # 将结果写入文件
    with open("DDresults111111.txt", "w") as file:
        file.write("Mean Coverages:\n")
        for key, value in mean_coverages.items():
            file.write(f"{key}: {value}\n")

    # all_results = []
    # all_resultss = {
    #     "beta1_direct_ols": [],
    #     "beta1_direct_nn_main": [],
    #     "beta1_direct_nn": [],
    #     "beta1_direct_lassoo_oollss": [],
    #     "beta1_direct_model_target": []
    # }
    # all_resultsss = []
    # for _ in range(num_iterations):
    #     Z, X, ZX_combined, Z_training, X_training, ZX_training_combined, Z_test, X_test, ZX_test_combine, D, D_training, D_test, Y, Y_training,Y_test = simulate_data2()
    #     model_ols, model_nn_initial,model_lasso_initial = train_models(Z_training, D_training)
    #     model_nn_initial, model_residual, model_oollss,model_lassoo_oollss = compute_residuals(model_nn_initial,model_lasso_initial,Z_test,D_test)
    #     model_nn_target=target_direct(Z_test, D_test)
    #     results = evaluate_models(model_ols, model_nn_initial, model_lasso_initial, model_residual, model_oollss,model_lassoo_oollss, Z, D, Y,model_nn_target)
    #     all_results.append(results)
    #     print(all_results)
    #     summary = bootstrap_evaluate(model_ols, model_nn_initial, model_lasso_initial, model_residual, model_oollss,model_lassoo_oollss, Z, D, Y,model_nn_target)
    #     print(summary)
    #     for key in summary:
    #         # 确保 key 是一个字符串
    #         if isinstance(key, str):
    #             covers_truth = 1 if summary[key]["covers_truth"] else 0
    #             all_resultss[key].append(covers_truth)
    #     print(all_resultss)
    #
    # # mean
    # mean_results = {}
    # for key in results.keys():
    #     mean_results[key] = np.mean([result[key] for result in all_results])
    #
    #
    # # variance
    # var_results = {}
    # for key in results.keys():
    #     var_results[key] = np.var([result[key] for result in all_results])
    #
    # # 将结果写入文件
    # with open("Dresults222.txt", "w") as file:
    #     file.write("mean：\n")
    #     for key, value in mean_results.items():
    #         file.write(f"{key}: {value}\n")
    #
    #     file.write("\nvar：\n")
    #     for key, value in var_results.items():
    #         file.write(f"{key}: {value}\n")
    # # 计算 all_resultss 中每个键的平均值
    # mean_coverages = {key: np.mean(values) for key, values in all_resultss.items()}
    #
    # # 将结果写入文件
    # with open("DDresults222.txt", "w") as file:
    #     file.write("Mean Coverages:\n")
    #     for key, value in mean_coverages.items():
    #         file.write(f"{key}: {value}\n")
    #
    # all_results = []
    # all_resultss = {
    #     "beta1_direct_ols": [],
    #     "beta1_direct_nn_main": [],
    #     "beta1_direct_nn": [],
    #     "beta1_direct_lassoo_oollss": [],
    #     "beta1_direct_model_target": []
    # }
    # all_resultsss = []
    # for _ in range(num_iterations):
    #     Z, X, ZX_combined, Z_training, X_training, ZX_training_combined, Z_test, X_test, ZX_test_combine, D, D_training, D_test, Y, Y_training,Y_test = simulate_data3()
    #     model_ols, model_nn_initial,model_lasso_initial = train_models(Z_training, D_training)
    #     model_nn_initial, model_residual, model_oollss,model_lassoo_oollss = compute_residuals(model_nn_initial,model_lasso_initial,Z_test,D_test)
    #     model_nn_target=target_direct(Z_test, D_test)
    #     results = evaluate_models(model_ols, model_nn_initial, model_lasso_initial, model_residual, model_oollss,model_lassoo_oollss, Z, D, Y,model_nn_target)
    #     all_results.append(results)
    #     print(all_results)
    #     summary = bootstrap_evaluate(model_ols, model_nn_initial, model_lasso_initial, model_residual, model_oollss,model_lassoo_oollss, Z, D, Y,model_nn_target)
    #     print(summary)
    #     for key in summary:
    #         # 确保 key 是一个字符串
    #         if isinstance(key, str):
    #             covers_truth = 1 if summary[key]["covers_truth"] else 0
    #             all_resultss[key].append(covers_truth)
    #     print(all_resultss)
    #
    # # mean
    # mean_results = {}
    # for key in results.keys():
    #     mean_results[key] = np.mean([result[key] for result in all_results])
    #
    #
    # # variance
    # var_results = {}
    # for key in results.keys():
    #     var_results[key] = np.var([result[key] for result in all_results])
    #
    # # 将结果写入文件
    # with open("Dresults333.txt", "w") as file:
    #     file.write("mean：\n")
    #     for key, value in mean_results.items():
    #         file.write(f"{key}: {value}\n")
    #
    #     file.write("\nvar：\n")
    #     for key, value in var_results.items():
    #         file.write(f"{key}: {value}\n")
    # # 计算 all_resultss 中每个键的平均值
    # mean_coverages = {key: np.mean(values) for key, values in all_resultss.items()}
    #
    # # 将结果写入文件
    # with open("DDresults333.txt", "w") as file:
    #     file.write("Mean Coverages:\n")
    #     for key, value in mean_coverages.items():
    #         file.write(f"{key}: {value}\n")
    #
    #
    # all_results = []
    # all_resultss = {
    #     "beta1_direct_ols": [],
    #     "beta1_direct_nn_main": [],
    #     "beta1_direct_nn": [],
    #     "beta1_direct_lassoo_oollss": [],
    #     "beta1_direct_model_target": []
    # }
    # all_resultsss = []
    # for _ in range(num_iterations):
    #     Z, X, ZX_combined, Z_training, X_training, ZX_training_combined, Z_test, X_test, ZX_test_combine, D, D_training, D_test, Y, Y_training,Y_test = simulate_data4()
    #     model_ols, model_nn_initial,model_lasso_initial = train_models(Z_training, D_training)
    #     model_nn_initial, model_residual, model_oollss,model_lassoo_oollss = compute_residuals(model_nn_initial,model_lasso_initial,Z_test,D_test)
    #     model_nn_target=target_direct(Z_test, D_test)
    #     results = evaluate_models(model_ols, model_nn_initial, model_lasso_initial, model_residual, model_oollss,model_lassoo_oollss, Z, D, Y,model_nn_target)
    #     all_results.append(results)
    #     print(all_results)
    #     summary = bootstrap_evaluate(model_ols, model_nn_initial, model_lasso_initial, model_residual, model_oollss,model_lassoo_oollss, Z, D, Y,model_nn_target)
    #     print(summary)
    #     for key in summary:
    #         # 确保 key 是一个字符串
    #         if isinstance(key, str):
    #             covers_truth = 1 if summary[key]["covers_truth"] else 0
    #             all_resultss[key].append(covers_truth)
    #     print(all_resultss)
    #
    # # mean
    # mean_results = {}
    # for key in results.keys():
    #     mean_results[key] = np.mean([result[key] for result in all_results])
    #
    #
    # # variance
    # var_results = {}
    # for key in results.keys():
    #     var_results[key] = np.var([result[key] for result in all_results])
    #
    # # 将结果写入文件
    # with open("Dresults444.txt", "w") as file:
    #     file.write("mean：\n")
    #     for key, value in mean_results.items():
    #         file.write(f"{key}: {value}\n")
    #
    #     file.write("\nvar：\n")
    #     for key, value in var_results.items():
    #         file.write(f"{key}: {value}\n")
    # # 计算 all_resultss 中每个键的平均值
    # mean_coverages = {key: np.mean(values) for key, values in all_resultss.items()}
    #
    # # 将结果写入文件
    # with open("DDresults444.txt", "w") as file:
    #     file.write("Mean Coverages:\n")
    #     for key, value in mean_coverages.items():
    #         file.write(f"{key}: {value}\n")
    #
    #
    # all_results = []
    # all_resultss = {
    #     "beta1_direct_ols": [],
    #     "beta1_direct_nn_main": [],
    #     "beta1_direct_nn": [],
    #     "beta1_direct_lassoo_oollss": [],
    #     "beta1_direct_model_target": []
    # }
    # all_resultsss = []
    # for _ in range(num_iterations):
    #     Z, X, ZX_combined, Z_training, X_training, ZX_training_combined, Z_test, X_test, ZX_test_combine, D, D_training, D_test, Y, Y_training,Y_test = simulate_data5()
    #     model_ols, model_nn_initial,model_lasso_initial = train_models(Z_training, D_training)
    #     model_nn_initial, model_residual, model_oollss,model_lassoo_oollss = compute_residuals(model_nn_initial,model_lasso_initial,Z_test,D_test)
    #     model_nn_target=target_direct(Z_test, D_test)
    #     results = evaluate_models(model_ols, model_nn_initial, model_lasso_initial, model_residual, model_oollss,model_lassoo_oollss, Z, D, Y,model_nn_target)
    #     all_results.append(results)
    #     print(all_results)
    #     summary = bootstrap_evaluate(model_ols, model_nn_initial, model_lasso_initial, model_residual, model_oollss,model_lassoo_oollss, Z, D, Y,model_nn_target)
    #     print(summary)
    #     for key in summary:
    #         # 确保 key 是一个字符串
    #         if isinstance(key, str):
    #             covers_truth = 1 if summary[key]["covers_truth"] else 0
    #             all_resultss[key].append(covers_truth)
    #     print(all_resultss)
    #
    # # mean
    # mean_results = {}
    # for key in results.keys():
    #     mean_results[key] = np.mean([result[key] for result in all_results])
    #
    #
    # # variance
    # var_results = {}
    # for key in results.keys():
    #     var_results[key] = np.var([result[key] for result in all_results])
    #
    # # 将结果写入文件
    # with open("Dresults555.txt", "w") as file:
    #     file.write("mean：\n")
    #     for key, value in mean_results.items():
    #         file.write(f"{key}: {value}\n")
    #
    #     file.write("\nvar：\n")
    #     for key, value in var_results.items():
    #         file.write(f"{key}: {value}\n")
    # # 计算 all_resultss 中每个键的平均值
    # mean_coverages = {key: np.mean(values) for key, values in all_resultss.items()}
    #
    # # 将结果写入文件
    # with open("DDresults555.txt", "w") as file:
    #     file.write("Mean Coverages:\n")
    #     for key, value in mean_coverages.items():
    #         file.write(f"{key}: {value}\n")
    #
    #
