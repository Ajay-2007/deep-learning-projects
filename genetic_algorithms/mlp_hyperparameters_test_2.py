from sklearn import model_selection
from sklearn import datasets
from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPClassifier
from sklearn.utils._testing import ignore_warnings

from math import floor


class MLpHyperparametersTest:
    NUM_FOLDS = 5

    def __init__(self, random_seed):
        self.random_seed = random_seed
        self.initDataset()
        # with shuffle=True accuracy is 98%
        # self.kfold = model_selection.KFold(n_splits=self.NUM_FOLDS, random_state=self.random_seed, shuffle=True)
        self.kfold = model_selection.KFold(n_splits=self.NUM_FOLDS, random_state=self.random_seed, shuffle=False)

    def initDataset(self):
        self.data = datasets.load_iris()

        self.X = self.data['data']
        self.y = self.data['target']

    # params contains: [layer_1_size, layer_2_size, layer_3_size, layer_4_size]
    # params contains floats representing the following
    # 'hidden_layer_sizes' : up to 4 positive integers
    # 'activation' : {'tanh', 'relu', 'logistic'},
    # 'solver' : {'sgd', 'adam', 'lbfgs'},
    # 'alpha' : float,
    # 'learning_rate' : {'constant', 'invscaling', 'adaptive'}

    def convert_params(self, params):
        # transform the layer sizes from float (possibly negative) values into hidden_layer_sizes tuple:

        if round(params[1]) <= 0:
            hidden_layer_sizes = round(params[0])
        elif round(params[2]) <= 0:
            hidden_layer_sizes = (round(params[0]), round(params[1]))
        elif round(params[3]) <= 0:
            hidden_layer_sizes = (round(params[0]), round(params[1]), round(params[2]))
        else:
            hidden_layer_sizes = (round(params[0]), round(params[1]), round(params[2]), round(params[3]))

        activation = ['tanh', 'relu', 'logistic'][floor(params[4])]
        solver = ['sgd', 'adam', 'lbfgs'][floor(params[5])]
        alpha = params[6]
        learning_rate = ['constant', 'invscaling', 'adaptive'][floor(params[7])]

        return hidden_layer_sizes, activation, solver, alpha, learning_rate

    @ignore_warnings(category=ConvergenceWarning)
    def get_accuracy(self, params):
        hidden_layer_sizes, activation, solver, alpha, learning_rate = self.convert_params(params)

        self.classifier = MLPClassifier(random_state=self.random_seed,
                                        hidden_layer_sizes=hidden_layer_sizes,
                                        activation=activation,
                                        solver=solver,
                                        alpha=alpha,
                                        learning_rate=learning_rate)
        cv_results = model_selection.cross_val_score(self.classifier,
                                                     self.X,
                                                     self.y,
                                                     cv=self.kfold,
                                                     scoring='accuracy')

        return cv_results.mean()

    def format_params(self, params):
        hidden_layer_sizes, activation, solver, alpha, learning_rate = self.convert_params(params)
        return "hidden_layer_sizes = {}\n" \
               "activation = {}\n" \
               "solver = {}\n" \
               "alpha = {}\n" \
               "learning_rate = {}\n".format(hidden_layer_sizes,
                                           activation,
                                           solver,
                                           alpha,
                                           learning_rate)