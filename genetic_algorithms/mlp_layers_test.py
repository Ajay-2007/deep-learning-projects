from sklearn import model_selection
from sklearn import datasets
from sklearn.exceptions import ConvergenceWarning
from sklearn.neural_network import MLPClassifier
from sklearn.utils._testing import ignore_warnings


class MLpLayersTest:
    NUM_FOLDS = 5

    def __init__(self, random_seed):
        self.random_seed = random_seed
        self.initDataset()
        self.kfold = model_selection.KFold(n_splits=self.NUM_FOLDS, random_state=self.random_seed, shuffle=True)

    def initDataset(self):
        self.data = datasets.load_iris()

        self.X = self.data['data']
        self.y = self.data['target']

    # params contains: [layer_1_size, layer_2_size, layer_3_size, layer_4_size]

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

        return hidden_layer_sizes

    @ignore_warnings(category=ConvergenceWarning)
    def get_accuracy(self, params):
        hidden_layer_sizes = self.convert_params(params)

        self.classifier = MLPClassifier(random_state=self.random_seed,
                                        hidden_layer_sizes=hidden_layer_sizes)
        cv_results = model_selection.cross_val_score(self.classifier,
                                                     self.X,
                                                     self.y,
                                                     cv=self.kfold,
                                                     scoring='accuracy')

        return cv_results.mean()


    def format_params(self, params):
        return "'hidden_layer_sizes '={}".format(self.convert_params(params))

