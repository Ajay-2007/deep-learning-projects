import numpy as np
import random
import time

from sklearn import model_selection
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import GridSearchCV
from evolutionary_search import EvolutionaryAlgorithmSearchCV
from pandas import read_csv


class HyperparameterTuningGrid:
    NUM_FOLDS = 5

    def __init__(self, random_seed):
        self.random_seed = random_seed
        self.init_wine_dataset()
        self.init_classifier()
        self.init_kfold()
        self.init_grid_params()

    def init_wine_dataset(self):
        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data'
        self.data = read_csv(url, header=None, usecols=range(0, 14))
        self.X = self.data.iloc[:, 1:14]
        self.y = self.data.iloc[:, 0]

    def init_classifier(self):
        self.classifier = AdaBoostClassifier(random_state=self.random_seed)

    def init_kfold(self):
        self.kfold = model_selection.KFold(n_splits=self.NUM_FOLDS,
                                           random_state=self.random_seed)

    def init_grid_params(self):
        self.grid_params = {
            "n_estimators": [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
            "learning_rate": np.logspace(-2, 0, num=10, base=10),
            "algorithm": ["SAMME", "SAMME.R"],
        }

    def get_default_accuracy(self):
        cv_results = model_selection.cross_val_score(self.classifier,
                                                     self.X,
                                                     self.y,
                                                     cv=self.kfold,
                                                     scoring="accuracy")

        return cv_results.mean()

    def grid_test(self):
        print("performing grid search...")
        grid_search = GridSearchCV(estimator=self.classifier,
                                   param_grid=self.grid_params,
                                   cv=self.kfold,
                                   scoring="accuracy",
                                   iid="False",
                                   n_jobs=4)

        grid_search.fit(self.X, self.y)
        print("best parameters: ", grid_search.best_params_)
        print("best score: ", grid_search.best_score_)

    def genetic_grid_test(self):
        print("performing Genetic grid search...")
        grid_search = EvolutionaryAlgorithmSearchCV(estimator=self.classifier,
                                                    params=self.grid_params,
                                                    scoring="accuracy",
                                                    cv=self.kfold,
                                                    verbose=True,
                                                    iid="False",
                                                    n_jobs=4,
                                                    population_size=20,
                                                    gene_mutation_prob=0.30,
                                                    tournament_size=2,
                                                    generations_number=5)

        grid_search.fit(self.X, self.y)


def main():
    RANDOM_SEED = 42
    random.seed(RANDOM_SEED)

    # create a problem instance
    test = HyperparameterTuningGrid(RANDOM_SEED)
    print("Default Classifier Hyperparameter values:")
    print(test.classifier.get_params())
    print("score with default values = ", test.get_default_accuracy())

    print()
    start = time.time()
    test.grid_test()
    end = time.time()
    print("Time Elapsed = ", end - start)
    print()
    start = time.time()
    test.genetic_grid_test()
    end = time.time()
    print("Time Elapsed = ", end - start)


if __name__ == "__main__":
    main()
