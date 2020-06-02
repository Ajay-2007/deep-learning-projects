import numpy as np

from sklearn import model_selection
from sklearn import datasets

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns


class FriedmanTest:
    """This class encapsulates the Friedman1 regression test for feature selection
    """

    VALIDATION_SIZE = 0.20
    NOISE = 1.0

    def __init__(self, num_features, num_samples, random_seed):
        """

        :param num_features: total number of features to be used ( at least 5)
        :param num_samples:  number of samples in dataset
        :param random_seed:  random seed value used for reproducible results
        """
        self.num_features = num_features
        self.num_samples = num_samples
        self.random_seed = random_seed

        # generate test data:
        self.X, self.y = datasets.make_friedman1(n_samples=self.num_samples,
                                                 n_features=self.num_features,
                                                 noise=self.NOISE,
                                                 random_state=self.random_seed)

        # divide the data to a training set and a validation set:
        self.X_train, self.X_validation, self.y_train, self.y_validation = \
            model_selection.train_test_split(self.X, self.y, test_size=self.VALIDATION_SIZE,
                                             random_state=self.random_seed)

        self.regressor = GradientBoostingRegressor(random_state=self.random_seed)

    def __len__(self):
        """

        :return: the total number of features
        """
        return self.num_features


    def get_mse(self, zero_one_list):
        """
        returns the mean squared error of the regressor, calculates for the validation set, after training
        using the features selected by the zero_one_list
        :param zero_one_list: a list of binary values corresponding the features in the dataset. A value of "1"
        represents selecting the corresponding feature, while a value of "0" means that the feature is dropped.
        :return: the mean squared error of the regressor when using the features selected by zero_one_list
        """

        # drop the columns of the training and validation sets that correspond to the unselected features:
        zero_indices = [i for i, n in enumerate(zero_one_list) if n == 0]
        current_x_train = np.delete(self.X_train, zero_indices, 1)
        current_x_validation = np.delete(self.X_validation, zero_indices, 1)

        # train the regressor model using the training set:
        self.regressor.fit(current_x_train, self.y_train)

        # calculate the regressor"s output for the validation set:
        prediction = self.regressor.predict(current_x_validation)

        # return the mean square error of prediction vs actual data:
        return mean_squared_error(self.y_validation, prediction)


# testing the class:
def main():
    # create a test instance
    test = FriedmanTest(num_features=15, num_samples=60, random_seed=42)

    scores = []

    # calculates MSE for "n" first features:
    for n in range(1, len(test)+1):
        n_first_features = [1] * n + [0] * (len(test) - n)
        score = test.get_mse(n_first_features)
        print("%d first features: score = %f"% (n, score))
        scores.append(score)


    # plot the graph:
    sns.set_style("whitegrid")
    plt.plot([i+1 for i in range(len(test))], scores, color="red")
    plt.xticks(np.arange(1, len(test) + 1, 1.0))
    plt.xlabel("n First Features")
    plt.ylabel("MSE")
    plt.title("MSE over Features Selected")
    plt.show()

if __name__ == "__main__":
    main()