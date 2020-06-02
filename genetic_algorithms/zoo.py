import random
from pandas import read_csv
from sklearn import model_selection
from sklearn.tree import DecisionTreeClassifier


class Zoo:
    """This class encapsulates the Friedman1 test for a regressor
    """

    DATASET_URL = 'https://archive.ics.uci.edu/ml/machine-learning-databases/zoo/zoo.data'
    NUM_FOLDS = 5

    def __init__(self, random_seed):
        """

        :param random_seed: random seed value used for reproducible results
        """
        self.random_seed = random_seed

        # read the dataset, skipping the first columns (animal name):
        self.data = read_csv(self.DATASET_URL, header=None, usecols=range(1, 18))

        # separate to input features and resulting category (last column):
        self.X = self.data.iloc[:, 0:16]
        self.y = self.data.iloc[:, 16]

        # split the data, creating a group of training/validation sets to be used in the k-fold validation process:

        self.kfold = model_selection.KFold(n_splits=self.NUM_FOLDS,random_state=self.random_seed)
        self.classifier = DecisionTreeClassifier(random_state=self.random_seed)

    def __len__(self):
        """

        :return: the total number of features used in this classification problem
        """
        return self.X.shape[1]

    def get_mean_accuracy(self, zero_one_list):
        """
        returns the mean accuracy measures of the classifier, calculated using k-fold validation process,
        using the features selected by the zero_one_list
        :param zero_one_list: a list of binary values corresponding to the features in the dataset. A value of "1"
        represents selecting the corresponding feature, while a value of "0" means that the feature is dropped.
        :return: the mean accuracy measure of the classifier when using the features selected by the zero_one_list
        """

        # drop the dataset columns that correspond to the unselected features:
        zero_indices = [i for i,n in enumerate(zero_one_list) if n == 0]
        current_x = self.X.drop(self.X.columns[zero_indices], axis=1)

        # perform k-fold validation and determine the accuracy measure of the classifier:
        cv_results = model_selection.cross_val_score(self.classifier, current_x, self.y, cv=self.kfold, scoring="accuracy")

        # return mean accuracy:
        return cv_results.mean()


# testing the class:
def main():
    # create a problem instance
    zoo = Zoo(random_seed=42)
    all_ones = [1] * len(zoo)
    print("-- All features selected: ", all_ones, ", accuracy = ", zoo.get_mean_accuracy(all_ones))

