import networkx as nx
import matplotlib.pyplot as plt
import numpy as np


class GraphColoringProblem:
    """This class encapsulates the Graph Coloring problem"""

    def __init__(self, graph, hard_constraint_penalty):
        """

        :param graph: a NetworkX graph to be colored
        :param hard_constraint_penalty: penalty for hard constraint (coloring violation)
        """

        # initialize instance variables:
        self.graph = graph
        self.hard_constraint_penalty = hard_constraint_penalty

        # a list of the nodes in the graph
        self.nodelist = list(self.graph.nodes)

        # adjacency matrix of the nodes -
        # matrix[i, j] equals "1" if i and j are connected, or "0" otherwise:
        self.adj_matrix = nx.adjacency_matrix(self.graph).todense()

    def __len__(self):
        """

        :return: the number of nodes in the graph
        """
        return nx.number_of_nodes(self.graph)

    def get_cost(self, color_arrangement):
        """
        Calculates the cost of the suggested color arrangement
        :param color_arrangement: a list of integers representing the suggested color arrangement for the nodes,
        one color per node in the graph
        :return: Calculated cost of the arrangement.
        """
        return self.hard_constraint_penalty * self.get_violations_count(color_arrangement) + \
               self.get_number_of_colors(color_arrangement)

    def get_violations_count(self, color_arrangement):
        """
        Calculates the number of violations in the given color arrangement. Each pair of interconnected nodes
        with the same color counts as one violation.
        :param color_arrangement: a list of integers representing the suggested color arrangement for the nodes,
        one color per node in the graph.
        :return: the calculated violations count
        """
        if len(color_arrangement) != self.__len__():
            raise ValueError("size of color arrangement should be equal to ", self.__len__())

        violations = 0
        # iterate over every pair of nodes and find if they are adjacent AND share the same color:
        for i in range(len(color_arrangement)):
            for j in range(i + 1, len(color_arrangement)):

                if self.adj_matrix[i, j]:  # these are adjacent nodes
                    if color_arrangement[i] == color_arrangement[j]:
                        violations += 1

        return violations

    def get_number_of_colors(self, color_arrangement):
        """
        Returns the number of different color in the suggested color arrangement
        :param color_arrangement: a list of integers representing the suggested color arrangement for the nodes,
        one color per node in the graph
        :return: number of different color
        """

        return len(set(color_arrangement))

    def plot_graph(self, color_arrangement):
        """
        Plots the graph with the nodes colored according to the given color arrangement
        :param color_arrangement: a list of integers representing the suggested color arrangement for the nodes,
        one color per node in the graph
        :return: plot of the graph
        """

        if len(color_arrangement) != self.__len__():
            raise ValueError("size of color arrangement should be equal to ", self.__len__())

        # create the list of unique colors in the arrangement:
        color_list = list(set(color_arrangement))

        # create the actual colors for the integers in the color list:
        colors = plt.cm.rainbow(np.linspace(0, 1, len(color_list)))

        # iterate over the nodes, and give each one of them its corresponding color:
        color_map = []
        for i in range(self.__len__()):
            color = colors[color_list.index(color_arrangement[i])]
            color_map.append(color)

        # plot the nodes with their labels and matching colors:
        nx.draw_kamada_kawai(self.graph, node_color=color_map, with_labels=True)
        # nx.draw_circular(self.graph, node_color=color_map, with_labels=True)

        return plt


# testing the class
def main():
    # create a problem instance with petersen graph:
    gcp = GraphColoringProblem(nx.petersen_graph(), 10)

    # generate a random solution with up to 5 different colors:
    solution = np.random.randint(5, size=len(gcp))

    print("solution = ", solution)
    print("number of colors = ", gcp.get_number_of_colors(solution))
    print("Number of violations = ", gcp.get_violations_count(solution))
    print("Cost = ", gcp.get_cost(solution))

    plot = gcp.plot_graph(solution)
    plot.show()


if __name__ == "__main__":
    main()
