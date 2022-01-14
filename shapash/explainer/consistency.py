import itertools
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.manifold import MDS
from shapash.explainer.smart_explainer import SmartExplainer


class Consistency():

    def compile(self, x=None, model=None, preprocessing=None, contributions=None, methods=["shap", "acv", "lime"]):
        """If not provided, compute contributions according to provided methods (default are shap, acv, lime).
        If provided, check whether they respect the correct format:
        contributions = {"method_name_1": contrib_1, "method_name_2": contrib_2, ...}
        where each contrib_i is a pandas DataFrame

        Parameters
        ----------
        x : DataFrame, optional
            Dataset on which to compute consistency metrics, by default None
        model : model object, optional
            Model used to compute contributions, by default None
        preprocessing : category_encoders, ColumnTransformer, list, dict, optional (default: None)
            --> Differents types of preprocessing are available:

            - A single category_encoders (OrdinalEncoder/OnehotEncoder/BaseNEncoder/BinaryEncoder/TargetEncoder)
            - A single ColumnTransformer with scikit-learn encoding or category_encoders transformers
            - A list with multiple category_encoders with optional (dict, list of dict)
            - A list with a single ColumnTransformer with optional (dict, list of dict)
            - A dict
            - A list of dict
        contributions : dict, optional
            Contributions provided by the user if no compute is required.
            Format must be {"method_name_1": contrib_1, "method_name_2": contrib_2, ...}
            where each contrib_i is a pandas DataFrame. By default None
        methods : list
            Methods used to compute contributions, by default ["shap", "acv", "lime"]
        """
        if contributions is None:
            if (x is None) or (model is None):
                raise ValueError('If no contributions are provided, parameters "x" and "model" must be defined')
            contributions = self.compute_contributions(x, model, methods, preprocessing)
        else:
            if not isinstance(contributions, dict):
                raise ValueError('Contributions must be a dictionary')
        self.methods = list(contributions.keys())
        self.weights = list(contributions.values())

        self.check_consistency_contributions(self.weights)
        self.index = self.weights[0].index

        self.weights = [weight.values for weight in self.weights]

    def compute_contributions(self, x, model, methods, preprocessing):
        """
        Compute contributions based on specified methods

        Parameters
        ----------
        x : pandas.DataFrame
            Prediction set.
            IMPORTANT: this should be the raw prediction set, whose values are seen by the end user.
            x is a preprocessed dataset: Shapash can apply the model to it
        model : model object
            Model used to consistency check. model object can also be used by some method to compute
            predict and predict_proba values
        methods : list, optional
            When contributions is None, list of methods to use to calculate contributions, by default ["shap", "acv"]
        preprocessing : category_encoders, ColumnTransformer, list, dict
                --> Differents types of preprocessing are available:

                - A single category_encoders (OrdinalEncoder/OnehotEncoder/BaseNEncoder/BinaryEncoder/TargetEncoder)
                - A single ColumnTransformer with scikit-learn encoding or category_encoders transformers
                - A list with multiple category_encoders with optional (dict, list of dict)
                - A list with a single ColumnTransformer with optional (dict, list of dict)
                - A dict
                - A list of dict

        Returns
        -------
        contributions : dict
            Dict whose keys are method names and values are the corresponding contributions
        """
        contributions = {}
        xpl = SmartExplainer()

        for backend in methods:
            xpl.compile(
                x=x,
                model=model,
                preprocessing=preprocessing,
                backend=backend
            )
            if xpl._case == "classification" and len(xpl._classes) == 2:
                contributions[backend] = xpl.contributions[1]
            elif xpl._case == "classification" and len(xpl._classes) > 2:
                raise AssertionError("Multi-class classification is not supported")
            else:
                contributions[backend] = xpl.contributions

        return contributions

    def check_consistency_contributions(self, weights):
        """
        Assert contributions calculated from different methods are dataframes
        of same shape with same column names and index names

        Parameters
        ----------
        weights : list
            List of contributions from different methods
        """
        if weights[0].ndim == 1:
            raise ValueError('Multiple datapoints are required to compute the metric')
        if not all(isinstance(x, pd.DataFrame) for x in weights):
            raise ValueError('Contributions must be pandas DataFrames')
        if not all(x.shape == weights[0].shape for x in weights):
            raise ValueError('Contributions must be of same shape')
        if not all(x.columns.tolist() == weights[0].columns.tolist() for x in weights):
            raise ValueError('Columns names are different between contributions')
        if not all(x.index.tolist() == weights[0].index.tolist() for x in weights):
            raise ValueError('Index names are different between contributions')

    def consistency_plot(self, selection=None, max_features=20):
        """
        The Consistency_plot has the main objective of comparing explainability methods.

        Because explainability methods are different from each other,
        they may not give the same explanation to the same instance.
        Then, which method should be selected?
        Answering this question is tough. This method compares methods between them
        and evaluates how close the explanations are from each other.
        The idea behind this is pretty simple: if underlying assumptions lead to similar results,
        we would be more confident in using those methods.
        If not, careful conideration should be taken in the interpretation of the explanations

        Parameters
        ----------
        selection: list
            Contains list of index, subset of the input DataFrame that we use
            for the compute of consitency statistics, by default None
        max_features: int, optional
            Maximum number of displayed features, by default 20
        """
        # Selection
        if selection is None:
            weights = self.weights
        elif isinstance(selection, list):
            if len(selection) == 1:
                raise ValueError('Selection must include multiple points')
            else:
                weights = [weight[selection] for weight in self.weights]
        else:
            raise ValueError('Parameter selection must be a list')

        all_comparisons, mean_distances = self.calculate_all_distances(self.methods, weights)

        method_1, method_2, l2, index, backend_name_1, backend_name_2 = \
            self.find_examples(mean_distances, all_comparisons, weights)

        self.plot_comparison(mean_distances)
        self.plot_examples(method_1, method_2, l2, index, backend_name_1, backend_name_2, max_features)

    def calculate_all_distances(self, methods, weights):
        """
        For each instance, measure a distance between contributions from different methods.
        In addition, calculate the mean distance between each pair of method

        Parameters
        ----------
        methods : list
            List of methods used in the calculation of contributions
        weights : list
            List of contributions from different methods

        Returns
        -------
        all_comparisons : array
            Array containing, for each instance and each pair of methods, the distance between the contribtuions
        mean_distances : DataFrame
            DataFrame storing all pairwise distances between methods
        """
        mean_distances = pd.DataFrame(np.zeros((len(methods), len(methods))), columns=methods, index=methods)

        # Initialize a (n choose 2)x4 array (n=num of instances)
        # that will contain : indices of methods that are compared, index of instance, L2 value of instance
        all_comparisons = np.array([np.repeat(None, 4)])

        for index_i, index_j in itertools.combinations(range(len(methods)), 2):
            l2_dist = self.calculate_pairwise_distances(weights, index_i, index_j)
            # Populate the (n choose 2)x4 array
            pairwise_comparison = np.column_stack(
                (np.repeat(index_i, len(l2_dist)), np.repeat(index_j, len(l2_dist)), np.arange(len(l2_dist)), l2_dist,)
            )
            all_comparisons = np.concatenate((all_comparisons, pairwise_comparison), axis=0)

            self.calculate_mean_distances(methods, mean_distances, index_i, index_j, l2_dist)

        all_comparisons = all_comparisons[1:, :]

        return all_comparisons, mean_distances

    def calculate_pairwise_distances(self, weights, index_i, index_j):
        """
        For a specific pair of methods, calculate the distance between the contributions for all instances.

        Parameters
        ----------
        weights : list
            List of contributions from 2 selected methods
        index_i : int
            Index of method 1
        index_j : int
            Index of method 2

        Returns
        -------
        l2_dist : array
            Distance between the two selected methods for all instances
        """
        # Normalize weights using L2 norm
        norm_weights_i = weights[index_i] / np.linalg.norm(weights[index_i], ord=2, axis=1)[:, np.newaxis]
        norm_weights_j = weights[index_j] / np.linalg.norm(weights[index_j], ord=2, axis=1)[:, np.newaxis]
        # And then take the L2 norm of the difference as a metric
        l2_dist = np.linalg.norm(norm_weights_i - norm_weights_j, ord=2, axis=1)

        return l2_dist

    def calculate_mean_distances(self, methods, mean_distances, index_i, index_j, l2_dist):
        """
        Given the contributions of all instances for two selected instances, calculate the distance between them

        Parameters
        ----------
        methods : list
            List of methods used in the calculation of contributions
        mean_distances : DataFrame
            DataFrame storing all pairwise distances between methods
        index_i : int
            Index of method 1
        index_j : int
            Index of method 2
        l2_dist : array
            Distance between the two selected methods for all instances
        """
        # Calculate mean distance between the two methods and update the matrix
        mean_distances.loc[methods[index_i], methods[index_j]] = np.mean(l2_dist)
        mean_distances.loc[methods[index_j], methods[index_i]] = np.mean(l2_dist)

    def find_examples(self, mean_distances, all_comparisons, weights):
        """
        To illustrate the meaning of distances between methods, extract 5 real examples from the dataset

        Parameters
        ----------
        mean_distances : DataFrame
            DataFrame storing all pairwise distances between methods
        all_comparisons : array
            Array containing, for each instance and each pair of methods, the distance between the contribtuions
        weights : list
            List of contributions from 2 selected methods

        Returns
        -------
        method_1 : list
            Contributions of 5 instances selected to display in the second plot for method 1
        method_2 : list
            Contributions of 5 instances selected to display in the second plot for method 2
        l2 : list
            Distance between method_1 and method_2 for the 5 instances
        index : list
            Index of the selected example
        backend_name_1 : list
            Name of the explainability method displayed on the left
        backend_name_2 : list
            Name of the explainability method displayed on the right
        """
        method_1 = []
        backend_name_1 = []
        method_2 = []
        backend_name_2 = []
        index = []
        l2 = []

        # Evenly split the scale of L2 distances (from min to max excluding 0)
        for i in np.linspace(start=mean_distances[mean_distances > 0].min().min(), stop=mean_distances.max().max(), num=5):
            # For each split, find the closest existing L2 distance
            closest_l2 = all_comparisons[:, -1][np.abs(all_comparisons[:, -1] - i).argmin()]
            # Return the row that contains this L2 distance
            row = all_comparisons[all_comparisons[:, -1] == closest_l2]
            # Extract corresponding SHAP Values
            contrib_1 = weights[int(row[0, 0])][int(row[0, 2])]
            contrib_2 = weights[int(row[0, 1])][int(row[0, 2])]
            # Extract method names
            method_name_1 = self.methods[int(row[0, 0])]
            method_name_2 = self.methods[int(row[0, 1])]
            # Extract index of the selected example
            index_example = self.index[int(row[0, 2])]
            # Prevent from displaying duplicate examples
            if closest_l2 in l2:
                continue
            method_1.append(contrib_1 / np.linalg.norm(contrib_1, ord=2))
            method_2.append(contrib_2 / np.linalg.norm(contrib_2, ord=2))
            l2.append(closest_l2)
            index.append(index_example)
            backend_name_1.append(method_name_1)
            backend_name_2.append(method_name_2)

        return method_1, method_2, l2, index, backend_name_1, backend_name_2

    def calculate_coords(self, mean_distances):
        """
        Calculate 2D coords to position the different methods in the main graph

        Parameters
        ----------
        mean_distances : DataFrame
            DataFrame storing all pairwise distances between methods

        Returns
        -------
        Coordinates of each method
        """
        return MDS(n_components=2, dissimilarity="precomputed", random_state=0).fit_transform(mean_distances)

    def plot_comparison(self, mean_distances):
        """
        Plot the main graph displaying distances between methods

        Parameters
        ----------
        mean_distances : DataFrame
            DataFrame storing all pairwise distances between methods
        """
        font = {"family": "Arial", "color": '#{:02x}{:02x}{:02x}'.format(50, 50, 50)}

        fig, ax = plt.subplots(ncols=1, figsize=(10, 6))

        ax.text(x=0.5, y=1.04, s="Consistency of explanations:", fontsize=24, ha="center", transform=fig.transFigure, **font)
        ax.text(x=0.5, y=0.98, s="How similar are explanations from different methods?",
                fontsize=18, ha="center", transform=fig.transFigure, **font)

        ax.set_title(
            "Average distances between the explanations", fontsize=14, pad=-60
        )

        coords = self.calculate_coords(mean_distances)

        ax.scatter(coords[:, 0], coords[:, 1], marker="o")

        for i in range(len(mean_distances.columns)):
            ax.annotate(
                mean_distances.columns[i],
                xy=coords[i, :],
                xytext=(-5, 5),
                textcoords="offset points",
                ha="right",
                va="bottom",
            )
            self.draw_arrow(
                ax,
                coords[i, :],
                coords[(i + 1) % mean_distances.shape[0], :],
                mean_distances.iloc[i, (i + 1) % mean_distances.shape[0]],
            )

        # set gray background
        ax.set_facecolor('#F5F5F2')
        # draw solid white grid lines
        ax.grid(color='w', linestyle='solid')

        lim = (coords.min().min(), coords.max().max())
        margin = 0.1 * (lim[1] - lim[0])
        lim = (lim[0] - margin, lim[1] + margin)
        ax.set(xlim=lim, ylim=lim)
        ax.set_aspect("equal", anchor="C")
        ax.set_xticklabels([])
        ax.set_yticklabels([])
        ax.set_xticks(ax.get_yticks())

        return fig

    def draw_arrow(self, ax, a, b, dst):
        """
        Add an arrow in the main graph between the methods

        Parameters
        ----------
        ax : ax
            Input ax used for the plot
        a : array
            Coordinates of method 1
        b : array
            Coordinates of method 2
        dst : float
            Distance between the methods
        """
        ax.annotate(
            "",
            xy=a - 0.05 * (a - b),
            xycoords="data",
            xytext=b + 0.05 * (a - b),
            textcoords="data",
            arrowprops=dict(arrowstyle="<->"),
        )
        ax.annotate(
            "%.2f" % dst,
            xy=(0.5 * (a[0] + b[0]), 0.5 * (a[1] + b[1])),
            xycoords="data",
            textcoords="data",
            ha="center",
        )

    def plot_examples(self, method_1, method_2, l2, index, backend_name_1, backend_name_2, max_features):
        """
        Plot the second graph that explains distances via the use of real exmaples extracted from the dataset

        Parameters
        ----------
        method_1 : list
            Contributions of 5 instances selected to display in the second plot for method 1
        method_2 : list
            Contributions of 5 instances selected to display in the second plot for method 2
        l2 : list
            Distance between method_1 and method_2 for the 5 instances

        Returns
        -------
        figure
        """
        y = np.arange(method_1[0].shape[0])
        fig, axes = plt.subplots(ncols=len(l2), figsize=(3*len(l2), 4))
        fig.subplots_adjust(wspace=.3, top=.8)
        if len(l2) == 1:
            axes = np.array([axes])
        fig.suptitle("Examples of explanations' comparisons for various distances (L2 norm)")

        for n, (i, j, k, l, m, o) in enumerate(zip(method_1, method_2, l2, index, backend_name_1, backend_name_2)):
            # Only keep top features according to both methods
            idx = np.flip(np.abs(np.concatenate([i, j])).argsort()) % len(i)
            _, first_occurrence_idx = np.unique(idx, return_index=True)
            idx, y = idx[np.sort(first_occurrence_idx)][:max_features], y[:max_features]
            i, j = i[idx], j[idx]
            # Sort by method_1 (no abs)
            idx = np.flip(i.argsort())
            i, j = i[idx], j[idx]

            axes[n].barh(y, i, label='method 1', left=0, color='#{:02x}{:02x}{:02x}'.format(255, 166, 17))
            axes[n].barh(y, j, label='method 2', left=np.abs(np.max(i)) + np.abs(np.min(j)) + np.max(i)/3,
                         color='#{:02x}{:02x}{:02x}'.format(117, 152, 189))  # /3 to add space

            # set gray background
            axes[n].set_facecolor('#F5F5F2')
            # draw solid white grid lines
            axes[n].grid(color='w', linestyle='solid')

            axes[n].set(title="%s: %s" %
                        (self.index.name if self.index.name is not None else "Id", l) + "\n$d_{L2}$ = " + str(round(k, 2)))
            axes[n].set_xlabel("Contributions")
            axes[n].set_ylabel(f"Top {max_features} features")
            axes[n].set_xticks([0, np.abs(np.max(i)) + np.abs(np.min(j)) + np.max(i)/3])
            axes[n].set_xticklabels([m, o])
            axes[n].set_yticks([])

        return fig
