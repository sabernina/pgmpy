
import logging

import numpy as np
from numpy.typing import NDArray
import networkx as nx
from typing import Union

from ordered_set import OrderedSet

from pgmpy.models import BayesianNetwork, LinearGaussianBayesianNetwork
from pgmpy.factors.continuous import LinearGaussianCPD
from pgmpy.factors.discrete import TabularCPD
from pgmpy.factors.distributions import GaussianDistribution


class HybridBayesianNetwork(LinearGaussianBayesianNetwork):
    """
    A Hybrid Bayesian Network is a Bayesian Network, that contains a mixture of
    discrete and continuous nodes, where the continuous CPDs are linear Gaussians and
    the discrete nodes are Tabular CPDs.

    This process is based on the algorithm outlined in:

    Bottcher, S.. (2001). Learning Bayesian networks with mixed variables.
    Proceedings of the Eighth International Workshop on Artificial Intelligence
    and Statistics, in Proceedings of Machine Learning Research R3:13-20
    Available from https://proceedings.mlr.press/r3/bottcher01a.html. Reissued by
    PMLR on 31 March 2021.
    """
    def __init__(self, ebunch=None, latents: OrderedSet = OrderedSet(), discrete: OrderedSet = OrderedSet()):
        super(BayesianNetwork, self).__init__(ebunch=ebunch, latents=latents)
        self.discrete = discrete
        self.continuous = self.nodes - self.discrete

    def add_cpds(self, *cpds) -> None:
        """
        Add linear Gaussian CPD (Conditional Probability Distribution)
        to the Bayesian Network.

        Parameters
        ----------
        cpds  :  instances of LinearGaussianCPD
            List of LinearGaussianCPDs which will be associated with the model

        Examples
        --------
        >>> from pgmpy.models import LinearGaussianBayesianNetwork
        >>> from pgmpy.factors.continuous import LinearGaussianCPD
        >>> model = LinearGaussianBayesianNetwork([('x1', 'x2'), ('x2', 'x3')])
        >>> cpd1 = LinearGaussianCPD('x1', [1], 4)
        >>> cpd2 = LinearGaussianCPD('x2', [-5, 0.5], 4, ['x1'])
        >>> cpd3 = LinearGaussianCPD('x3', [4, -1], 3, ['x2'])
        >>> model.add_cpds(cpd1, cpd2, cpd3)
        >>> for cpd in model.cpds:
        ...     print(cpd)

        P(x1) = N(1; 4)
        P(x2| x1) = N(0.5*x1_mu); -5)
        P(x3| x2) = N(-1*x2_mu); 4)

        """
        for cpd in cpds:
            if not isinstance(cpd, LinearGaussianCPD) or not isinstance(cpd, TabularCPD):
                raise ValueError("Only LinearGaussianCPD or TabularCPD can be added.")

            if set(cpd.variables) - set(cpd.variables).intersection(set(self.nodes())):
                raise ValueError("CPD defined on variable not in the model", cpd)

            for prev_cpd_index in range(len(self.cpds)):
                if self.cpds[prev_cpd_index].variable == cpd.variable:
                    logging.warning(f"Replacing existing CPD for {cpd.variable}")
                    self.cpds[prev_cpd_index] = cpd
                    break
            else:
                self.cpds.append(cpd)

    def get_cpds(self, node=None) -> list[Union[TabularCPD, LinearGaussianCPD]]:
        """
        Returns the cpd of the node. If node is not specified returns all the CPDs
        that have been added till now to the graph

        Parameter
        ---------
        node: any hashable python object (optional)
            The node whose CPD we want. If node not specified returns all the
            CPDs added to the model.

        Returns
        -------
        A list of linear Gaussian CPDs.

        Examples
        --------
        >>> from pgmpy.models import LinearGaussianBayesianNetwork
        >>> from pgmpy.factors.continuous import LinearGaussianCPD
        >>> model = LinearGaussianBayesianNetwork([('x1', 'x2'), ('x2', 'x3')])
        >>> cpd1 = LinearGaussianCPD('x1', [1], 4)
        >>> cpd2 = LinearGaussianCPD('x2', [-5, 0.5], 4, ['x1'])
        >>> cpd3 = LinearGaussianCPD('x3', [4, -1], 3, ['x2'])
        >>> model.add_cpds(cpd1, cpd2, cpd3)
        >>> model.get_cpds()
        """
        return super(LinearGaussianBayesianNetwork, self).get_cpds(node)

    def remove_cpds(self, *cpds) -> None:
        """
        Removes the cpds that are provided in the argument.

        Parameters
        ----------
        *cpds: LinearGaussianCPD object
            A LinearGaussianCPD object on any subset of the variables
            of the model which is to be associated with the model.

        Examples
        --------
        >>> from pgmpy.models import LinearGaussianBayesianNetwork
        >>> from pgmpy.factors.continuous import LinearGaussianCPD
        >>> model = LinearGaussianBayesianNetwork([('x1', 'x2'), ('x2', 'x3')])
        >>> cpd1 = LinearGaussianCPD('x1', [1], 4)
        >>> cpd2 = LinearGaussianCPD('x2', [-5, 0.5], 4, ['x1'])
        >>> cpd3 = LinearGaussianCPD('x3', [4, -1], 3, ['x2'])
        >>> model.add_cpds(cpd1, cpd2, cpd3)
        >>> for cpd in model.get_cpds():
        ...     print(cpd)

        P(x1) = N(1; 4)
        P(x2| x1) = N(0.5*x1_mu); -5)
        P(x3| x2) = N(-1*x2_mu); 4)

        >>> model.remove_cpds(cpd2, cpd3)
        >>> for cpd in model.get_cpds():
        ...     print(cpd)

        P(x1) = N(1; 4)

        """
        return super(BayesianNetwork, self).remove_cpds(*cpds)

    def to_joint_gaussian(self):
        """
        The linear Gaussian Bayesian Networks are an alternative
        representation for the class of multivariate Gaussian distributions.
        This method returns an equivalent joint Gaussian distribution.

        Returns
        -------
        GaussianDistribution: An equivalent joint Gaussian
                                   distribution for the network.

        Reference
        ---------
        Section 7.2, Example 7.3,
        Probabilistic Graphical Models, Principles and Techniques

        Examples
        --------
        >>> from pgmpy.models import LinearGaussianBayesianNetwork
        >>> from pgmpy.factors.continuous import LinearGaussianCPD
        >>> model = LinearGaussianBayesianNetwork([('x1', 'x2'), ('x2', 'x3')])
        >>> cpd1 = LinearGaussianCPD('x1', [1], 4)
        >>> cpd2 = LinearGaussianCPD('x2', [-5, 0.5], 4, ['x1'])
        >>> cpd3 = LinearGaussianCPD('x3', [4, -1], 3, ['x2'])
        >>> model.add_cpds(cpd1, cpd2, cpd3)
        >>> jgd = model.to_joint_gaussian()
        >>> jgd.variables
        ['x1', 'x2', 'x3']
        >>> jgd.mean
        array([[ 1. ],
               [-4.5],
               [ 8.5]])
        >>> jgd.covariance
        array([[ 4.,  2., -2.],
               [ 2.,  5., -5.],
               [-2., -5.,  8.]])

        """
        if self.discrete:
            raise Exception("Cannot convert a hybrid Bayesian Network to a "
                            "joint Gaussian distribution")
        return super().to_joint_gaussian()

    def check_model(self):
        """
        Checks the model for various errors. This method checks for the following
        error -

        * Checks if the CPDs associated with nodes are consistent with their parents.

        Returns
        -------
        check: boolean
            True if all the checks pass.

        """
        for node in self.nodes():
            cpd = self.get_cpds(node=node)

            if isinstance(cpd, LinearGaussianCPD):
                if set(cpd.evidence) != set(self.get_parents(node)):
                    raise ValueError(
                        "CPD associated with %s doesn't have "
                        "proper parents associated with it." % node
                    )
        return True

    def get_cardinality(self, node):
        """
        Cardinality is not defined for continuous variables.
        """
        if node in self.discrete:
             return self.get_cpds(node).cardinality[0]
        else:
            raise ValueError("Cardinality is not defined for continuous variables.")

    def fit(
        self,
        data: pd.DataFrame,
        discrete_estimator = None,
        continuous_estimator = None,
        state_names=[],
        complete_samples_only=True,
        **kwargs
    ):
        """
        Estimate the Linear Gaussian CPD for each variable on a given data set.


        Parameters
        ----------
        data: pandas DataFrame object
            DataFrame object with column names identical to the variable names of the network.
            (If some values in the data are missing the data cells should be set to `numpy.NaN`.
            Note that pandas converts each column containing `numpy.NaN`s to dtype `float`.)
        estimator: Estimator class
            One of:
            - MaximumLikelihoodEstimator (default)
            - BayesianEstimator: In this case, pass 'prior_type' and either 'pseudo_counts'
            or 'equivalent_sample_size' as additional keyword arguments.
            See `BayesianEstimator.get_parameters()` for usage.
            - ExpectationMaximization
        state_names: dict (optional)
            A dict indicating, for each variable, the discrete set of states
            that the variable can take. If unspecified, the observed values
            in the data set are taken to be the only possible states.
        complete_samples_only: bool (default `True`)
            Specifies how to deal with missing data, if present. If set to `True` all rows
            that contain `np.Nan` somewhere are ignored. If `False` then, for each variable,
            every row where neither the variable nor its parents are `np.NaN` is used.
        n_jobs: int (default: -1)
            Number of threads/processes to use for estimation. It improves speed only
            for large networks (>100 nodes). For smaller networks might reduce
            performance.
        Returns
        -------
        Fitted Model: None
            Modifies the network inplace and adds the `cpds` property.
        Examples
        --------
        >>> import pandas as pd
        >>> from pgmpy.models import LinearGaussianBayesianNetwork
        >>> from pgmpy.estimators import MaximumLikelihoodEstimator
        >>> data = pd.DataFrame(data={'A': [0, 0, 1], 'B': [0, 1, 0], 'C': [1, 1, 0]})
        >>> model = LinearGaussianBayesianNetwork([('A', 'C'), ('B', 'C')])
        >>> model.fit(data)
        >>> model.get_cpds()
        [<LinearGaussianCPD representing P(A:2) at 0x7fb98a7d50f0>,
        <LinearGaussianCPD representing P(B:2) at 0x7fb98a7d5588>,
        <LinearGaussianCPD representing P(C:2 | A:2, B:2) at 0x7fb98a7b1f98>]
        """
        from pgmpy.estimators import BaseEstimator, MasterPriorEstimator

        if self.discrete:
            if discrete_estimator is None:
                estimator = MasterPriorEstimator
            else:
                if not issubclass(estimator, BaseEstimator):
                    raise TypeError("Estimator object should be a valid pgmpy estimator.")

            _estimator = estimator(
                self,
                data[self.discrete],
                state_names=state_names,
                complete_samples_only=complete_samples_only,
            )
            cpds_list = _estimator.get_parameters(n_jobs=n_jobs, **kwargs)
            self.add_cpds(*cpds_list)

    def predict(self, data):
        """
        For now, predict method has not been implemented for LinearGaussianBayesianNetwork.
        """
        raise NotImplementedError(
            "predict method has not been implemented for LinearGaussianBayesianNetwork."
        )

    def to_markov_model(self):
        """
        For now, to_markov_model method has not been implemented for LinearGaussianBayesianNetwork.
        """
        raise NotImplementedError(
            "to_markov_model method has not been implemented for LinearGaussianBayesianNetwork."
        )

    def is_imap(self, JPD):
        """
        For now, is_imap method has not been implemented for LinearGaussianBayesianNetwork.
        """
        raise NotImplementedError(
            "is_imap method has not been implemented for LinearGaussianBayesianNetwork."
        )
