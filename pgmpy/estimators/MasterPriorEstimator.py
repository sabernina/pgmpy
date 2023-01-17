# -*- coding: utf-8 -*-

import numbers
from itertools import chain
from warnings import warn

import numpy as np
from joblib import Parallel, delayed
import pandas as pd

from pgmpy.estimators import ParameterEstimator
from pgmpy.factors.discrete import TabularCPD
from pgmpy.models import HybridBayesianNetwork, BayesianNetwork


class MasterPriorEstimator(ParameterEstimator):
    def __init__(self, model: HybridBayesianNetwork, data: pd.DataFrame, **kwargs):
        """
        Class used to compute parameters for a hybrid model using Master Prior
        Estimation.

        Based on the Algorithm designed by Bottcher:
        Bottcher, S.. (2001). Learning Bayesian networks with mixed variables.
        Proceedings of the Eighth International Workshop on Artificial Intelligence
        and Statistics, in Proceedings of Machine Learning Research R3:13-20 Available
        from https://proceedings.mlr.press/r3/bottcher01a.html. Reissued by PMLR on 31
        March 2021.
        """
        if not issubclass(model, BayesianNetwork):
            raise NotImplementedError("Master Prior Estimation is only valid for "
                                      "Bayesian Networks")
        elif len(model.latents) != 0:
            raise ValueError(
                f"Bayesian Parameter Estimation works only on models with all observed variables. Found latent variables: {model.latents}"
            )

        super(MasterPriorEstimator, self).__init__(model, data, **kwargs)

    def get_parameters(
        self,
        prior_type: str = "bottcher",
        equivalent_sample_size=5,
        pseudo_counts=None,
        n_jobs=-1,
        weighted=False,
    ):
        """
        Method to estimate the model parameters (CPDs).

        Parameters
        ----------
        prior_type: 'bottcher', 'heckerman'
            string indicting which type of prior to use for the model parameters.
            - If 'prior_type' is 'dirichlet', the following must be provided:
                'pseudo_counts' = dirichlet hyperparameters; a single number or a dict containing, for each
                 variable, a 2-D array of the shape (node_card, product of parents_card) with a "virtual"
                 count for each variable state in the CPD, that is added to the state counts.
                 (lexicographic ordering of states assumed)
        weighted: bool
            If weighted=True, the data must contain a `_weight` column specifying the
            weight of each datapoint (row). If False, assigns an equal weight to each
            datapoint.

        Returns
        -------
        parameters: list
            List of TabularCPDs, one for each variable of the model

        Examples
        --------
        >>> import numpy as np
        >>> import pandas as pd
        >>> from pgmpy.models import BayesianNetwork
        >>> from pgmpy.estimators import BayesianEstimator
        >>> values = pd.DataFrame(np.random.randint(low=0, high=2, size=(1000, 4)),
        ...                       columns=['A', 'B', 'C', 'D'])
        >>> model = BayesianNetwork([('A', 'B'), ('C', 'B'), ('C', 'D')])
        >>> estimator = BayesianEstimator(model, values)
        >>> estimator.get_parameters(prior_type='BDeu', equivalent_sample_size=5)
        [<TabularCPD representing P(C:2) at 0x7f7b534251d0>,
        <TabularCPD representing P(B:2 | C:2, A:2) at 0x7f7b4dfd4da0>,
        <TabularCPD representing P(A:2) at 0x7f7b4dfd4fd0>,
        <TabularCPD representing P(D:2 | C:2) at 0x7f7b4df822b0>]
        """
        def _get_disc_node_param(node):
            _equivalent_sample_size = (
                equivalent_sample_size[node]
                if isinstance(equivalent_sample_size, dict)
                else equivalent_sample_size
            )
            if isinstance(pseudo_counts, numbers.Real):
                _pseudo_counts = pseudo_counts
            else:
                _pseudo_counts = pseudo_counts[node] if pseudo_counts else None

            cpd = self.estimate_cpd(
                node,
                prior_type=prior_type,
                equivalent_sample_size=_equivalent_sample_size,
                pseudo_counts=_pseudo_counts,
                weighted=weighted,
            )
            return cpd

        parameters = Parallel(n_jobs=n_jobs, prefer="threads")(
            delayed(_get_node_param)(node) for node in self.model.discrete
        )

        return parameters

    def estimate_cpd(
        self,
        node,
        prior_type: str = "bottcher",
        pseudo_counts=[],
        equivalent_sample_size=5,
        weighted=False,
    ):
        """
        Method to estimate the CPD for a given variable.

        Parameters
        ----------
        node: int, string (any hashable python object)
            The name of the variable for which the CPD is to be estimated.

        prior_type: 'dirichlet', 'BDeu', 'K2',
            string indicting which type of prior to use for the model parameters.
            - If 'prior_type' is 'dirichlet', the following must be provided:
                'pseudo_counts' = dirichlet hyperparameters; a single number or 2-D array
                 of shape (node_card, product of parents_card) with a "virtual" count for
                 each variable state in the CPD. The virtual counts are added to the
                 actual state counts found in the data. (if a list is provided, a
                 lexicographic ordering of states is assumed)
            - If 'prior_type' is 'BDeu', then an 'equivalent_sample_size'
                 must be specified instead of 'pseudo_counts'. This is equivalent to
                 'prior_type=dirichlet' and using uniform 'pseudo_counts' of
                 `equivalent_sample_size/(node_cardinality*np.prod(parents_cardinalities))`.
            - A prior_type of 'K2' is a shorthand for 'dirichlet' + setting every
              pseudo_count to 1, regardless of the cardinality of the variable.

        weighted: bool
            If weighted=True, the data must contain a `_weight` column specifying the
            weight of each datapoint (row). If False, assigns an equal weight to each
            datapoint.

        Returns
        -------
        CPD: TabularCPD
            The estimated CPD for `node`.

        Examples
        --------
        >>> import pandas as pd
        >>> from pgmpy.models import BayesianNetwork
        >>> from pgmpy.estimators import BayesianEstimator
        >>> data = pd.DataFrame(data={'A': [0, 0, 1], 'B': [0, 1, 0], 'C': [1, 1, 0]})
        >>> model = BayesianNetwork([('A', 'C'), ('B', 'C')])
        >>> estimator = BayesianEstimator(model, data)
        >>> cpd_C = estimator.estimate_cpd('C', prior_type="dirichlet",
        ...                                pseudo_counts=[[1, 1, 1, 1],
        ...                                               [2, 2, 2, 2]])
        >>> print(cpd_C)
        ╒══════╤══════╤══════╤══════╤════════════════════╕
        │ A    │ A(0) │ A(0) │ A(1) │ A(1)               │
        ├──────┼──────┼──────┼──────┼────────────────────┤
        │ B    │ B(0) │ B(1) │ B(0) │ B(1)               │
        ├──────┼──────┼──────┼──────┼────────────────────┤
        │ C(0) │ 0.25 │ 0.25 │ 0.5  │ 0.3333333333333333 │
        ├──────┼──────┼──────┼──────┼────────────────────┤
        │ C(1) │ 0.75 │ 0.75 │ 0.5  │ 0.6666666666666666 │
        ╘══════╧══════╧══════╧══════╧════════════════════╛

        """
        if node in self.model.discrete:
        node_cardinality = len(self.state_names[node])
        parents = sorted(self.model.get_parents(node))
        parents_cardinalities = [len(self.state_names[parent]) for parent in parents]
        cpd_shape = (node_cardinality, np.prod(parents_cardinalities, dtype=int))

        prior_type = prior_type.lower()

        # Throw a warning if pseudo_count is specified without prior_type=dirichlet
        #     cast to np.array first to use the array.size attribute, which returns 0 also for [[],[]]
        #     (where len([[],[]]) evaluates to 2)
        if (
            pseudo_counts is not None
            and np.array(pseudo_counts).size > 0
            and (prior_type != "dirichlet")
        ):
            warn(
                f"pseudo count specified with {prior_type} prior. It will be ignored, use dirichlet prior for specifying pseudo_counts"
            )

        if prior_type == "k2":
            pseudo_counts = np.ones(cpd_shape, dtype=int)
        elif prior_type == "bdeu":
            alpha = float(equivalent_sample_size) / (
                node_cardinality * np.prod(parents_cardinalities)
            )
            pseudo_counts = np.ones(cpd_shape, dtype=float) * alpha
        elif prior_type == "dirichlet":
            if isinstance(pseudo_counts, numbers.Real):
                pseudo_counts = np.ones(cpd_shape, dtype=int) * pseudo_counts

            else:
                pseudo_counts = np.array(pseudo_counts)
                if pseudo_counts.shape != cpd_shape:
                    raise ValueError(
                        f"The shape of pseudo_counts for the node: {node} must be of shape: {str(cpd_shape)}"
                    )
        else:
            raise ValueError("'prior_type' not specified")

        state_counts = self.state_counts(node, weighted=weighted)
        bayesian_counts = state_counts + pseudo_counts

        cpd = TabularCPD(
            node,
            node_cardinality,
            np.array(bayesian_counts),
            evidence=parents,
            evidence_card=parents_cardinalities,
            state_names={var: self.state_names[var] for var in chain([node], parents)},
        )
        cpd.normalize()
        return cpd

    def add_joint_priors(self, phi_prior: str = "bottcher") -> float:
        """
        Set up a joint prior distribution for the parameters
        """
        joint_alpha, joint_nu, joint_rho = self._add_discrete_priors()

        joint_phi, joint_mu, joint_sigma = self._add_continuous_priors(
            joint_alpha,
            phi_prior
        )

    def _add_discrete_priors(self, sample_size: int = None) -> tuple[NDArray. NDArray, NDArray]:
        if self.discrete:
            joint_prob = self._calculate_discrete_priors()

            # Determine smallest possible sample size
            min_sample_size = np.nanmin(2 / joint_prob).astype(int)
            if sample_size is None or sample_size < min_sample_size:
                sample_size = min_sample_size

            joint_alpha = joint_prob * sample_size
            joint_nu = joint_alpha
            joint_rho = joint_alpha
        else:
            joint_alpha = sample_size
            joint_nu = sample_size
            joint_rho = sample_size
        return joint_alpha, joint_nu, joint_rho

    def _calculate_discrete_priors(self):
        """
        From the discrete part of nw, the joint distribution is
        determined from the local distributions in the nodes
        """
        disc_card = [self.get_cardinality(node) for node in self.discrete]

        joint_prob = [1, ] + disc_card.values()

        for node in self.discrete.keys():
            parents = self.get_parents(node)

            # Order set by node, discrete parents, discrete non-parents
            disc_parents = parents & self.discrete
            joint_prior_nodes = OrderedSet([node,]) | disc_parents
            disc_ordered = joint_prior_nodes | self.discrete

            joint_dim = [disc_card[n] for n in disc_ordered]
            joint_dist = None  # TODO probability dist of discrete node
            joint_dist = np.reshape(joint_dist, joint_dim, "F")

            perm_idx = [disc_ordered.index(j) for j in self.discrete]
            joint_dist = joint_dist.transpose(perm_idx)

            joint_prob = joint_prob * joint_dist
        return joint_prob

    def _add_continuous_priors(self, joint_alpha: NDArray, phi_prior: str = "bottcher"):
        if self.continuous:
            n = np.prod(joint_alpha.shape)

            joint_mu = np.empty((n, len(self.continuous)))
            joint_cont = self._calculate_continuous_priors()
        else:
            return

    def _calc_continuous_priors(self):
        n = len(self.continuous)
        dim = {n: self.get_cardinality(n) for n in self.discrete.keys()}
        dim_prod = np.prod(dim.values())

        mu = np.zeros((dim_prod, n))
        sigma = np.zeros((n, n))
        prob = None

        seen = OrderedSet()
        hidden = self.continuous.copy()
        i = 0
        while hidden - seen:
            i = i % n
            node = self.continuous[i]

            if node in seen:
                continue

            local_dist = self.get_prob(node)
            parents = self.get_parents(node)

            cont_parents = parents & self.continuous
            disc_parents = parents & self.discrete

            # all continuous parents analysed
            if (cont_parents - seen):
                continue

            # calculate unconditional mu, sigma2 from node|parents
            if not cont_parents:
                M = np.arange(dim_prod).reshape(dim)
                if disc_parents:
                    m_dim = [self.get_cardinality(n) for n in disc_parents]
                    m = np.arange(dim_prod).reshape(m_dim)

                    disc_ordered = disc_parents | self.discrete
                    dist_dim = [dim[n] for n in disc_ordered]
                    M_ = np.array(m, shape=dist_dim)

                    perm_idx = [disc_ordered.index(n) for n in self.discrete]
                    M_ = M_.transpose(perm_idx)

                    for i in range(np.unique(M_).shape[0]):
                        idx = M[M_ = i]
                        # TODO
                else:
                    for i in range(dim_prod):
                        mu[i, self.nodes.index(node)] = prob
                        # TODO
            else:
                for i in range(dim_prod):
                    if disc_parents:
                        m_dim = [self.get_cardinality(n) for n in disc_parents]
                        # TODO
                    else:
                        kidx = 1
        return

