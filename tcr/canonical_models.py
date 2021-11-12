"""
Non-deep learning models
"""
import collections
from typing import *

import numpy as np

from sklearn.decomposition import PCA
from sklearn.base import BaseEstimator
from sklearn.kernel_approximation import Nystroem
from sklearn.svm import LinearSVC, SVC


class ModelOnPCA(object):
    """
    Model that runs PCA before the model itself
    PCA can be specified in integer number of components
    or a float representing the minimum explained variance
    """

    def __init__(
        self,
        model_class: BaseEstimator = SVC,
        n_components: Union[int, float] = 50,
        **model_kwargs,
    ):
        self.n_components = n_components
        self.model_class = model_class
        self.model_kwargs = model_kwargs
        self.pca = PCA(n_components=n_components)
        self.model = model_class(**model_kwargs)
        self._fitted = False

    def __str__(self) -> str:
        return f"ModelOnPCA with {self.model} on {self.n_components} PCs"

    def fit(self, X, y):
        x_pca = self.pca.fit_transform(X)
        self.model.fit(x_pca, y)
        self._fitted = True

    def score(self, X):
        assert self._fitted
        x_pca = self.pca.transform(X)
        return self.model.score(x_pca)

    def predict(self, X):
        assert self._fitted
        x_pca = self.pca.transform(X)
        return self.model.predict(x_pca)

    def predict_proba(self, X):
        assert self._fitted
        x_pca = self.pca.transform(X)
        return self.model.predict_proba(x_pca)

    def classes_(self):
        return self.model.classes_

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        retval = {
            "n_components": self.n_components,
            "model_class": self.model_class,
            **self.model_kwargs,
        }
        return retval

    def set_params(self, **params):
        """
        Set parameters of this estimator
        https://github.com/scikit-learn/scikit-learn/blob/2beed55847ee70d363bdbfe14ee4401438fba057/sklearn/base.py#L141
        """
        if not params:
            return self
        valid_params = self.get_params(deep=True)
        nested_params = collections.defaultdict(dict)
        # print(params)
        for key, value in params.items():
            key, delim, sub_key = key.partition("__")
            if delim:
                raise NotImplementedError
                # nested_params[key][sub_key] = value
            else:
                setattr(self, key, value)
                valid_params[key] = value
        # for key, sub_params in nested_params.items():
        #     print(key, sub_params)
        #     valid_params[key].set_params(**sub_params)
        return self


class SVMLike:
    """
    SVM scales poorly to larger datsets, so instead use a Nystroem transformer
    with a linear SVC
    """

    def __init__(self, kernel_ratio: float = 0.1, **kwargs) -> None:
        self.nys_kwargs = kwargs
        self.nys = Nystroem(**self.nys_kwargs)
        self.rng = np.random.default_rng(seed=64)
        self.ratio = kernel_ratio
        self.svc = LinearSVC()
        self._fitted = False

    def fit(self, X, y):
        idx_subset = self.rng.choice(
            np.arange(len(X)), size=int(self.ratio * len(X)), replace=False
        )
        X_sub = X[idx_subset]
        self.nys.fit(X_sub)

        X_trans = self.nys.transform(X)
        self.svc.fit(X_trans, y)
        self._fitted = True

    def score(self, X):
        assert self._fitted
        x_trans = self.nys.transform(X)
        return self.svc.score(x_trans)

    def predict(self, X):
        assert self._fitted
        x_trans = self.nys.transform(X)
        return self.svc.predict(x_trans)

    def predict_proba(self, X):
        raise NotImplementedError

    def classes_(self):
        return self.svc.classes_

    def get_params(self, deep: bool = True) -> Dict[str, Any]:
        retval = {
            "kernel_ratio": self.ratio,
            **self.nys_kwargs,
        }
        return retval


def main():
    from sklearn.svm import SVC

    m = SVMLike(kernel="rbf")
    print(m)


if __name__ == "__main__":
    main()
