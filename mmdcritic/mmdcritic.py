import numpy as np
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.neighbors import KNeighborsClassifier
# from dask import delayed use if potentially to parallelize
import math
import sys


class Classifier:
    model = None

    def __init__(self):
        pass

    def build_model(self, trainX, trainy):
        print("building model using %d points " % len(trainy))
        self.model = KNeighborsClassifier(n_neighbors=1)
        self.model.fit(trainX, trainy)

    def classify(self, testX, testy):

        print("classifying %d points " % len(testy))
        predy = self.model.predict(testX)

        ncorrect = np.sum(predy == testy)
        return 1.0 - ncorrect / (len(predy) + 0.0)


class Tuner:
    raise NotImplementedError


class MMDCritic:
    def __init__(self, X: np.array, gamma=0.026):
        self.X = X
        self.gamma = gamma
        self.kernel = rbf_kernel(self.X, gamma=self.gamma)
        self.selected_protos = None
        self.selected_criticism

    @classmethod
    def from_files(cls, Xpath, gamma):
        """Constructs class from .npy file
        
        Arguments:
            Xpath {str} -- Path to npy file with X 
            gamma {float} -- Gamma kernel parameter
        
        Returns:
            [cls] -- MMDCritic class
        """
        
        return cls(X, gamma)

    def select_prototypes(self, m):
        selected = MMDCritic.greedy_select_protos(
            self.kernel, np.array(range(np.shape(self.kernel)[0])), m
        )
        self.selected_protos = selected

        return selected

    def select_criticism(self, m, reg="logdet"):
        if self.selected_protos is None:
            return ValueError("there are no selected protoypes")
        selected = MMDCritic._select_criticism_regularized(
            self.kernel, self.selectedprotos, m, reg
        )
        self.selected_criticism = selected

        return selected

    @staticmethod
    def _select_criticism_regularized(
        K, selectedprotos, m, reg="logdet", is_K_sparse=True
    ):
        """
        
        Arguments:
            K {np.array} -- Kernel matrix
            selectedprotos {[type]} -- alreday selected prototypes
            m {int} -- umber of criticisms
        
        Keyword Arguments:
            reg {str} -- regularizer type (default: {'logdet'})
            is_K_sparse {bool} -- True means K is the pre-computed  csc sparse matrix? False means it is a dense matrix. (default: {True})
        
        Returns:
            [np.array] -- indices selected as criticisms
        """
        n = np.shape(K)[0]
        if reg in ["None", "logdet", "iterative"]:
            pass
        else:
            print("wrong regularizer :" + reg)
            exit(1)
        options = dict()

        selected = np.array([], dtype=int)
        candidates2 = np.setdiff1d(range(n), selectedprotos)
        inverse_of_prev_selected = None  # should be a matrix

        if is_K_sparse:
            colsum = np.array(K.sum(0)).ravel() / n
        else:
            colsum = np.sum(K, axis=0) / n

        for i in range(m):
            maxx = -sys.float_info.max
            argmax = -1
            candidates = np.setdiff1d(candidates2, selected)

            s1array = colsum[candidates]

            temp = K[selectedprotos, :][:, candidates]
            if is_K_sparse:
                s2array = temp.sum(0)
            else:
                s2array = np.sum(temp, axis=0)

            s2array = s2array / (len(selectedprotos))

            s1array = np.abs(s1array - s2array)
            if reg == "logdet":
                if (
                    inverse_of_prev_selected is not None
                ):  # first call has been made already
                    temp = K[selected, :][:, candidates]
                    if is_K_sparse:
                        temp2 = temp.transpose().dot(inverse_of_prev_selected)
                        regularizer = temp.transpose().multiply(temp2)
                        regcolsum = regularizer.sum(
                            1
                        ).ravel()  # np.sum(regularizer, axis=0)
                        regularizer = np.abs(K.diagonal()[candidates] - regcolsum)

                    else:
                        # hadamard product
                        temp2 = np.array(np.dot(inverse_of_prev_selected, temp))
                        regularizer = temp2 * temp
                        regcolsum = np.sum(regularizer, axis=0)
                        regularizer = np.log(
                            np.abs(np.diagonal(K)[candidates] - regcolsum)
                        )
                    s1array = s1array + regularizer
                else:
                    if is_K_sparse:
                        s1array = s1array - np.log(np.abs(K.diagonal()[candidates]))
                    else:
                        s1array = s1array - np.log(np.abs(np.diagonal(K)[candidates]))
            argmax = candidates[np.argmax(s1array)]
            maxx = np.max(s1array)

            selected = np.append(selected, argmax)
            if reg == "logdet":
                KK = K[selected, :][:, selected]
                if is_K_sparse:
                    KK = KK.todense()

                inverse_of_prev_selected = np.linalg.inv(KK)  # shortcut
            if reg == "iterative":
                selectedprotos = np.append(selectedprotos, argmax)

        return selected

    @staticmethod
    def _greedy_select_protos(K, candidate_indices, m, is_K_sparse=False):
        """
        
        Arguments:
            K {np.array} -- kernel matrix
            candidate_indices {np.array} -- array of potential choices for selections, returned values are chosen from these  indices
            m {int} -- number of selections to be made
        
        Keyword Arguments:
            is_K_sparse {bool} -- True means K is the pre-computed  csc sparse matrix? False means it is a dense matrix. (default: {False})
        
        Returns:
            [np.array] -- subset of candidate_indices which are selected as prototypes
        """
        if len(candidate_indices) != np.shape(K)[0]:
            K = K[:, candidate_indices][candidate_indices, :]

        n = len(candidate_indices)

        # colsum = np.array(K.sum(0)).ravel() # same as rowsum
        if is_K_sparse:
            colsum = 2 * np.array(K.sum(0)).ravel() / n
        else:
            colsum = 2 * np.sum(K, axis=0) / n

        selected = np.array([], dtype=int)
        value = np.array([])
        for i in range(m):
            maxx = -sys.float_info.max
            argmax = -1
            candidates = np.setdiff1d(range(n), selected)

            s1array = colsum[candidates]
            if len(selected) > 0:
                temp = K[selected, :][:, candidates]
                if is_K_sparse:
                    # s2array = temp.sum(0) *2
                    s2array = temp.sum(0) * 2 + K.diagonal()[candidates]

                else:
                    s2array = np.sum(temp, axis=0) * 2 + np.diagonal(K)[candidates]

                s2array = s2array / (len(selected) + 1)

                s1array = s1array - s2array

            else:
                if is_K_sparse:
                    s1array = s1array - (np.abs(K.diagonal()[candidates]))
                else:
                    s1array = s1array - (np.abs(np.diagonal(K)[candidates]))

            argmax = candidates[np.argmax(s1array)]
            # print("max %f" %np.max(s1array))

            selected = np.append(selected, argmax)
            # value = np.append(value,maxx)
            KK = K[selected, :][:, selected]
            if is_K_sparse:
                KK = KK.todense()

            inverse_of_prev_selected = np.linalg.inv(KK)  # shortcut

        return candidate_indices[selected]
