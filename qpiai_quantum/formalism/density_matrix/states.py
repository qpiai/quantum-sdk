from .density_matrix import DensityMatrix
from .exceptions import DensityMatrixException

import numpy as np
import random


class SpecialStates:
    @staticmethod
    def werner_state(seed: float) -> tuple[DensityMatrix, list[float]]:
        """
        Generate a Werner state for a given seed value

        Args:
            seed (float): The seed value for generating the Werner state
        """

        if not (0 <= seed <= 1):
            raise DensityMatrixException("Expected the seed to be between 0 and 1")

        dm = DensityMatrix(
            np.array(
                [
                    [0.5 * value for value in row]
                    for row in [
                        [1, 0, 0, 1],
                        [0, 0, 0, 0],
                        [0, 0, 0, 0],
                        [1, 0, 0, 1],
                    ]
                ]
            )
        )
        state = dm.depol(seed)

        return state, [seed]

    @staticmethod
    def isotropic_state(seed: float) -> DensityMatrix:
        """
        Generate a Isotropic state for a given seed value

        Args:
            seed (float): The seed value for generating the Isotropic state
        """
        if not (0 <= seed <= 1):
            raise DensityMatrixException("Expected the seed to be between 0 and 1")

        dm = DensityMatrix(
            np.array(
                [
                    [0.5 * value for value in row]
                    for row in [
                        [1, 0, 0, 1],
                        [0, 0, 0, 0],
                        [0, 0, 0, 0],
                        [1, 0, 0, 1],
                    ]
                ]
            )
        )
        state = dm.depol(seed)

        return state

    @staticmethod
    def mems(seed: float) -> DensityMatrix:
        """
        Generate a Maximally Entangled Mixed State (MEMS) for a given seed value

        Args:
            seed (float): The seed value for generating the MEMS
        """
        random.seed(seed)
        random_numbers = [random.random() for _ in range(4)]
        total_sum = sum(random_numbers)
        probabilities = [p / total_sum for p in random_numbers]
        p1 = probabilities[0]
        p2 = probabilities[1]
        p3 = probabilities[2]
        p4 = probabilities[3]
        phi_plus = np.array(
            [
                [0.5 * value for value in row]
                for row in [
                    [1, 0, 0, 1],
                    [0, 0, 0, 0],
                    [0, 0, 0, 0],
                    [1, 0, 0, 1],
                ]
            ]
        )
        phi_minus = np.array(
            [
                [0.5 * value for value in row]
                for row in [
                    [1, 0, 0, -1],
                    [0, 0, 0, 0],
                    [0, 0, 0, 0],
                    [-1, 0, 0, 1],
                ]
            ]
        )
        zero_one = np.array([[0, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]])
        one_zero = np.array([[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 0]])
        state = DensityMatrix(
            (p3 * phi_plus) + (p1 * phi_minus) + (p2 * zero_one) + (p4 * one_zero)
        )
        return state

    @staticmethod
    def haar_random_state(number_of_qubits: int, seed: float) -> DensityMatrix:
        """
        Generate a Haar random state for a given seed value

        Args:
            number_of_qubits (int): The number of qubits for the Haar random state
            seed (float): The seed value for generating the Haar random state
        """

        if not (0 <= seed <= 1):
            raise DensityMatrixException("Expected the seed to be between 0 and 1")

        if number_of_qubits < 1:
            raise DensityMatrixException(
                "Expected the number of qubits to be greater than 0"
            )

        n = number_of_qubits
        N = 2**n

        np.random.seed(int(seed * 1000000))

        A, B = np.random.normal(size=(N, N)), np.random.normal(size=(N, N))
        Z = A + 1j * B
        Q, R = np.linalg.qr(Z)

        Lambda = np.diag([R[i, i] / np.abs(R[i, i]) for i in range(N)])
        u = np.dot(Q, Lambda)

        dm = np.zeros((N, N), dtype=complex)
        dm[0, 0] = 1.0
        dm_prime = DensityMatrix(
            np.asarray(np.round(np.dot(np.dot(u, dm), u.conj().T), 4))
        )

        return dm_prime
