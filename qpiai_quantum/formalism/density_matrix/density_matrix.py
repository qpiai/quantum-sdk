from typing import List, Union
import numpy as np
import scipy.linalg as splinalg
import scipy.sparse as spsparce
from cvxpy.expressions.expression import Expression
import cvxpy as cp
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from .exceptions import DensityMatrixException
from .base_density_matrix import BaseDensityMatrix


class DensityMatrix(BaseDensityMatrix):
    """
    The density matrix formalism for creating and manipulating quantum states using density matrices.
    """

    def __init__(self, state: Union[List[List[complex]], np.ndarray]):
        """
        Creates a DensityMatrix object for a given quantum state as numpy array or list.

        Args:
            state (List[List[complex]]): The quantum state as array
        """
        if isinstance(state, np.ndarray):
            state = state

        elif isinstance(state, List):
            state = np.array(state)

        else:
            raise DensityMatrixException(
                "Cannot understand the type of the input state. Use np.array or list"
            )

        _state: np.ndarray = state
        rows, cols = _state.shape

        if rows == cols:
            pass
        else:
            _state = self._convert_state_vector_to_dm(_state)

        # assign the state as the density matrix
        self.state: np.ndarray = _state

    # Implement abstract methods from BaseDensityMatrix
    def _get_data(self) -> np.ndarray:
        """Return the underlying density matrix data."""
        return self.state

    def _get_num_qubits(self) -> int:
        """Return the number of qubits."""
        dim = self.state.shape[0]
        return int(np.log2(dim))

    @property
    def state(self) -> np.array:
        """
        Returns the density matrix
        """

        return self._state

    @state.setter
    def state(self, state: np.array):
        """
        Sets the density matrix

        Args:
            state (np.array): The density matrix to be set
        """

        if isinstance(state, np.ndarray):
            state = state

        elif isinstance(state, List):
            state = np.array(state)

        else:
            raise DensityMatrixException(
                "Cannot understand the type of the input state. Use np.array or list"
            )

        rows, cols = state.shape
        if rows == cols:
            pass
        else:
            state = self._convert_state_vector_to_dm(state)

        # assign the state as the density matrix
        self._state: np.array = state

    def get_state(self) -> np.array:
        """
        Returns the density matrix
        """

        return self.state

    def _convert_state_vector_to_dm(self, state: np.array):
        """
        Converts a given statevector into a density matrix
        """

        rows, cols = state.shape

        if cols != 1:
            raise DensityMatrixException(
                "Expected the state vector to be a column vector"
            )

        return np.matmul(state, state.conj().T)

    def show(self):
        """
        Prints the desity matrix into the console
        """

        print(self.state)

    @staticmethod
    def check(densityMatrix: "DensityMatrix", verbose: bool = False) -> bool:
        """
        Validate if the given desity matrix is valid

        Args:
            densityMatrix (DensityMatrix): The density matrix to be validated
        """

        state = densityMatrix.state

        trace_check = (np.trace(state) < 1.000001) and (np.trace(state) > 0.999999)
        positive_semi_definiteness = not (np.all(np.linalg.eigvals(state) >= 0.000001))
        hermitian_check = splinalg.ishermitian(state)

        if verbose:
            (
                print("Trace check passed")
                if trace_check
                else print("Trace check failed")
            )
            (
                print("Positive semi-definiteness passed")
                if positive_semi_definiteness
                else print("Positive semi-definiteness test failed")
            )
            (
                print("Hermitian check passed")
                if hermitian_check
                else print("Hermitian check failed")
            )

        return trace_check and positive_semi_definiteness and hermitian_check

    def ADC(self, param: float):
        """ """

        # works only for 2 qubit states
        if self.state.shape != (4, 4):
            raise DensityMatrixException("This method works only for 2 qubit states")

        if not (0 <= param <= 1):
            raise DensityMatrixException("Expected the parameter to be between 0 and 1")

        state = self.state
        dim = int(state.shape[0])

        a0 = np.array([[1, 0], [0, np.sqrt(1 - param)]])
        a0d = np.array([[1, 0], [0, np.sqrt(1 - param)]])

        a1 = np.array([[0, np.sqrt(param)], [0, 0]])
        a1d = np.array([[0, 0], [np.sqrt(param), 0]])
        a = [a0, a1]
        ad = [a0d, a1d]

        rho = np.zeros((dim, dim), dtype=np.complex128)
        for i in range(len(a)):
            for j in range(len(ad)):
                rho += np.kron(a[i], a[j]) @ state @ (np.kron(ad[i], ad[j]))

        return DensityMatrix(rho)

    def depol(self, param: float):
        """ """

        # works only for 2 qubit states
        if self.state.shape != (4, 4):
            raise DensityMatrixException("This method works only for 2 qubit states")

        if not (0 <= param <= 1):
            raise DensityMatrixException("Expected the parameter to be between 0 and 1")

        state = self.state
        dim = int(state.shape[0])

        a0 = np.sqrt(1 - param) * np.matrix([[1, 0], [0, 1]])
        a1 = np.sqrt(param / 3) * np.matrix([[0, 1], [1, 0]])
        a2 = np.sqrt(param / 3) * np.matrix([[0, -1j], [1j, 0]])
        a3 = np.sqrt(param / 3) * np.matrix([[1, 0], [0, -1]])
        a = [a0, a1, a2, a3]
        ad = [a0, a1, a2, a3]

        rho = np.zeros((dim, dim), dtype=np.complex128)
        for i in range(len(a)):
            for j in range(len(ad)):
                rho += np.kron(a[i], a[j]).dot(state).dot(np.kron(ad[i], ad[j]))

        return DensityMatrix(rho)

    # von_neumann_entropy() is inherited from BaseDensityMatrix with improved implementation

    def reyni(self, base: float = 2, alpha: float = 1):
        """
        Compute the entropy of a density matrix rho.

        Parameters:
        - rho : ndarray
            The density matrix.
        - base : int or float, optional, default 2
            The base of the logarithm. Defaults to base 2.
        - alpha : int or float, optional, default 1
            The Renyi parameter. Defaults to 1 (von Neumann entropy).

        Returns:
        - ent : float
            The computed entropy value.
        """

        # Ensure rho is a NumPy array and compute eigenvalues
        rho = self.state
        lam = np.linalg.eigvals(rho)
        lam = lam[lam > 0]  # Only consider positive eigenvalues

        # If alpha == 1, compute the von Neumann entropy
        if abs(alpha - 1) <= np.finfo(float).eps ** (3 / 4):
            if base == 2:
                ent = -np.sum(np.real(lam * np.log2(lam)))
            else:
                ent = -np.sum(np.real(lam * np.log(lam))) / np.log(base)
        elif alpha >= 0:
            if alpha < np.inf:  # Renyi-alpha entropy with alpha < Inf
                ent = np.log(np.sum(lam**alpha)) / (np.log(base) * (1 - alpha))

                # Check if numerical problems occurred due to a large alpha
                if np.isinf(ent):
                    alpha = np.inf
                    print(
                        "Warning: Numerical problems encountered due to a large value of ALPHA. Computing the entropy with ALPHA = Inf instead."
                    )
            else:
                ent = np.inf  # for cases where alpha = Inf

            if alpha == np.inf:  # Renyi-infinity entropy
                ent = -np.log(np.max(lam)) / np.log(base)
        else:
            raise DensityMatrixException("ALPHA must be non-negative.")

        return ent

    def kraus(self, operator: List[np.array]):
        """
        Applies a sequence of Kraus operators on the density matrix
        """

        # works only on 2 quibt states
        if self.state.shape != (4, 4):
            raise DensityMatrixException("This method works only for 2 qubit states")

        state = self.state
        dim = int(state.shape[0])
        a = operator
        ad = [index.getH() for index in a]

        d = a[0].shape[0]
        rho_flag = np.zeros((d, d), dtype=np.complex128)
        for index in range(len(a)):
            rho_flag += ad[index].dot(a[index])

        if not np.allclose(rho_flag, np.identity(d)):
            raise ValueError("Invalid Kraus Operators!")
        else:
            rho = np.zeros((dim, dim), dtype=np.complex128)
            for i in range(len(a)):
                for j in range(len(ad)):
                    rho += np.kron(a[i], a[j]).dot(state).dot(np.kron(ad[i], ad[j]))
            return DensityMatrix(rho)

    def partial_transpose(self, dims: List[int], axis: int) -> "DensityMatrix":
        """
        Performs partial transpose on the density matrix
        """

        state = self.state
        rho = Expression.cast_to_const(state)

        if rho.ndim < 2 or rho.shape[0] != rho.shape[1]:
            raise DensityMatrixException("Not a square matrix.")
        if axis < 0 or axis >= len(dims):
            raise DensityMatrixException(
                f"Invalid subsystem argument, should be between 0 and {len(dims)}, got {axis}."
            )
        if rho.shape[0] != np.prod(dims):
            raise DensityMatrixException(
                "Dimension of system doesn't correspond to dimension of subsystems."
            )

        def term(rho: np.array, i: int, j: int, dims: List[int], axis=0):
            """ """
            a = spsparce.coo_matrix(([1.0], ([0], [0])))
            for i_axis, dim in enumerate(dims):
                if i_axis == axis:
                    v = spsparce.coo_matrix(([1], ([i], [j])), shape=(dim, dim))
                    a = spsparce.kron(a, v)
                else:
                    eye_mat = spsparce.eye(dim)
                    a = spsparce.kron(a, eye_mat)
            return a @ rho @ a

        return DensityMatrix(
            sum(
                [
                    term(rho, i, j, dims, axis)
                    for i in range(dims[axis])
                    for j in range(dims[axis])
                ]
            ).value
        )

    def partial_trace(self, dims: List[int], axis: int) -> "DensityMatrix":  # type: ignore[override]
        """ """

        state = self.state
        dims_ = np.array(dims)
        reshaped_rho = np.reshape(state, np.concatenate((dims_, dims_), axis=None))

        reshaped_rho = np.moveaxis(reshaped_rho, axis, -1)
        reshaped_rho = np.moveaxis(reshaped_rho, len(dims) + axis - 1, -1)

        traced_out_rho = np.trace(reshaped_rho, axis1=-2, axis2=-1)

        dims_untraced = np.delete(dims_, axis)
        rho_dim = np.prod(dims_untraced)

        flag = traced_out_rho.reshape([rho_dim, rho_dim])
        return DensityMatrix(flag / np.trace(flag))

    def concurrence(self) -> float:
        """
        Computes the concurrence of a 2-qubit density matrix.
        Concurrence C(rho) = max(0, lambda_1 - lambda_2 - lambda_3 - lambda_4)
        where lambda_i are the square roots of the eigenvalues of rho * rho_tilde,
        sorted in descending order.
        """

        sy = np.array([[0, -1j], [1j, 0]])
        state = self.state
        statec = state.conjugate()

        state_tilde = np.kron(sy, sy) @ statec @ np.kron(sy, sy)

        flag = np.dot(state, state_tilde)
        flag_eig = splinalg.eigvals(flag)

        # Eigenvalues of rho * rho_tilde are theoretically real and non-negative.
        # We take the real part and maximum with 0 to handle numerical precision issues.
        real_eigvals = np.maximum(0, np.real(flag_eig))
        sqrt_eigvals = np.sqrt(real_eigvals)

        # Sort in descending order
        sqrt_eigvals_sorted = sorted(sqrt_eigvals, reverse=True)

        conc = sqrt_eigvals_sorted[0]
        for index in range(1, len(sqrt_eigvals_sorted)):
            conc -= sqrt_eigvals_sorted[index]

        return float(np.max([conc, 0]))

    # fidelity() is inherited from BaseDensityMatrix

    def l1_norm(self) -> float:
        """
        Computes the L1 norm of the density matrix
        """
        state = self.state
        return np.sum(np.abs(state)) - np.trace(state)

    # purity() is inherited from BaseDensityMatrix

    def distinguishability(self, X, *varargin):
        # Check if X is a cell array (list of density matrices) or a matrix (list of pure states)
        if isinstance(X, list):
            num_ops = len(X)
            dim = len(X[0])
        else:
            num_ops, dim = X.shape

        # Set default for p if not provided
        p = np.ones(num_ops) / num_ops if not varargin else np.array(varargin[0])

        # Check that p is a valid probability distribution
        if (
            np.abs(np.sum(p) - 1) > num_ops**2 * np.finfo(float).eps
            or len(p) != num_ops
        ):
            raise ValueError(
                "The vector P must be a probability distribution of the same length as the number of states: its elements must be non-negative and they must sum to 1."
            )

        # Handle trivial case (num_ops == 1 or one probability is 1)
        if num_ops == 1 or np.max(p) >= 1:
            dist = 1
            meas = np.eye(dim) if len(varargin) > 1 else None
            return dist, meas if len(varargin) > 1 else dist

        # If X is a list of density matrices
        if isinstance(X, list):
            # Scale density matrices to ensure trace = 1
            for j in range(num_ops):
                X[j] = X[j] / np.trace(X[j])

            # For two density matrices: closed-form expression for distinguishability
            if num_ops == 2:
                dist = 0.5 + np.linalg.norm(p[0] * X[0] - p[1] * X[1], ord="nuc") / 2
                if len(varargin) > 1:
                    eigvals, eigvecs = np.linalg.eigh(p[0] * X[0] - p[1] * X[1])
                    pind = np.where(eigvals >= 0)[0]
                    meas = [
                        eigvecs[:, pind] @ eigvecs[:, pind].T,
                        np.eye(dim) - eigvecs[:, pind] @ eigvecs[:, pind].T,
                    ]
                return dist, meas if len(varargin) > 1 else dist

            # Check for mutually orthogonal states
            if num_ops <= dim:
                mut_orth = all(
                    np.max(np.abs(X[j] @ X[k])) <= np.finfo(float).eps * dim**2
                    for j in range(num_ops)
                    for k in range(j + 1, num_ops)
                )
                if mut_orth:
                    dist = 1
                    if len(varargin) > 1:
                        meas_sum = np.zeros((dim, dim))
                        for j in range(num_ops - 1, 0, -1):
                            oX = np.linalg.qr(X[j])[0]
                            meas_sum += oX @ oX.T
                        meas = [np.eye(dim) - meas_sum]
                    return dist, meas if len(varargin) > 1 else dist

        # If X is a matrix of pure states
        else:
            X = X / np.linalg.norm(X, axis=0)  # Normalize the columns

            # For two pure states: closed-form expression for distinguishability
            if num_ops == 2:
                dist = (
                    0.5
                    + np.sqrt(
                        2 * (p[0] ** 2 + p[1] ** 2)
                        - 4 * p[0] * p[1] * np.abs(X[:, 0].T @ X[:, 1]) ** 2
                    )
                    / 2
                )
                if len(varargin) > 1:
                    eigvals, eigvecs = np.linalg.eigh(
                        p[0] * np.outer(X[:, 0], X[:, 0])
                        - p[1] * np.outer(X[:, 1], X[:, 1])
                    )
                    pind = np.where(eigvals >= 0)[0]
                    meas = [
                        eigvecs[:, pind] @ eigvecs[:, pind].T,
                        np.eye(dim) - eigvecs[:, pind] @ eigvecs[:, pind].T,
                    ]
                return dist, meas if len(varargin) > 1 else dist

            # Check for mutually orthogonal states
            if num_ops <= dim:
                X2 = X.T @ X
                if (
                    np.max(np.abs(X2 - np.diag(np.diag(X2))))
                    < np.finfo(float).eps * dim**2
                ):
                    dist = 1
                    if len(varargin) > 1:
                        meas_sum = np.zeros((dim, dim))
                        for j in range(num_ops - 1, 0, -1):
                            meas_sum += np.outer(X[:, j], X[:, j])
                        meas = [np.eye(dim) - meas_sum]
                    return dist, meas if len(varargin) > 1 else dist

            # Convert pure states to density matrices
            X = [np.outer(X[:, j], X[:, j]) for j in range(num_ops)]

        # For 3 or more density matrices, use semidefinite programming
        P = cp.Variable((dim, dim, num_ops), hermitian=True)
        P_tr = sum(p[j] * cp.trace(P[:, :, j] @ X[j]) for j in range(num_ops))
        P_sum = cp.sum(P, axis=2)

        constraints = [P_sum == np.eye(dim), P >= 0]
        objective = cp.Maximize(P_tr)
        problem = cp.Problem(objective, constraints)
        problem.solve()

        dist = np.real(problem.value) / 2

        # Return the optimal measurements if requested
        meas = [P[:, :, j].value for j in range(num_ops)] if len(varargin) > 1 else None
        return dist, meas if len(varargin) > 1 else dist

    def relative_entropy_coherence(self):
        """ """

        # Assuming 'rho' is the density matrix
        rho = np.matrix(self.dm)

        # Calculate the entropy as per the given formula
        rho_diag = DensityMatrix(np.diag(rho))  # Extract diagonal matrix of rho
        return rho_diag.reyni() - rho_diag.reyni()

    def max_bell_value(self):
        """ """

        rho = self.state

        # works only on 2 qubits
        if rho.shape != (4, 4):
            raise DensityMatrixException("This method works only for 2 qubit states")

        s1 = np.array([[0, 1], [1, 0]])
        s2 = np.array([[0, -1j], [1j, 0]])
        s3 = np.array([[1, 0], [0, -1]])

        matrices = [s1, s2, s3]
        t = np.zeros([3, 3], dtype=complex)

        for i in range(3):
            for j in range(3):
                kron_result = np.kron(matrices[i], matrices[j])
                t[i][j] = np.trace(rho.dot(kron_result))

        tdag = np.conj(t).T
        dm1 = tdag.dot(t)
        u = np.sort(np.linalg.eigvals(dm1))

        m = u[-1] + u[-2]
        Bell = np.real(2 * np.sqrt(m))
        return Bell

    def teleportation_fidelity(self):
        """ """

        rho = self.state

        s1 = np.array([[0, 1], [1, 0]])
        s2 = np.array([[0, -1j], [1j, 0]])
        s3 = np.array([[1, 0], [0, -1]])

        matrices = [s1, s2, s3]
        t = np.zeros([3, 3], dtype=complex)

        for i in range(3):
            for j in range(3):
                kron_result = np.kron(matrices[i], matrices[j])
                t[i][j] = np.trace(rho.dot(kron_result))

        tdag = np.conj(t).T
        dm1 = tdag.dot(t)
        u = np.sort(np.linalg.eigvals(dm1))

        n = np.sqrt(u[0]) + np.sqrt(u[1]) + np.sqrt(u[2])
        fid = np.real(0.5 * (1 + (n / 3)))
        return fid

    def correlation_points(self, state: np.array):
        """ """

        sx = np.array([[0, 1], [1, 0]])
        sy = np.array([[0, -1j], [1j, 0]])
        sz = np.array([[1, 0], [0, -1]])

        cxx = np.real(np.trace(np.dot(state, np.kron(sx, sx))))
        cyy = np.real(np.trace(np.dot(state, np.kron(sy, sy))))
        czz = np.real(np.trace(np.dot(state, np.kron(sz, sz))))

        return [cxx, cyy, czz]

    def plot_correlation_space_2q(self, p_adc=False, p_depol=False):
        state = np.array(self.dm)

        psi_plus = [1.0, -1.0, 1.0]
        psi_minus = [1.0, 1.0, -1.0]
        phi_plus = [-1.0, -1.0, -1.0]
        phi_minus = [-1.0, 1.0, 1.0]

        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

        ax.set_xlabel(r"$\sigma_x \otimes \sigma_x$")
        ax.set_ylabel(r"$\sigma_y \otimes \sigma_y$")
        ax.set_zlabel(r"$\sigma_z \otimes \sigma_z$", rotation=90)

        ax.set_xlim([-1, 1])
        ax.set_ylim([-1, 1])
        ax.set_zlim([-1, 1])
        ax.set_xticks(np.arange(-1, 1.5, 0.5))
        ax.set_yticks(np.arange(-1, 1.5, 0.5))
        ax.set_zticks(np.arange(-1, 1.5, 0.5))

        ax.grid(False)

        vertices = np.array([psi_plus, psi_minus, phi_plus, phi_minus])

        faces = [
            [vertices[j] for j in [0, 1, 2]],
            [vertices[j] for j in [0, 1, 3]],
            [vertices[j] for j in [0, 2, 3]],
            [vertices[j] for j in [1, 2, 3]],
        ]

        tetrahedron = Poly3DCollection(
            faces,
            alpha=0.3,
            facecolors=(0.2, 0.6, 0.8),
            linewidths=0.1,
            edgecolors="k",
        )
        ax.add_collection3d(tetrahedron)

        x_limits = ax.get_xlim()
        y_limits = ax.get_ylim()
        z_limits = ax.get_zlim()

        frame_edges = [
            [
                [x_limits[0], x_limits[0]],
                [y_limits[0], y_limits[0]],
                [z_limits[0], z_limits[1]],
            ],
            [
                [x_limits[0], x_limits[0]],
                [y_limits[0], y_limits[1]],
                [z_limits[0], z_limits[0]],
            ],
            [
                [x_limits[0], x_limits[1]],
                [y_limits[0], y_limits[0]],
                [z_limits[0], z_limits[0]],
            ],
            [
                [x_limits[1], x_limits[1]],
                [y_limits[0], y_limits[0]],
                [z_limits[0], z_limits[1]],
            ],
            [
                [x_limits[1], x_limits[1]],
                [y_limits[0], y_limits[1]],
                [z_limits[0], z_limits[0]],
            ],
            [
                [x_limits[0], x_limits[1]],
                [y_limits[1], y_limits[1]],
                [z_limits[0], z_limits[0]],
            ],
            [
                [x_limits[0], x_limits[0]],
                [y_limits[1], y_limits[1]],
                [z_limits[0], z_limits[1]],
            ],
            [
                [x_limits[1], x_limits[1]],
                [y_limits[1], y_limits[1]],
                [z_limits[0], z_limits[1]],
            ],
            [
                [x_limits[0], x_limits[0]],
                [y_limits[0], y_limits[0]],
                [z_limits[1], z_limits[1]],
            ],
            [
                [x_limits[0], x_limits[1]],
                [y_limits[0], y_limits[0]],
                [z_limits[1], z_limits[1]],
            ],
            [
                [x_limits[1], x_limits[1]],
                [y_limits[0], y_limits[1]],
                [z_limits[1], z_limits[1]],
            ],
            [
                [x_limits[0], x_limits[1]],
                [y_limits[1], y_limits[1]],
                [z_limits[1], z_limits[1]],
            ],
        ]

        for edge in frame_edges:
            ax.plot(edge[0], edge[1], edge[2], color="k", linewidth=0.1)

        ax.scatter(psi_plus[0], psi_plus[1], psi_plus[2], color="b", s=20)
        ax.scatter(psi_minus[0], psi_minus[1], psi_minus[2], color="r", s=20)
        ax.scatter(phi_plus[0], phi_plus[1], phi_plus[2], color="g", s=20)
        ax.scatter(phi_minus[0], phi_minus[1], phi_minus[2], color="y", s=20)

        offset = 0.15
        ax.text(
            psi_plus[0],
            psi_plus[1],
            psi_plus[2] + offset,
            r"$\psi^+$",
            color="b",
        )
        ax.text(
            psi_minus[0],
            psi_minus[1],
            psi_minus[2] + offset,
            r"$\psi^-$",
            color="r",
        )
        ax.text(
            phi_plus[0],
            phi_plus[1],
            phi_plus[2] + offset,
            r"$\phi^+$",
            color="g",
        )
        ax.text(
            phi_minus[0],
            phi_minus[1],
            phi_minus[2] + offset,
            r"$\phi^-$",
            color="y",
        )

        if (not p_adc) and (not p_depol):
            pts = self.correlation_points(state)
            ax.scatter(pts[0], pts[1], pts[2], color="black", s=20, marker=".")
            ax.scatter([], [], color="black", s=20, marker=".", label="state")

        if p_adc:
            for index in np.arange(0, 1, 0.05):
                r = self.ADC_2q(index).get_state()
                pts = self.correlation_points(r)
                ax.scatter(pts[0], pts[1], pts[2], color="black", s=5, marker="*")
            ax.scatter([], [], color="black", s=5, marker="*", label="ADC")

            if p_depol:
                for index in np.arange(0, 1, 0.05):
                    r = self.depol(index).get_state()
                    pts = self.correlation_points(r)
                    ax.scatter(pts[0], pts[1], pts[2], color="red", s=5, marker=".")
                ax.scatter([], [], color="red", s=5, marker=".", label="Depol")

        elif not p_adc:
            if p_depol:
                for index in np.arange(0, 1, 0.05):
                    r = self.depol(index).get_state()
                    pts = self.correlation_points(r)
                    ax.scatter(pts[0], pts[1], pts[2], color="red", s=5, marker=".")
                ax.scatter([], [], color="red", s=5, marker=".", label="Depol")

        ax.legend(frameon=False)

        ax.zaxis.labelpad = -0.1
        plt.title("Two-qubit Correlation space")
        plt.show()

    def eof(self):
        """
        Calculates the entanglement of formation of a density matrix
        """

        def ent(x):
            return -np.where(x == 0, 0, x * np.log2(x)) - np.where(
                x == 1, 0, (1 - x) * np.log2(1 - x)
            )

        value = self.concurrence()

        # ent calculates the shannon entropy
        x = np.round((1 + np.sqrt(1 - (value * value))) / 2, 3)
        return np.real(ent(x))

    def schmidt_rank(self):
        """
        Computes the Schmidt rank of a density matrix
        """

        matrix = self.state
        U, S, V = np.linalg.svd(matrix)

        return np.linalg.matrix_rank(S)

    def has_symmetric_extension(self):
        # Reference paper: arXiv:1310.3530v2 [quant-ph] (2014)
        return np.trace(
            self.partial_trace(dims=[2, 2], axis=0).get_state() ** 2
        ) >= np.trace(self.get_state() ** 2) - 4 * np.sqrt(
            np.linalg.det(self.get_state())
        )

    @staticmethod
    def basis_operators(argument: int):
        identity = np.array([[1, 0], [0, 1]])
        sx = np.array([[0, 1], [1, 0]])
        sy = np.array([[0, -1j], [1j, 0]])
        sz = np.array([[1, 0], [0, -1]])

        L1 = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]])
        L2 = np.array([[0, -1j, 0], [1j, 0, 0], [0, 0, 0]])
        L3 = np.array([[1, 0, 0], [0, -1, 0], [0, 0, 0]])
        L4 = np.array([[0, 0, 1], [0, 0, 0], [1, 0, 0]])
        L5 = np.array([[0, 0, -1j], [0, 0, 0], [1j, 0, 0]])
        L6 = np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0]])
        L7 = np.array([[0, 0, 0], [0, 0, -1j], [0, 1j, 0]])
        L8 = np.array(
            [
                [1 / np.sqrt(3), 0, 0],
                [0, 1 / np.sqrt(3), 0],
                [0, 0, -2 / np.sqrt(3)],
            ]
        )

        switcher = {
            0: identity,
            1: sx,
            2: sy,
            3: sz,
            4: L1,
            5: L2,
            6: L3,
            7: L4,
            8: L5,
            9: L6,
            10: L7,
            11: L8,
        }

        return switcher.get(argument, "invalid argument choice")

    @staticmethod
    def check_MUB(a: List[int], b: List[int]):
        dima = [len(item) for item in a]
        dimb = [len(item) for item in b]

        if len(set(dima)) == 1:
            if len(set(dimb)) == 1:
                if max(dima) == max(dimb):
                    flag = []
                    for outer_index in range(len(a)):
                        for inner_index in range(len(b)):
                            flag.append(
                                np.round(
                                    np.abs(np.dot(a[outer_index], b[inner_index])) ** 2,
                                    4,
                                )
                                - (1 / len(a))
                            )

                    if np.count_nonzero(flag) == 0:
                        print("Checking for MUB: passed for test all cases")
                    else:
                        print("Checking for MUB: some test cases failed")
                else:
                    raise ValueError("Dimensions of the two bases do not match!")
            else:
                raise ValueError(
                    "Dimensions of the second bases elements are not the same!"
                )
        else:
            raise ValueError("Dimensions of the first bases elements are not the same!")

    @staticmethod
    def tensor_product(*args):
        matrices = args[0] if len(args) == 1 and isinstance(args[0], list) else args
        result = matrices[0]
        for matrix in matrices[1:]:
            result = np.kron(result, matrix)
        return result

    @staticmethod
    def gate_expand_2toN(U, N, control=None, target=None, targets=None):
        if targets is not None:
            control, target = targets
        if control is None or target is None:
            raise ValueError("Specify value of control and target")
        if N < 2:
            raise ValueError("integer N must be larger or equal to 2")
        if control >= N or target >= N:
            raise ValueError("control and not target must be integer < integer N")
        if control == target:
            raise ValueError("target and not control cannot be equal")

        # Create permutation list
        p = list(range(N))
        if target == 0 and control == 1:
            p[control], p[target] = p[target], p[control]
        elif target == 0:
            p[1], p[target] = p[target], p[1]
            p[1], p[control] = p[control], p[1]
        else:
            p[1], p[target] = p[target], p[1]
            p[0], p[control] = p[control], p[0]

        # Create expanded operator
        expanded = DensityMatrix.tensor_product(
            [U] + [np.eye(2, dtype=complex)] * (N - 2)
        )

        # Reshape for permutation
        dims = [2] * N
        tensor_shape = dims + dims
        tensor_data = expanded.reshape(tensor_shape)

        # Permute subsystems
        new_order = list(p) + [o + N for o in p]
        permuted = np.transpose(tensor_data, new_order)

        # Reshape back to matrix
        return permuted.reshape((2**N, 2**N))

    @staticmethod
    def swap(N=None, targets=[0, 1]):
        swap_matrix = np.array(
            [[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]],
            dtype=complex,
        )

        if targets != [0, 1] and N is None:
            N = 2

        if N is not None:
            return DensityMatrix.gate_expand_2toN(swap_matrix, N, targets=targets)
        else:
            return swap_matrix

    @staticmethod
    def entangling_power_2q(U: np.array):
        if U.shape != (4, 4):
            raise Exception("U must be a two-qubit gate (4x4 matrix).")

        UU = DensityMatrix.tensor_product(U, U)
        S13 = DensityMatrix.swap(N=4, targets=[1, 3])
        S = DensityMatrix.swap()

        a_term = np.dot(UU.conj().T, np.dot(S13, np.dot(UU, S13)))

        SU = np.dot(S, U)
        SUSU = DensityMatrix.tensor_product(SU, SU)
        b_term = np.dot(SUSU.conj().T, np.dot(S13, np.dot(SUSU, S13)))

        return 5.0 / 9 - 1.0 / 36 * (np.trace(a_term) + np.trace(b_term)).real
