"""Spectral Embedding."""

# Author: Gael Varoquaux <gael.varoquaux@normalesup.org>
#         Wei LI <kuantkid@gmail.com>
# License: BSD 3 clause


from numbers import Integral, Real
import warnings
import numbers

import numpy as np
from scipy import sparse
from scipy.linalg import eigh
from scipy.sparse.linalg import eigsh, lobpcg
from scipy.sparse.csgraph import connected_components
from scipy.sparse.csgraph import laplacian as csgraph_laplacian
import scipy.sparse as sp
from sklearn.cluster import KMeans

# from ..base import BaseEstimator
# from ..utils import (
#     check_array,
#     check_random_state,
#     check_symmetric,
# )
# from ..utils._arpack import _init_arpack_v0
# from ..utils.extmath import _deterministic_vector_sign_flip
# from ..utils._param_validation import Interval, StrOptions
# from ..utils.fixes import lobpcg
# from ..metrics.pairwise import rbf_kernel
# from ..neighbors import kneighbors_graph, NearestNeighbors


def _deterministic_vector_sign_flip(u):
    """Modify the sign of vectors for reproducibility.
    Flips the sign of elements of all the vectors (rows of u) such that
    the absolute maximum element of each vector is positive.
    Parameters
    ----------
    u : ndarray
        Array with vectors as its rows.
    Returns
    -------
    u_flipped : ndarray with same shape as u
        Array with the sign flipped vectors as its rows.
    """
    max_abs_rows = np.argmax(np.abs(u), axis=1)
    signs = np.sign(u[range(u.shape[0]), max_abs_rows])
    u *= signs[:, np.newaxis]
    return u


def _pandas_dtype_needs_early_conversion(pd_dtype):
    """Return True if pandas extension pd_dtype need to be converted early."""
    # Check these early for pandas versions without extension dtypes
    from pandas.api.types import (
        is_bool_dtype,
        is_sparse,
        is_float_dtype,
        is_integer_dtype,
    )

    if is_bool_dtype(pd_dtype):
        # bool and extension booleans need early converstion because __array__
        # converts mixed dtype dataframes into object dtypes
        return True

    if is_sparse(pd_dtype):
        # Sparse arrays will be converted later in `check_array`
        return False

    try:
        from pandas.api.types import is_extension_array_dtype
    except ImportError:
        return False

    if is_sparse(pd_dtype) or not is_extension_array_dtype(pd_dtype):
        # Sparse arrays will be converted later in `check_array`
        # Only handle extension arrays for integer and floats
        return False
    elif is_float_dtype(pd_dtype):
        # Float ndarrays can normally support nans. They need to be converted
        # first to map pd.NA to np.nan
        return True
    elif is_integer_dtype(pd_dtype):
        # XXX: Warn when converting from a high integer to a float
        return True

    return False

# def check_symmetric(array, *, tol=1e-10, raise_warning=True, raise_exception=False):
#     """Make sure that array is 2D, square and symmetric.
#     If the array is not symmetric, then a symmetrized version is returned.
#     Optionally, a warning or exception is raised if the matrix is not
#     symmetric.
#     Parameters
#     ----------
#     array : {ndarray, sparse matrix}
#         Input object to check / convert. Must be two-dimensional and square,
#         otherwise a ValueError will be raised.
#     tol : float, default=1e-10
#         Absolute tolerance for equivalence of arrays. Default = 1E-10.
#     raise_warning : bool, default=True
#         If True then raise a warning if conversion is required.
#     raise_exception : bool, default=False
#         If True then raise an exception if array is not symmetric.
#     Returns
#     -------
#     array_sym : {ndarray, sparse matrix}
#         Symmetrized version of the input array, i.e. the average of array
#         and array.transpose(). If sparse, then duplicate entries are first
#         summed and zeros are eliminated.
#     """
#     if (array.ndim != 2) or (array.shape[0] != array.shape[1]):
#         raise ValueError(
#             "array must be 2-dimensional and square. shape = {0}".format(array.shape)
#         )

#     if sp.issparse(array):
#         diff = array - array.T
#         # only csr, csc, and coo have `data` attribute
#         if diff.format not in ["csr", "csc", "coo"]:
#             diff = diff.tocsr()
#         symmetric = np.all(abs(diff.data) < tol)
#     else:
#         symmetric = np.allclose(array, array.T, atol=tol)

#     if not symmetric:
#         if raise_exception:
#             raise ValueError("Array must be symmetric")
#         if raise_warning:
#             warnings.warn(
#                 "Array is not symmetric, and will be converted "
#                 "to symmetric by average with its transpose.",
#                 stacklevel=2,
#             )
#         if sp.issparse(array):
#             conversion = "to" + array.format
#             array = getattr(0.5 * (array + array.T), conversion)()
#         else:
#             array = 0.5 * (array + array.T)

#     return 

# def check_array(
#     array,
#     accept_sparse=False,
#     *,
#     accept_large_sparse=True,
#     dtype="numeric",
#     order=None,
#     copy=False,
#     force_all_finite=True,
#     ensure_2d=True,
#     allow_nd=False,
#     ensure_min_samples=1,
#     ensure_min_features=1,
#     estimator=None,
#     input_name="",
# ):

#     """Input validation on an array, list, sparse matrix or similar.
#     By default, the input is checked to be a non-empty 2D array containing
#     only finite values. If the dtype of the array is object, attempt
#     converting to float, raising on failure.
#     Parameters
#     ----------
#     array : object
#         Input object to check / convert.
#     accept_sparse : str, bool or list/tuple of str, default=False
#         String[s] representing allowed sparse matrix formats, such as 'csc',
#         'csr', etc. If the input is sparse but not in the allowed format,
#         it will be converted to the first listed format. True allows the input
#         to be any format. False means that a sparse matrix input will
#         raise an error.
#     accept_large_sparse : bool, default=True
#         If a CSR, CSC, COO or BSR sparse matrix is supplied and accepted by
#         accept_sparse, accept_large_sparse=False will cause it to be accepted
#         only if its indices are stored with a 32-bit dtype.
#         .. versionadded:: 0.20
#     dtype : 'numeric', type, list of type or None, default='numeric'
#         Data type of result. If None, the dtype of the input is preserved.
#         If "numeric", dtype is preserved unless array.dtype is object.
#         If dtype is a list of types, conversion on the first type is only
#         performed if the dtype of the input is not in the list.
#     order : {'F', 'C'} or None, default=None
#         Whether an array will be forced to be fortran or c-style.
#         When order is None (default), then if copy=False, nothing is ensured
#         about the memory layout of the output array; otherwise (copy=True)
#         the memory layout of the returned array is kept as close as possible
#         to the original array.
#     copy : bool, default=False
#         Whether a forced copy will be triggered. If copy=False, a copy might
#         be triggered by a conversion.
#     force_all_finite : bool or 'allow-nan', default=True
#         Whether to raise an error on np.inf, np.nan, pd.NA in array. The
#         possibilities are:
#         - True: Force all values of array to be finite.
#         - False: accepts np.inf, np.nan, pd.NA in array.
#         - 'allow-nan': accepts only np.nan and pd.NA values in array. Values
#           cannot be infinite.
#         .. versionadded:: 0.20
#            ``force_all_finite`` accepts the string ``'allow-nan'``.
#         .. versionchanged:: 0.23
#            Accepts `pd.NA` and converts it into `np.nan`
#     ensure_2d : bool, default=True
#         Whether to raise a value error if array is not 2D.
#     allow_nd : bool, default=False
#         Whether to allow array.ndim > 2.
#     ensure_min_samples : int, default=1
#         Make sure that the array has a minimum number of samples in its first
#         axis (rows for a 2D array). Setting to 0 disables this check.
#     ensure_min_features : int, default=1
#         Make sure that the 2D array has some minimum number of features
#         (columns). The default value of 1 rejects empty datasets.
#         This check is only enforced when the input data has effectively 2
#         dimensions or is originally 1D and ``ensure_2d`` is True. Setting to 0
#         disables this check.
#     estimator : str or estimator instance, default=None
#         If passed, include the name of the estimator in warning messages.
#     input_name : str, default=""
#         The data name used to construct the error message. In particular
#         if `input_name` is "X" and the data has NaN values and
#         allow_nan is False, the error message will link to the imputer
#         documentation.
#         .. versionadded:: 1.1.0
#     Returns
#     -------
#     array_converted : object
#         The converted and validated array.
#     """
#     if isinstance(array, np.matrix):
#         raise TypeError(
#             "np.matrix is not supported. Please convert to a numpy array with "
#             "np.asarray. For more information see: "
#             "https://numpy.org/doc/stable/reference/generated/numpy.matrix.html"
#         )

#     xp, is_array_api = get_namespace(array)

#     # store reference to original array to check if copy is needed when
#     # function returns
#     array_orig = array

#     # store whether originally we wanted numeric dtype
#     dtype_numeric = isinstance(dtype, str) and dtype == "numeric"

#     dtype_orig = getattr(array, "dtype", None)
#     if not hasattr(dtype_orig, "kind"):
#         # not a data type (e.g. a column named dtype in a pandas DataFrame)
#         dtype_orig = None

#     # check if the object contains several dtypes (typically a pandas
#     # DataFrame), and store them. If not, store None.
#     dtypes_orig = None
#     pandas_requires_conversion = False
#     if hasattr(array, "dtypes") and hasattr(array.dtypes, "__array__"):
#         # throw warning if columns are sparse. If all columns are sparse, then
#         # array.sparse exists and sparsity will be preserved (later).
#         with suppress(ImportError):
#             from pandas.api.types import is_sparse

#             if not hasattr(array, "sparse") and array.dtypes.apply(is_sparse).any():
#                 warnings.warn(
#                     "pandas.DataFrame with sparse columns found."
#                     "It will be converted to a dense numpy array."
#                 )

#         dtypes_orig = list(array.dtypes)
#         pandas_requires_conversion = any(
#             _pandas_dtype_needs_early_conversion(i) for i in dtypes_orig
#         )
#         if all(isinstance(dtype_iter, np.dtype) for dtype_iter in dtypes_orig):
#             dtype_orig = np.result_type(*dtypes_orig)

#     elif hasattr(array, "iloc") and hasattr(array, "dtype"):
#         # array is a pandas series
#         pandas_requires_conversion = _pandas_dtype_needs_early_conversion(array.dtype)
#         if isinstance(array.dtype, np.dtype):
#             dtype_orig = array.dtype
#         else:
#             # Set to None to let array.astype work out the best dtype
#             dtype_orig = None

#     if dtype_numeric:
#         if dtype_orig is not None and dtype_orig.kind == "O":
#             # if input is object, convert to float.
#             dtype = xp.float64
#         else:
#             dtype = None

#     if isinstance(dtype, (list, tuple)):
#         if dtype_orig is not None and dtype_orig in dtype:
#             # no dtype conversion required
#             dtype = None
#         else:
#             # dtype conversion required. Let's select the first element of the
#             # list of accepted types.
#             dtype = dtype[0]

#     if pandas_requires_conversion:
#         # pandas dataframe requires conversion earlier to handle extension dtypes with
#         # nans
#         # Use the original dtype for conversion if dtype is None
#         new_dtype = dtype_orig if dtype is None else dtype
#         array = array.astype(new_dtype)
#         # Since we converted here, we do not need to convert again later
#         dtype = None

#     if force_all_finite not in (True, False, "allow-nan"):
#         raise ValueError(
#             'force_all_finite should be a bool or "allow-nan". Got {!r} instead'.format(
#                 force_all_finite
#             )
#         )

#     estimator_name = _check_estimator_name(estimator)
#     context = " by %s" % estimator_name if estimator is not None else ""

#     # When all dataframe columns are sparse, convert to a sparse array
#     if hasattr(array, "sparse") and array.ndim > 1:
#         with suppress(ImportError):
#             from pandas.api.types import is_sparse

#             if array.dtypes.apply(is_sparse).all():
#                 # DataFrame.sparse only supports `to_coo`
#                 array = array.sparse.to_coo()
#                 if array.dtype == np.dtype("object"):
#                     unique_dtypes = set([dt.subtype.name for dt in array_orig.dtypes])
#                     if len(unique_dtypes) > 1:
#                         raise ValueError(
#                             "Pandas DataFrame with mixed sparse extension arrays "
#                             "generated a sparse matrix with object dtype which "
#                             "can not be converted to a scipy sparse matrix."
#                             "Sparse extension arrays should all have the same "
#                             "numeric type."
#                         )

#     if sp.issparse(array):
#         _ensure_no_complex_data(array)
#         array = _ensure_sparse_format(
#             array,
#             accept_sparse=accept_sparse,
#             dtype=dtype,
#             copy=copy,
#             force_all_finite=force_all_finite,
#             accept_large_sparse=accept_large_sparse,
#             estimator_name=estimator_name,
#             input_name=input_name,
#         )
#     else:
#         # If np.array(..) gives ComplexWarning, then we convert the warning
#         # to an error. This is needed because specifying a non complex
#         # dtype to the function converts complex to real dtype,
#         # thereby passing the test made in the lines following the scope
#         # of warnings context manager.
#         with warnings.catch_warnings():
#             try:
#                 warnings.simplefilter("error", ComplexWarning)
#                 if dtype is not None and np.dtype(dtype).kind in "iu":
#                     # Conversion float -> int should not contain NaN or
#                     # inf (numpy#14412). We cannot use casting='safe' because
#                     # then conversion float -> int would be disallowed.
#                     array = _asarray_with_order(array, order=order, xp=xp)
#                     if array.dtype.kind == "f":
#                         _assert_all_finite(
#                             array,
#                             allow_nan=False,
#                             msg_dtype=dtype,
#                             estimator_name=estimator_name,
#                             input_name=input_name,
#                         )
#                     array = xp.astype(array, dtype, copy=False)
#                 else:
#                     array = _asarray_with_order(array, order=order, dtype=dtype, xp=xp)
#             except ComplexWarning as complex_warning:
#                 raise ValueError(
#                     "Complex data not supported\n{}\n".format(array)
#                 ) from complex_warning

#         # It is possible that the np.array(..) gave no warning. This happens
#         # when no dtype conversion happened, for example dtype = None. The
#         # result is that np.array(..) produces an array of complex dtype
#         # and we need to catch and raise exception for such cases.
#         _ensure_no_complex_data(array)

#         if ensure_2d:
#             # If input is scalar raise error
#             if array.ndim == 0:
#                 raise ValueError(
#                     "Expected 2D array, got scalar array instead:\narray={}.\n"
#                     "Reshape your data either using array.reshape(-1, 1) if "
#                     "your data has a single feature or array.reshape(1, -1) "
#                     "if it contains a single sample.".format(array)
#                 )
#             # If input is 1D raise error
#             if array.ndim == 1:
#                 raise ValueError(
#                     "Expected 2D array, got 1D array instead:\narray={}.\n"
#                     "Reshape your data either using array.reshape(-1, 1) if "
#                     "your data has a single feature or array.reshape(1, -1) "
#                     "if it contains a single sample.".format(array)
#                 )

#         if dtype_numeric and array.dtype.kind in "USV":
#             raise ValueError(
#                 "dtype='numeric' is not compatible with arrays of bytes/strings."
#                 "Convert your data to numeric values explicitly instead."
#             )
#         if not allow_nd and array.ndim >= 3:
#             raise ValueError(
#                 "Found array with dim %d. %s expected <= 2."
#                 % (array.ndim, estimator_name)
#             )

#         if force_all_finite:
#             _assert_all_finite(
#                 array,
#                 input_name=input_name,
#                 estimator_name=estimator_name,
#                 allow_nan=force_all_finite == "allow-nan",
#             )

#     if ensure_min_samples > 0:
#         n_samples = _num_samples(array)
#         if n_samples < ensure_min_samples:
#             raise ValueError(
#                 "Found array with %d sample(s) (shape=%s) while a"
#                 " minimum of %d is required%s."
#                 % (n_samples, array.shape, ensure_min_samples, context)
#             )

#     if ensure_min_features > 0 and array.ndim == 2:
#         n_features = array.shape[1]
#         if n_features < ensure_min_features:
#             raise ValueError(
#                 "Found array with %d feature(s) (shape=%s) while"
#                 " a minimum of %d is required%s."
#                 % (n_features, array.shape, ensure_min_features, context)
#             )

#     if copy:
#         if xp.__name__ in {"numpy", "numpy.array_api"}:
#             # only make a copy if `array` and `array_orig` may share memory`
#             if np.may_share_memory(array, array_orig):
#                 array = _asarray_with_order(
#                     array, dtype=dtype, order=order, copy=True, xp=xp
#                 )
#         else:
#             # always make a copy for non-numpy arrays
#             array = _asarray_with_order(
#                 array, dtype=dtype, order=order, copy=True, xp=xp
#             )

#     return array


def check_random_state(seed):
    """Turn seed into a np.random.RandomState instance.
    Parameters
    ----------
    seed : None, int or instance of RandomState
        If seed is None, return the RandomState singleton used by np.random.
        If seed is an int, return a new RandomState instance seeded with seed.
        If seed is already a RandomState instance, return it.
        Otherwise raise ValueError.
    Returns
    -------
    :class:`numpy:numpy.random.RandomState`
        The random state object based on `seed` parameter.
    """
    if seed is None or seed is np.random:
        return np.random.mtrand._rand
    if isinstance(seed, numbers.Integral):
        return np.random.RandomState(seed)
    if isinstance(seed, np.random.RandomState):
        return seed
    raise ValueError(
        "%r cannot be used to seed a numpy.random.RandomState instance" % seed
    )


def _graph_connected_component(graph, node_id):
    """Find the largest graph connected components that contains one
    given node.

    Parameters
    ----------
    graph : array-like of shape (n_samples, n_samples)
        Adjacency matrix of the graph, non-zero weight means an edge
        between the nodes.

    node_id : int
        The index of the query node of the graph.

    Returns
    -------
    connected_components_matrix : array-like of shape (n_samples,)
        An array of bool value indicating the indexes of the nodes
        belonging to the largest connected components of the given query
        node.
    """
    n_node = graph.shape[0]
    if sparse.issparse(graph):
        # speed up row-wise access to boolean connection mask
        graph = graph.tocsr()
    connected_nodes = np.zeros(n_node, dtype=bool)
    nodes_to_explore = np.zeros(n_node, dtype=bool)
    nodes_to_explore[node_id] = True
    for _ in range(n_node):
        last_num_component = connected_nodes.sum()
        np.logical_or(connected_nodes, nodes_to_explore, out=connected_nodes)
        if last_num_component >= connected_nodes.sum():
            break
        indices = np.where(nodes_to_explore)[0]
        nodes_to_explore.fill(False)
        for i in indices:
            if sparse.issparse(graph):
                neighbors = graph[i].toarray().ravel()
            else:
                neighbors = graph[i]
            np.logical_or(nodes_to_explore, neighbors, out=nodes_to_explore)
    return connected_nodes


def _graph_is_connected(graph):
    """Return whether the graph is connected (True) or Not (False).

    Parameters
    ----------
    graph : {array-like, sparse matrix} of shape (n_samples, n_samples)
        Adjacency matrix of the graph, non-zero weight means an edge
        between the nodes.

    Returns
    -------
    is_connected : bool
        True means the graph is fully connected and False means not.
    """
    if sparse.isspmatrix(graph):
        # sparse graph, find all the connected components
        n_connected_components, _ = connected_components(graph)
        return n_connected_components == 1
    else:
        # dense graph, find all connected components start from node 0
        return _graph_connected_component(graph, 0).sum() == graph.shape[0]


def _set_diag(laplacian, value, norm_laplacian):
    """Set the diagonal of the laplacian matrix and convert it to a
    sparse format well suited for eigenvalue decomposition.

    Parameters
    ----------
    laplacian : {ndarray, sparse matrix}
        The graph laplacian.

    value : float
        The value of the diagonal.

    norm_laplacian : bool
        Whether the value of the diagonal should be changed or not.

    Returns
    -------
    laplacian : {array, sparse matrix}
        An array of matrix in a form that is well suited to fast
        eigenvalue decomposition, depending on the band width of the
        matrix.
    """
    n_nodes = laplacian.shape[0]
    # We need all entries in the diagonal to values
    if not sparse.isspmatrix(laplacian):
        if norm_laplacian:
            laplacian.flat[:: n_nodes + 1] = value
    else:
        laplacian = laplacian.tocoo()
        if norm_laplacian:
            diag_idx = laplacian.row == laplacian.col
            laplacian.data[diag_idx] = value
        # If the matrix has a small number of diagonals (as in the
        # case of structured matrices coming from images), the
        # dia format might be best suited for matvec products:
        n_diags = np.unique(laplacian.row - laplacian.col).size
        if n_diags <= 7:
            # 3 or less outer diagonals on each side
            laplacian = laplacian.todia()
        else:
            # csr has the fastest matvec and is thus best suited to
            # arpack
            laplacian = laplacian.tocsr()
    return laplacian


def spectral_embedding(
    adjacency,
    *,
    n_components=8,
    eigen_solver=None,
    random_state=None,
    eigen_tol="auto",
    norm_laplacian=True,
    drop_first=True,
):
    """Project the sample on the first eigenvectors of the graph Laplacian.

    The adjacency matrix is used to compute a normalized graph Laplacian
    whose spectrum (especially the eigenvectors associated to the
    smallest eigenvalues) has an interpretation in terms of minimal
    number of cuts necessary to split the graph into comparably sized
    components.

    This embedding can also 'work' even if the ``adjacency`` variable is
    not strictly the adjacency matrix of a graph but more generally
    an affinity or similarity matrix between samples (for instance the
    heat kernel of a euclidean distance matrix or a k-NN matrix).

    However care must taken to always make the affinity matrix symmetric
    so that the eigenvector decomposition works as expected.

    Note : Laplacian Eigenmaps is the actual algorithm implemented here.

    Read more in the :ref:`User Guide <spectral_embedding>`.

    Parameters
    ----------
    adjacency : {array-like, sparse graph} of shape (n_samples, n_samples)
        The adjacency matrix of the graph to embed.

    n_components : int, default=8
        The dimension of the projection subspace.

    eigen_solver : {'arpack', 'lobpcg', 'amg'}, default=None
        The eigenvalue decomposition strategy to use. AMG requires pyamg
        to be installed. It can be faster on very large, sparse problems,
        but may also lead to instabilities. If None, then ``'arpack'`` is
        used.

    random_state : int, RandomState instance or None, default=None
        A pseudo random number generator used for the initialization
        of the lobpcg eigen vectors decomposition when `eigen_solver ==
        'amg'`, and for the K-Means initialization. Use an int to make
        the results deterministic across calls (See
        :term:`Glossary <random_state>`).

        .. note::
            When using `eigen_solver == 'amg'`,
            it is necessary to also fix the global numpy seed with
            `np.random.seed(int)` to get deterministic results. See
            https://github.com/pyamg/pyamg/issues/139 for further
            information.

    eigen_tol : float, default="auto"
        Stopping criterion for eigendecomposition of the Laplacian matrix.
        If `eigen_tol="auto"` then the passed tolerance will depend on the
        `eigen_solver`:

        - If `eigen_solver="arpack"`, then `eigen_tol=0.0`;
        - If `eigen_solver="lobpcg"` or `eigen_solver="amg"`, then
          `eigen_tol=None` which configures the underlying `lobpcg` solver to
          automatically resolve the value according to their heuristics. See,
          :func:`scipy.sparse.linalg.lobpcg` for details.

        Note that when using `eigen_solver="amg"` values of `tol<1e-5` may lead
        to convergence issues and should be avoided.

        .. versionadded:: 1.2
           Added 'auto' option.

    norm_laplacian : bool, default=True
        If True, then compute symmetric normalized Laplacian.

    drop_first : bool, default=True
        Whether to drop the first eigenvector. For spectral embedding, this
        should be True as the first eigenvector should be constant vector for
        connected graph, but for spectral clustering, this should be kept as
        False to retain the first eigenvector.

    Returns
    -------
    embedding : ndarray of shape (n_samples, n_components)
        The reduced samples.

    Notes
    -----
    Spectral Embedding (Laplacian Eigenmaps) is most useful when the graph
    has one connected component. If there graph has many components, the first
    few eigenvectors will simply uncover the connected components of the graph.

    References
    ----------
    * https://en.wikipedia.org/wiki/LOBPCG

    * :doi:`"Toward the Optimal Preconditioned Eigensolver: Locally Optimal
      Block Preconditioned Conjugate Gradient Method",
      Andrew V. Knyazev
      <10.1137/S1064827500366124>`
    """
    # adjacency = check_symmetric(adjacency)

    try:
        from pyamg import smoothed_aggregation_solver
    except ImportError as e:
        if eigen_solver == "amg":
            raise ValueError(
                "The eigen_solver was set to 'amg', but pyamg is not available."
            ) from e

    # if eigen_solver is None:
    #     eigen_solver = "arpack"
    # elif eigen_solver not in ("arpack", "lobpcg", "amg"):
    #     raise ValueError(
    #         "Unknown value for eigen_solver: '%s'."
    #         "Should be 'amg', 'arpack', or 'lobpcg'" % eigen_solver
    #     )

    random_state = check_random_state(random_state)

    n_nodes = adjacency.shape[0]
    # Whether to drop the first eigenvector
    if drop_first:
        n_components = n_components + 1

    if not _graph_is_connected(adjacency):
        warnings.warn(
            "Graph is not fully connected, spectral embedding may not work as expected."
        )

    laplacian, dd = csgraph_laplacian(
        adjacency, normed=norm_laplacian, return_diag=True
    )
    if (
        eigen_solver == "arpack"
        or eigen_solver != "lobpcg"
        and (not sparse.isspmatrix(laplacian) or n_nodes < 5 * n_components)
    ):
        # lobpcg used with eigen_solver='amg' has bugs for low number of nodes
        # for details see the source code in scipy:
        # https://github.com/scipy/scipy/blob/v0.11.0/scipy/sparse/linalg/eigen
        # /lobpcg/lobpcg.py#L237
        # or matlab:
        # https://www.mathworks.com/matlabcentral/fileexchange/48-lobpcg-m
        laplacian = _set_diag(laplacian, 1, norm_laplacian)

        # Here we'll use shift-invert mode for fast eigenvalues
        # (see https://docs.scipy.org/doc/scipy/reference/tutorial/arpack.html
        #  for a short explanation of what this means)
        # Because the normalized Laplacian has eigenvalues between 0 and 2,
        # I - L has eigenvalues between -1 and 1.  ARPACK is most efficient
        # when finding eigenvalues of largest magnitude (keyword which='LM')
        # and when these eigenvalues are very large compared to the rest.
        # For very large, very sparse graphs, I - L can have many, many
        # eigenvalues very near 1.0.  This leads to slow convergence.  So
        # instead, we'll use ARPACK's shift-invert mode, asking for the
        # eigenvalues near 1.0.  This effectively spreads-out the spectrum
        # near 1.0 and leads to much faster convergence: potentially an
        # orders-of-magnitude speedup over simply using keyword which='LA'
        # in standard mode.
        try:
            # We are computing the opposite of the laplacian inplace so as
            # to spare a memory allocation of a possibly very large array
            tol = 0 if eigen_tol == "auto" else eigen_tol
            laplacian *= -1
            v0 = _init_arpack_v0(laplacian.shape[0], random_state)
            _, diffusion_map = eigsh(
                laplacian, k=n_components, sigma=1.0, which="LM", tol=tol, v0=v0
            )
            embedding = diffusion_map.T[n_components::-1]
            if norm_laplacian:
                # recover u = D^-1/2 x from the eigenvector output x
                embedding = embedding / dd
        except RuntimeError:
            # When submatrices are exactly singular, an LU decomposition
            # in arpack fails. We fallback to lobpcg
            eigen_solver = "lobpcg"
            # Revert the laplacian to its opposite to have lobpcg work
            laplacian *= -1

    elif eigen_solver == "amg":
        # Use AMG to get a preconditioner and speed up the eigenvalue
        # problem.
        if not sparse.issparse(laplacian):
            warnings.warn("AMG works better for sparse matrices")
        # laplacian = check_array(
        #     laplacian, dtype=[np.float64, np.float32], accept_sparse=True
        # )
        laplacian = _set_diag(laplacian, 1, norm_laplacian)

        # The Laplacian matrix is always singular, having at least one zero
        # eigenvalue, corresponding to the trivial eigenvector, which is a
        # constant. Using a singular matrix for preconditioning may result in
        # random failures in LOBPCG and is not supported by the existing
        # theory:
        #     see https://doi.org/10.1007/s10208-015-9297-1
        # Shift the Laplacian so its diagononal is not all ones. The shift
        # does change the eigenpairs however, so we'll feed the shifted
        # matrix to the solver and afterward set it back to the original.
        diag_shift = 1e-5 * sparse.eye(laplacian.shape[0])
        laplacian += diag_shift
        ml = smoothed_aggregation_solver(laplacian)
        laplacian -= diag_shift

        M = ml.aspreconditioner()
        # Create initial approximation X to eigenvectors
        X = random_state.standard_normal(size=(laplacian.shape[0], n_components + 1))
        X[:, 0] = dd.ravel()
        X = X.astype(laplacian.dtype)

        tol = None if eigen_tol == "auto" else eigen_tol
        _, diffusion_map = lobpcg(laplacian, X, M=M, tol=tol, largest=False)
        embedding = diffusion_map.T
        if norm_laplacian:
            # recover u = D^-1/2 x from the eigenvector output x
            embedding = embedding / dd
        if embedding.shape[0] == 1:
            raise ValueError

    if eigen_solver == "lobpcg":
        # laplacian = check_array(
        #     laplacian, dtype=[np.float64, np.float32], accept_sparse=True
        # )
        if n_nodes < 5 * n_components + 1:
            # see note above under arpack why lobpcg has problems with small
            # number of nodes
            # lobpcg will fallback to eigh, so we short circuit it
            if sparse.isspmatrix(laplacian):
                laplacian = laplacian.toarray()
            _, diffusion_map = eigh(laplacian, check_finite=False)
            embedding = diffusion_map.T[:n_components]
            if norm_laplacian:
                # recover u = D^-1/2 x from the eigenvector output x
                embedding = embedding / dd
        else:
            laplacian = _set_diag(laplacian, 1, norm_laplacian)
            # We increase the number of eigenvectors requested, as lobpcg
            # doesn't behave well in low dimension and create initial
            # approximation X to eigenvectors
            X = random_state.standard_normal(
                size=(laplacian.shape[0], n_components + 1)
            )
            X[:, 0] = dd.ravel()
            X = X.astype(laplacian.dtype)
            tol = None if eigen_tol == "auto" else eigen_tol
            _, diffusion_map = lobpcg(
                laplacian, X, tol=tol, largest=False, maxiter=2000
            )
            embedding = diffusion_map.T[:n_components]
            if norm_laplacian:
                # recover u = D^-1/2 x from the eigenvector output x
                embedding = embedding / dd
            if embedding.shape[0] == 1:
                raise ValueError

    embedding = _deterministic_vector_sign_flip(embedding)
    if drop_first:
        return embedding[1:n_components].T
    else:
        return embedding[:n_components].T


# class SpectralEmbedding(BaseEstimator):
    """Spectral embedding for non-linear dimensionality reduction.

    Forms an affinity matrix given by the specified function and
    applies spectral decomposition to the corresponding graph laplacian.
    The resulting transformation is given by the value of the
    eigenvectors for each data point.

    Note : Laplacian Eigenmaps is the actual algorithm implemented here.

    Read more in the :ref:`User Guide <spectral_embedding>`.

    Parameters
    ----------
    n_components : int, default=2
        The dimension of the projected subspace.

    affinity : {'nearest_neighbors', 'rbf', 'precomputed', \
                'precomputed_nearest_neighbors'} or callable, \
                default='nearest_neighbors'
        How to construct the affinity matrix.
         - 'nearest_neighbors' : construct the affinity matrix by computing a
           graph of nearest neighbors.
         - 'rbf' : construct the affinity matrix by computing a radial basis
           function (RBF) kernel.
         - 'precomputed' : interpret ``X`` as a precomputed affinity matrix.
         - 'precomputed_nearest_neighbors' : interpret ``X`` as a sparse graph
           of precomputed nearest neighbors, and constructs the affinity matrix
           by selecting the ``n_neighbors`` nearest neighbors.
         - callable : use passed in function as affinity
           the function takes in data matrix (n_samples, n_features)
           and return affinity matrix (n_samples, n_samples).

    gamma : float, default=None
        Kernel coefficient for rbf kernel. If None, gamma will be set to
        1/n_features.

    random_state : int, RandomState instance or None, default=None
        A pseudo random number generator used for the initialization
        of the lobpcg eigen vectors decomposition when `eigen_solver ==
        'amg'`, and for the K-Means initialization. Use an int to make
        the results deterministic across calls (See
        :term:`Glossary <random_state>`).

        .. note::
            When using `eigen_solver == 'amg'`,
            it is necessary to also fix the global numpy seed with
            `np.random.seed(int)` to get deterministic results. See
            https://github.com/pyamg/pyamg/issues/139 for further
            information.

    eigen_solver : {'arpack', 'lobpcg', 'amg'}, default=None
        The eigenvalue decomposition strategy to use. AMG requires pyamg
        to be installed. It can be faster on very large, sparse problems.
        If None, then ``'arpack'`` is used.

    eigen_tol : float, default="auto"
        Stopping criterion for eigendecomposition of the Laplacian matrix.
        If `eigen_tol="auto"` then the passed tolerance will depend on the
        `eigen_solver`:

        - If `eigen_solver="arpack"`, then `eigen_tol=0.0`;
        - If `eigen_solver="lobpcg"` or `eigen_solver="amg"`, then
          `eigen_tol=None` which configures the underlying `lobpcg` solver to
          automatically resolve the value according to their heuristics. See,
          :func:`scipy.sparse.linalg.lobpcg` for details.

        Note that when using `eigen_solver="lobpcg"` or `eigen_solver="amg"`
        values of `tol<1e-5` may lead to convergence issues and should be
        avoided.

        .. versionadded:: 1.2

    n_neighbors : int, default=None
        Number of nearest neighbors for nearest_neighbors graph building.
        If None, n_neighbors will be set to max(n_samples/10, 1).

    n_jobs : int, default=None
        The number of parallel jobs to run.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    Attributes
    ----------
    embedding_ : ndarray of shape (n_samples, n_components)
        Spectral embedding of the training matrix.

    affinity_matrix_ : ndarray of shape (n_samples, n_samples)
        Affinity_matrix constructed from samples or precomputed.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

        .. versionadded:: 0.24

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

        .. versionadded:: 1.0

    n_neighbors_ : int
        Number of nearest neighbors effectively used.

    See Also
    --------
    Isomap : Non-linear dimensionality reduction through Isometric Mapping.

    References
    ----------

    - :doi:`A Tutorial on Spectral Clustering, 2007
      Ulrike von Luxburg
      <10.1007/s11222-007-9033-z>`

    - `On Spectral Clustering: Analysis and an algorithm, 2001
      Andrew Y. Ng, Michael I. Jordan, Yair Weiss
      <https://citeseerx.ist.psu.edu/doc_view/pid/796c5d6336fc52aa84db575fb821c78918b65f58>`_

    - :doi:`Normalized cuts and image segmentation, 2000
      Jianbo Shi, Jitendra Malik
      <10.1109/34.868688>`

    Examples
    --------
    >>> from sklearn.datasets import load_digits
    >>> from sklearn.manifold import SpectralEmbedding
    >>> X, _ = load_digits(return_X_y=True)
    >>> X.shape
    (1797, 64)
    >>> embedding = SpectralEmbedding(n_components=2)
    >>> X_transformed = embedding.fit_transform(X[:100])
    >>> X_transformed.shape
    (100, 2)
    """

    _parameter_constraints: dict = {
        "n_components": [Interval(Integral, 1, None, closed="left")],
        "affinity": [
            StrOptions(
                {
                    "nearest_neighbors",
                    "rbf",
                    "precomputed",
                    "precomputed_nearest_neighbors",
                },
            ),
            callable,
        ],
        "gamma": [Interval(Real, 0, None, closed="left"), None],
        "random_state": ["random_state"],
        "eigen_solver": [StrOptions({"arpack", "lobpcg", "amg"}), None],
        "eigen_tol": [Interval(Real, 0, None, closed="left"), StrOptions({"auto"})],
        "n_neighbors": [Interval(Integral, 1, None, closed="left"), None],
        "n_jobs": [None, Integral],
    }

    def __init__(
        self,
        n_components=2,
        *,
        affinity="nearest_neighbors",
        gamma=None,
        random_state=None,
        eigen_solver=None,
        eigen_tol="auto",
        n_neighbors=None,
        n_jobs=None,
    ):
        self.n_components = n_components
        self.affinity = affinity
        self.gamma = gamma
        self.random_state = random_state
        self.eigen_solver = eigen_solver
        self.eigen_tol = eigen_tol
        self.n_neighbors = n_neighbors
        self.n_jobs = n_jobs

    def _more_tags(self):
        return {
            "pairwise": self.affinity
            in ["precomputed", "precomputed_nearest_neighbors"]
        }

    def _get_affinity_matrix(self, X, Y=None):
        """Calculate the affinity matrix from data
        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Training vector, where `n_samples` is the number of samples
            and `n_features` is the number of features.

            If affinity is "precomputed"
            X : array-like of shape (n_samples, n_samples),
            Interpret X as precomputed adjacency graph computed from
            samples.

        Y: Ignored

        Returns
        -------
        affinity_matrix of shape (n_samples, n_samples)
        """
        if self.affinity == "precomputed":
            self.affinity_matrix_ = X
            return self.affinity_matrix_
        if self.affinity == "precomputed_nearest_neighbors":
            estimator = NearestNeighbors(
                n_neighbors=self.n_neighbors, n_jobs=self.n_jobs, metric="precomputed"
            ).fit(X)
            connectivity = estimator.kneighbors_graph(X=X, mode="connectivity")
            self.affinity_matrix_ = 0.5 * (connectivity + connectivity.T)
            return self.affinity_matrix_
        if self.affinity == "nearest_neighbors":
            if sparse.issparse(X):
                warnings.warn(
                    "Nearest neighbors affinity currently does "
                    "not support sparse input, falling back to "
                    "rbf affinity"
                )
                self.affinity = "rbf"
            else:
                self.n_neighbors_ = (
                    self.n_neighbors
                    if self.n_neighbors is not None
                    else max(int(X.shape[0] / 10), 1)
                )
                self.affinity_matrix_ = kneighbors_graph(
                    X, self.n_neighbors_, include_self=True, n_jobs=self.n_jobs
                )
                # currently only symmetric affinity_matrix supported
                self.affinity_matrix_ = 0.5 * (
                    self.affinity_matrix_ + self.affinity_matrix_.T
                )
                return self.affinity_matrix_
        if self.affinity == "rbf":
            self.gamma_ = self.gamma if self.gamma is not None else 1.0 / X.shape[1]
            self.affinity_matrix_ = rbf_kernel(X, gamma=self.gamma_)
            return self.affinity_matrix_
        self.affinity_matrix_ = self.affinity(X)
        return self.affinity_matrix_

    def fit(self, X, y=None):
        """Fit the model from data in X.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training vector, where `n_samples` is the number of samples
            and `n_features` is the number of features.

            If affinity is "precomputed"
            X : {array-like, sparse matrix}, shape (n_samples, n_samples),
            Interpret X as precomputed adjacency graph computed from
            samples.

        y : Ignored
            Not used, present for API consistency by convention.

        Returns
        -------
        self : object
            Returns the instance itself.
        """
        self._validate_params()

        X = self._validate_data(X, accept_sparse="csr", ensure_min_samples=2)

        random_state = check_random_state(self.random_state)

        affinity_matrix = self._get_affinity_matrix(X)
        self.embedding_ = spectral_embedding(
            affinity_matrix,
            n_components=self.n_components,
            eigen_solver=self.eigen_solver,
            eigen_tol=self.eigen_tol,
            random_state=random_state,
        )
        return self

    def fit_transform(self, X, y=None):
        """Fit the model from data in X and transform X.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Training vector, where `n_samples` is the number of samples
            and `n_features` is the number of features.

            If affinity is "precomputed"
            X : {array-like, sparse matrix} of shape (n_samples, n_samples),
            Interpret X as precomputed adjacency graph computed from
            samples.

        y : Ignored
            Not used, present for API consistency by convention.

        Returns
        -------
        X_new : array-like of shape (n_samples, n_components)
            Spectral embedding of the training matrix.
        """
        self.fit(X)
        return self.embedding_