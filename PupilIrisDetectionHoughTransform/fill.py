import heapq

import numpy as np
from scipy import ndimage


def fast_fill(input_array, four_way=False):
    """
    Fast flood fill depressions/sinks in floating point array

    Parameters
    ----------
    input_array : ndarray
        Input array to be filled
    four_way : bool, optional
        If True, search 4 immediately adjacent cells (cross structuring element)
        If False, search all 8 adjacent cells (square structuring element).
        The Default is False.

    Returns
    -------
    out : ndarray
        Filled array

    References
    ----------
    Soille and Gratin, 1994. An Efficient Algorithm for Drainage Networks 
        Extraction on DEMs. Journal of Visual Communication and Image 
        Representation, 5(2), 181-189
    Liu et al., 2009. Another Fast and Simple DEM Depression-Filling 
        Algorithm Based on Priority Queue Structure. Atmopsheric and Oceanic 
        Science Letters, 2(4) 214-219

    """
    print('Fast Fill')

    # Rename or copy input so that input_array is a local variable?
    # input_array = np.copy(input_array)
     
    # Set h_max to a value larger than the array maximum to ensure
    #   that the while loop will terminate
    h_max = np.max(input_array * 2.0)
     
    # Build mask of cells with data not on the edge of the image
    # Use 3x3 square structuring element
    # Build Structuring element only using NumPy module
    # Structuring element could also be built using SciPy ndimage module
    #   el = ndimage.generate_binary_structure(2,2).astype(np.int)
    data_mask = np.isfinite(input_array)
    inside_mask = ndimage.binary_erosion(
        data_mask, 
        structure=np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]]).astype(np.bool))
    edge_mask = (data_mask & ~inside_mask)
     
    # Initialize output array as max value test_array except edges
    output_array = np.copy(input_array)
    output_array[inside_mask] = h_max
     
    # Build priority queue and place edge pixels into priority queue
    # Last value is flag to indicate if cell is an edge cell
    put = heapq.heappush
    get = heapq.heappop
    fill_heap = [
        (output_array[t_row, t_col], int(t_row), int(t_col), True)
        for t_row, t_col in np.transpose(np.where(edge_mask))]
    heapq.heapify(fill_heap)
 
    def neighbors(row, col, four_way=False):
        """Return indices of adjacent cells"""
        if four_way:
            return [
                (row - 1, col), (row, col + 1), 
                (row + 1, col), (row, col - 1)]
        else:
            return [
                (row - 1, col), (row - 1, col + 1), 
                (row, col + 1), (row + 1, col + 1), 
                (row + 1, col), (row + 1, col - 1), 
                (row, col - 1), (row - 1, col - 1)]

    # Iterate until priority queue is empty
    while True:
        try: 
            h_crt, t_row, t_col, edge_flag = get(fill_heap)
        except IndexError: 
            break
        for n_row, n_col in neighbors(t_row, t_col, four_way):
            # Skip cell if outside array edges
            if edge_flag:
                try:
                    if not inside_mask[n_row, n_col]: 
                        continue
                except IndexError: 
                    continue
            if output_array[n_row, n_col] == h_max:
                output_array[n_row, n_col] = max(
                    h_crt, input_array[n_row, n_col])
                put(fill_heap, (output_array[n_row, n_col], n_row, n_col, False))
    return output_array


def slow_fill(input_array, four_way=False):
    """
    Slow flood fill depressions/sinks in floating point array

    Parameters
    ----------
    input_array : ndarray
        Input array to be filled
    four_way : bool, optional
        If True, search 4 immediately adjacent cells (cross structuring element)
        If False, search all 8 adjacent cells (square structuring element).
        The Default is False.

    Returns
    -------
    out : ndarray
        Filled array

    References
    ----------
    Soile, P., Vogt, J., and Colombo, R., 2003. Carving and Adaptive Drainage Enforcement of Grid Digital Elevation Models. Water Resources Research, 39(12), 1366
    Soille, P., 1999. Morphological Image Analysis: Principles and Applications, Springer-Verlag, pp. 173-174

    """
    #print('Slow Fill')

    # Rename or copy input so that input_array is a local variable?
    # input_array = np.copy(input_array)
 
    # Set h_max to a value larger than the array maximum to ensure
    #   that the while loop will terminate
    h_max = np.max(input_array * 2.0)

    # Build mask of cells with data not on the edge of the image
    # Use 3x3 square Structuring element
    # Build Structuring element only using NumPy module
    # Structuring element could also be built using SciPy ndimage module
    #   el = ndimage.generate_binary_structure(2,2).astype(np.int)
    inside_mask = ndimage.binary_erosion(
        np.isfinite(input_array), 
        structure=np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]]).astype(np.bool))
     
    # Initialize output array as max value test_array except edges
    output_array = np.copy(input_array)
    output_array[inside_mask] = h_max
 
    # Array for storing previous iteration
    output_old_array = np.copy(input_array)
    output_old_array[:] = 0
 
    # Cross structuring element
    if four_way:
        el = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]]).astype(np.bool)
        # el = ndimage.generate_binary_structure(2, 1).astype(np.int)
    else:
        el = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]]).astype(np.bool)
        # el = ndimage.generate_binary_structure(2, 2).astype(np.int)
 
    # Iterate until marker array doesn't change
    while not np.array_equal(output_old_array, output_array):
        output_old_array = np.copy(output_array)
        output_array = np.maximum(
            input_array,
            ndimage.grey_erosion(output_array, size=(3, 3), footprint=el))
    return output_array


def slow_fill1(input_array, four_way=False):
    """
    Slow flood fill depressions/sinks in floating point array

    Parameters
    ----------
    input_array : ndarray
        Input array to be filled
    four_way : bool, optional
        If True, search 4 immediately adjacent cells
            (cross structuring element)
        If False, search all 8 adjacent cells
            (square structuring element).
        The Default is False.

    Returns
    -------
    out : ndarray
        Filled array

    References
    ----------
    .. [3] Soile, P., Vogt, J., and Colombo, R., 2003. Carving and Adaptive
        Drainage Enforcement of Grid Digital Elevation Models.
        Water Resources Research, 39(12), 1366
    .. [4] Soille, P., 1999. Morphological Image Analysis: Principles and
        Applications, Springer-Verlag, pp. 173-174

    """
    print('Slow Fill')

    # Rename or copy input so that input_array is a local variable?
    # input_array = np.copy(input_array)

    # Set h_max to a value larger than the array maximum to ensure
    #   that the while loop will terminate
    h_max = input_array.max() + 100

    # Build mask of cells with data not on the edge of the image
    # Use 3x3 square Structuring element
    inside_mask = np_binary_erosion(
        np.isfinite(input_array),
        structure=np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]]).astype(np.bool))

    # Initialize output array as max value test_array except edges
    output_array = np.copy(input_array)
    output_array[inside_mask] = h_max

    # Array for storing previous iteration
    output_old_array = np.copy(input_array)
    output_old_array[:] = 0

    # Cross structuring element
    if four_way:
        el = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]]).astype(np.bool)
    else:
        el = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]]).astype(np.bool)

    # Iterate until marker array doesn't change
    while not np.array_equal(output_old_array, output_array):
        output_old_array = np.copy(output_array)
        output_array = np.maximum(
            input_array,
            ndimage.grey_erosion(output_array, size=(3, 3), footprint=el))
    return output_array


def np_binary_erosion(input_array,
                      structure=np.ones((3, 3)).astype(np.bool)):
    """
    Multi-dimensional binary erosion with a given structuring element.

    Binary erosion is a mathematical morphology operation used for image
    processing.

    Notes
    -----
    Pure NumPy replacement for SciPy ndimage.binary_erosion()
    No error checking on input array (type)
    No error checking on structure element (# of dimensions, shape, type, etc.)

    Parameters
    ----------
    input : array_like
        Binary image to be eroded. Non-zero (True) elements form
        the subset to be eroded.
    structure : array_like, optional
        Structuring element used for the erosion. Non-zero elements are
        considered True. If no structuring element is provided, an element
        is generated with a square connectivity equal to one.

    Returns
    -------
    binary_erosion: Erosion of the input by the stucturing element

    """
    rows, cols = input_array.shape

    # Pad output array (binary_erosion) with extra cells around the edge
    # so that structuring element will fit without wrapping.
    # A 3x3 structure, will need 1 additional cell around the edge
    # A 5x5 structure, will need 2 additional cells around the edge
    output_shape = tuple(
        ss + dd - 1 for ss, dd in zip(input_array.shape, structure.shape))
    input_pad_array = np.zeros(output_shape).astype(np.bool)
    input_pad_array[1: rows+1, 1: cols+1] = input_array
    binary_erosion = np.zeros(output_shape).astype(np.bool)

    # Cast structure element to boolean
    struc_mask = structure.astype(np.bool)

    # Iterate over each cell
    for row in range(rows):
        for col in range(cols):
            # The value of the output pixel is the minimum value of all the
            #   pixels in the input pixel's neighborhood.
            binary_erosion[row+1, col+1] = np.min(
                input_pad_array[row: row+3, col: col+3][struc_mask])

    return binary_erosion[1: rows+1, 1: cols+1]