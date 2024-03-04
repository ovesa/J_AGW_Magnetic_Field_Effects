import numpy as np
import numpy.ma as ma
import copy


def compute_magnetic_nonmagnetic_inds(magnetogram, mag_val):
    """
    Finds indices of pixels that are larger and smaller than
    the mag_val of the magnetogram.

    Input:
    magnetogram = magnetogram [3D datacube]
    mag_val = chosen magnetogram value for masking [G]

    Output:
    locs_above_med = indices of pixels above the mag_val
    locs_below_med = indices of pixels below the mag_val
    """

    magmap = np.abs(copy.deepcopy(magnetogram))

    locs_above_med = np.where((magmap) >= mag_val)
    locs_below_med = np.where((magmap) < mag_val)

    return locs_above_med, locs_below_med


def inds_on_mag(magnetogram, locs):
    """
    This function takes in the magnetogram and the indices
    provided by the compute_magnetic_nonmagnetic_inds function.
    Indices above or below the chosen_value are set to 0.

    Input:
    magnetogram = magnetogram [3D datacube]
    locs = indices of pixels that are above or below the chosen_value

    Output:
    ibis_abs = masked array [3D datacube]
    """
    ibis_abs = np.abs(copy.deepcopy(magnetogram))
    ibis_abs[locs] = 0
    return ibis_abs


def masked_magnetic_maps(magnetogram_array, mask_value):
    """
    Mask the magnetogram based on the chosen value.

    Arguments:
        magnetogram_array -- The magnetogram time series
        mask_value -- The chosen value for the masking process [G]

    Returns:
        The masked magnetogram larger than the chosen value and the
        masked magnetogram smaller than or equal to the chosen value.
    """

    # Copy magnetogram
    cop_mag = np.abs(copy.deepcopy(magnetogram_array))

    # Masked magnetogram less than or equal to the chosen value
    magnetic_map = ma.masked_less_equal(cop_mag, mask_value, copy=True)

    # Masked magnetogram greater than the chosen value
    nonmagnetic_map = ma.masked_greater(cop_mag, mask_value, copy=True)
    return magnetic_map, nonmagnetic_map


def masked_IBIS_cubes_based_on_masked_magnetogram(diagnostic_map, magnetic_map_mask):
    """
    Mask the pixels in the IBIS map that corresponds to the masked magnetogram
    that is less than or equal to the chosen magnetic value.

    Arguments:
        diagnostic_map -- IBIS diagnostic map to mask
        magnetic_map_mask -- Masked magnetogram

    Returns:
        The masked IBIS diagnostic map corresponding
    """

    # Copy IBIS diagnostic map
    coparr = copy.deepcopy(diagnostic_map)

    # Obtain mask corresponding the magnetogram
    grab_mask = np.ma.getmask(copy.deepcopy(magnetic_map_mask))

    # Mask IBIS diagnostic map
    masked_cube = np.ma.masked_array(
        coparr, mask=grab_mask, fill_value=0, hard_mask=True
    )
    return masked_cube
