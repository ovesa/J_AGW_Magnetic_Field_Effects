import numpy as np
import numpy.fft as fft
from tqdm import tqdm
import copy


def conversion_arcseconds_to_Mm(distance):
    """Convert arcseconds on the Sun as viewed from Earth
    to Megameters.
    NOTE: 1 arsecond ~= 0.713 Mm

    Arguments:
        distance -- Distance between the Sun and Earth [km].

    Returns:
        Factor to convert arcseconds as seen on Earth to Mm on the Sun [Mm].
    """

    distance_between_Sun_and_Earth = distance  # 147.1e6[] km]

    convert_arcsecond_to_rad = (1 / 3600) * (np.pi / 180)  # [rad]

    conversion_calculation = (
        distance_between_Sun_and_Earth * convert_arcsecond_to_rad
    )  # [km]

    convert_arcseconds_to_Mm = (conversion_calculation * 1000) * 10 ** (-6)  # [Mm]
    return convert_arcseconds_to_Mm


def cross_spectrum(time_series1, time_series2):
    """Compute the complex cross-spectrum of two different
    times series (which correspond to different wavelengths and
    different atmospheric heights). If the two inputs are the
    same, a power spectrum is computed. This code assumes that
    the cubes have been detrended and/or apodized.

    Input:
    time_series1 = Time series correspoding to diagnostic that forms lower in the atmosphere [x,y,t]
    time_series2 = Time series correspoding to diagnostic that forms higher in the atmosphere [x,y,t]

    Output:
    cross_spectrum = A complex cross-spectrum that is the same size as the inputs.
    """

    # Compute N-Dimension FFT of time series
    fft_1 = fft.fftn(time_series1)
    fft_2 = fft.fftn(time_series2)

    # Compute cross spectrum
    # and shift cube to align frequencies
    cross_spectrum = fft.fftshift(fft_1 * np.conjugate(fft_2))

    return cross_spectrum


def power_spectrum(time_series):
    """Compute the 3D power spectrum.

     Arguments:
         time_series -- Time series array [x,y,t].

    Returns:
         Power spectrum that is the same size as the inputs
    """

    # Compute N-Dimension FFT of time series
    # and shift cube to align frequencies
    fshift = fft.fftshift(fft.fftn(time_series))

    # Compute power
    power = np.abs(fshift) ** 2
    return power


def compute_amplitude_spectrum(time_series):
    """Compute the amplitude spectrum from a velocity time series
    with the proper units [km/s].
    NOTE: Python's forward DFT has no normalization factor.

    Input:
    time_series = velocity time series [x,y,t]

    Output:
    amplitude = amplitude spectrum [km/s]
    """

    # Compute N-Dimension FFT of time series
    # and shift cube to align frequencies
    fshift = fft.fftshift(fft.fftn(time_series))

    # Compute power
    power = np.abs(fshift) ** 2

    # Normalize power by size
    power = power / power.size**2

    # Compute amplitude
    amplitude = np.sqrt(4 * power)
    return amplitude


def phase_difference_correction(omega, mid_time, mid_space, dtau):
    """Compute the linear phase difference correction to the azimuthally
    averaged phase difference spectrum. It accounts for the time delay
    caused by the sequentual sampling of the IBIS line core diagnostics.

    Input:
    omega = Nyquist frequency [rad/s]
    mid_time = half the length of the time domain
    mid_space = half the length of a spatial domain
    dtau = time lag [s]

    Output:
    Phase difference correction matching size of the
    azimuthally averaged datacube.
    """

    # frequency correction - del phi = omega * phase time
    phase_correction = np.linspace(0.0, omega, mid_time) * dtau

    # replicate vector k times
    phase_correction = np.tile(phase_correction, (mid_space, 1))

    # transpose to match shape of azimuthally averaged cross power
    phase_correction = phase_correction
    return phase_correction


def phase_difference_correction_3D(omega, mid_time, mid_space, dtau):
    """Compute the linear phase difference correction to the azimuthally
    averaged phase difference spectrum. It accounts for the time delay
    caused by the sequentual sampling of the IBIS line core diagnostics.

    Input:
    omega = Nyquist frequency [rad/s]
    mid_time = half the length of the time domain
    mid_space = half the length of a spatial domain
    dtau = time lag [s]

    Output:
    Phase difference correction matching size of datacube
    """

    # frequency correction - del phi = omega * phase time
    phase_correction = np.linspace(0.0, omega, mid_time * 2) * dtau
    # replicate vector k times
    phase_correction = np.tile(phase_correction, (mid_space, mid_space, 1))
    # transpose to match shape of azimuthally averaged cross power
    phase_correction = phase_correction
    return phase_correction


def find(condition):
    """Returns a flattened array of indices related to the condition that are
    non-zero.

    Input:
    condition = some condition which should be true or false (think np.where)

    Output:
    result = outputs the indices of the array that matches that condition.
    """

    # returns indices that are non-zero in the flattened array
    (result,) = np.nonzero(np.ravel(condition, order="C"))
    return result


def azimuthal_averaging(mid_time, end_time, array, mid_space, radial_meshgrid):
    """Compute an azimuthally averaged 2D data cube from a 3D FFT cube.
    Assumed array is a square.

    Input:
    mid_time = Half the total time array. Number of positive frequencies
    end_time = Total length of the time/frequency array
    array = 3D array to azimuthally average [x,y,t]
    mid_space = Half the total length the spatial dimensions (kx or ky array)
    radial_meshgrid = A grid that maps radius for each pixel


    output:
    azim = azimuthally averaged array (2D)
    """
    # Array to store azimuthally averaged datacube
    if end_time % 2 == 0:
        azim = np.zeros([mid_space, mid_time])
    else:
        azim = np.zeros([mid_space, mid_time + 1])

    # Copy array
    coparr = copy.deepcopy(array)

    # Pixel size to azimuthally average over
    w = 0.5

    # Avoid Negative Frequencies -- only grab indices corresponding to the positive frequencies
    for j in tqdm(range(mid_time, end_time), desc="Azimuthal Averaging"):

        # 3D FFT array is read in
        arr_product = coparr[:, :, j]

        # Ignore Negative Spatial Kx, Ky
        for k in range(1, int(mid_space) + 1):

            # Result is an array showing the truth values element wise
            desired_condition = np.logical_and(
                radial_meshgrid >= k - w, radial_meshgrid < k + w
            )

            # Returns the indices that are non-zero from the flattened array where the condition corresponds to True
            inds = find(desired_condition == True)

            # Returns a collapsed array (1-D)
            flat_array = arr_product.flatten(order="C")

            # Corresponding values within the flattened array that match the desired condition and average them
            aa = np.mean(flat_array[inds])

            azim[k - 1, j - int(mid_time)] = aa
    return azim


# def azimuthal_averaging_td(mid_time, end_time, array, mid_space, radial_meshgrid):
#     """Computes an azimuthally averaged 2D data cube from a 3D FFT cube.
#     Assumed array is a square.

#     Input:

#     mid_time = Half the total time array. Number of positive frequencies
#     end_time = Total length of the time/frequency array
#     array = 3D array to azimuthally average
#     mid_space = Half the total length the spatial dimensions (kx or ky array)
#     radial_meshgrid = A grid that maps radius for each pixel


#     output:
#     azim = azimuthally averaged array (2D)
#     """

#     # initializes a matrix full of 0s
#     # for the azimuthally averaged product
#     azim = np.zeros([mid_space, mid_time + 1])

#     # pixel size to azimuthally average over
#     w = 0.5

#     # Avoid Negative Frequencies
#     # Only grab indices corresponding to the positive frequencies
#     for j in tqdm(range(mid_time, end_time), desc="Azimuthal Averaging"):

#         # 3D FFT array is read in
#         arr_product = array[:, :, j]

#         # Ignore Negative Spatial Kx, Ky
#         for k in range(1, int(mid_space) + 1):

#             # result is an array showing the truth values element wise
#             desired_condition = np.logical_and(
#                 radial_meshgrid >= k - w, radial_meshgrid < k + w
#             )

#             # returns the indices that are non-zero from the flattened array
#             # where the condition corresponds to True
#             inds = find(desired_condition == True)

#             # returns a collapsed array (1-D)
#             flat_array = arr_product.flatten(order="C")

#             # Corresponding values within the flattened array that match the
#             # desired condition
#             # and average them
#             aa = np.mean(flat_array[inds])

#             azim[k - 1, j - int(mid_time)] = aa
#     return azim


# def azimuthal_averaging_running_difference(
#     mid_time, end_time, array, mid_space, radial_meshgrid
# ):
#     """Computes an azimuthally averaged 2D data cube from a 3D FFT cube.
#     Assumed array is a square.

#     Input:

#     mid_time = Half the total time array. Number of positive frequencies
#     end_time = Total length of the time/frequency array
#     array = 3D array to azimuthally average
#     mid_space = Half the total length the spatial dimensions (kx or ky array)
#     radial_meshgrid = A grid that maps radius for each pixel


#     output:
#     azim = azimuthally averaged array (2D)
#     """

#     # Array to store azimuthally averaged datacube
#     azim = np.zeros([mid_space, mid_time + 1])

#     # Copy array
#     coparr = copy.deepcopy(array)

#     # Pixel size to azimuthally average over
#     w = 0.5

#     # Avoid Negative Frequencies -- only grab indices corresponding to the positive frequencies
#     for j in tqdm(range(mid_time, end_time), desc="Azimuthal Averaging"):

#         # 3D FFT array is read in
#         arr_product = coparr[:, :, j]

#         # Ignore Negative Spatial Kx, Ky
#         for k in range(1, int(mid_space) + 1):

#             # Result is an array showing the truth values element wise
#             desired_condition = np.logical_and(
#                 radial_meshgrid >= k - w, radial_meshgrid < k + w
#             )

#             # Returns the indices that are non-zero from the flattened array where the condition corresponds to True
#             inds = find(desired_condition == True)

#             # Returns a collapsed array (1-D)
#             flat_array = arr_product.flatten(order="C")

#             # Corresponding values within the flattened array that match the desired condition and average them
#             aa = np.mean(flat_array[inds])

#             azim[k - 1, j - int(mid_time)] = aa
#     return azim


# def azimuthal_averaging_coherence(
#     mid_time, end_time, array, mid_space, radial_meshgrid
# ):
#     """Computes an azimuthally averaged 2D data cube from a 3D FFT cube.
#     Assumed array is a square.

#     Input:

#     mid_time = Half the total time array. Number of positive frequencies
#     end_time = Total length of the time/frequency array
#     array = 3D array to azimuthally average
#     mid_space = Half the total length the spatial dimensions (kx or ky array)
#     radial_meshgrid = A grid that maps radius for each pixel


#     output:
#     azim = azimuthally averaged array (2D)
#     """

#     # initializes a matrix full of 0s
#     # for the azimuthally averaged product
#     azim = np.zeros([mid_space, mid_time], np.complex_)

#     # pixel size to azimuthally average over
#     w = 0.5

#     # Avoid Negative Frequencies
#     # Only grab indices corresponding to the positive frequencies
#     for j in tqdm(range(mid_time, end_time), desc="Azimuthal Averaging"):

#         # 3D FFT array is read in
#         arr_product = array[:, :, j]

#         # Ignore Negative Spatial Kx, Ky
#         for k in range(1, int(mid_space) + 1):

#             # result is an array showing the truth values element wise
#             desired_condition = np.logical_and(
#                 radial_meshgrid >= k - w, radial_meshgrid < k + w
#             )

#             # returns the indices that are non-zero from the flattened array
#             # where the condition corresponds to True
#             inds = find(desired_condition == True)

#             # returns a collapsed array (1-D)
#             flat_array = arr_product.flatten(order="C")

#             # Corresponding values within the flattened array that match the
#             # desired condition
#             # and average them
#             aa = np.mean(flat_array[inds])

#             azim[k - 1, j - int(mid_time)] = aa
#     return azim


# def azimuthal_averaging_coherence_td(
#     mid_time, end_time, array, mid_space, radial_meshgrid
# ):
#     """Computes an azimuthally averaged 2D data cube from a 3D FFT cube.
#     Assumed array is a square.

#     Input:

#     mid_time = Half the total time array. Number of positive frequencies
#     end_time = Total length of the time/frequency array
#     array = 3D array to azimuthally average
#     mid_space = Half the total length the spatial dimensions (kx or ky array)
#     radial_meshgrid = A grid that maps radius for each pixel


#     output:
#     azim = azimuthally averaged array (2D)
#     """

#     # initializes a matrix full of 0s
#     # for the azimuthally averaged product
#     azim = np.zeros([mid_space, mid_time + 1], np.complex_)

#     # pixel size to azimuthally average over
#     w = 0.5

#     # Avoid Negative Frequencies
#     # Only grab indices corresponding to the positive frequencies
#     for j in tqdm(range(mid_time, end_time), desc="Azimuthal Averaging"):

#         # 3D FFT array is read in
#         arr_product = array[:, :, j]

#         # Ignore Negative Spatial Kx, Ky
#         for k in range(1, int(mid_space) + 1):

#             # result is an array showing the truth values element wise
#             desired_condition = np.logical_and(
#                 radial_meshgrid >= k - w, radial_meshgrid < k + w
#             )

#             # returns the indices that are non-zero from the flattened array
#             # where the condition corresponds to True
#             inds = find(desired_condition == True)

#             # returns a collapsed array (1-D)
#             flat_array = arr_product.flatten(order="C")

#             # Corresponding values within the flattened array that match the
#             # desired condition
#             # and average them
#             aa = np.mean(flat_array[inds])

#             azim[k - 1, j - int(mid_time)] = aa
#     return azim
