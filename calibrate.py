"""
This script selects the best calibration target from a set of candidates and applies the calibration
to radiance data, saving the resulting spectrum to a text file. If an output directory is not
specified, a subdirectory named the value of OUT_DIR_NAME will be created at the same level as the
parent directory of the input radiance file(s).

A balance-based variance scoring system is used to evaluate the calibration targets based on their
spectral characteristics.

The script is designed to work with Supercam spectral data and requires candidate calibration files
and radiance data in text or CSV format. It is intended to be run from the command line with those
files/directories as input. The expected directory structure for input data is as follows:

    DatasetDirectory/
    ├── RAD/
    │   ├── data1_RAD.csv
    │   ├── data2_RAD.csv
    │   └── ...
    ├── data1.fits
    ├── data2.fits
    └── ...

Note that the .fits files are only needed when COSINE_CORRECTION = True. The headers will be
read to extract the required values for that correction. The fits files are expected to be
named the same as the radiance files besides the .fits extension replacing '_RAD.csv'.

The code will work if you supply a single radiance file, a path to a RAD/ directory, or a path
to a directory containing one or more RAD directories at any sublevel, meaning you can process
multiple datasets at once. However, the calibration target files should be in a single directory
outside of these datasets.

Example command-line usage:
    
    python calibrate.py /path/to/calibration/files /path/to/radiance/file(s)
    python calibrate.py /path/to/calibration/files /path/containing/one/or/more/datasets

Contact: Calley.Tinsman@jhuapl.edu, Jeffrey.R.Johnson@jhuapl.edu
Developed in Python 3.12.4 with the following dependencies:
    numpy=2.2.1
    scipy=1.15.1
    natsort=8.4.0
"""

import argparse
from pathlib import Path

import natsort
import numpy as np
from scipy.signal import savgol_filter


# Constants / Parameters
# These can be adjusted as needed.
# --------------------------------
OUT_DIR_NAME = "REF"                 # Name of output directory for selected target reflectance files
RAD_DIR_MATCH = "RAD"                # Name of directory containing radiance files (used to recursively find radiance files)
RAD_FILE_MATCH = "*RAD.csv"          # Pattern to match radiance files (used to recursively find all radiance files in a given directory)
CAL_FILE_MATCH = "*.txt"             # Pattern to match calibration files (used to recursively find all calibration files in a given directory)
FIX_BAD_WLS = True                   # Whether to fix known bad wavelengths in the dataset ( wl[4420:4423] = (712.204, 712.205, 712.206) )
COSINE_CORRECTION = False            # Whether to apply cosine correction in calibration
VIS_START_WL = 537.28                # Start of visible spectral region (inclusive, nm)
VIO_END_WL = 465                     # End of VIO spectral region (exclusive, nm)
VIO_SAVGOL_WINDOW_LENGTH = 51        # Window length for savgol filter applied to VIO data (must be odd)
VIO_SAVGOL_POLYORDER = 1             # Polynomial order for savgol filter applied to VIO data
ABSORPTION_ARTIFACT_WL = (690, 698)  # Wavelength range of known absorption artifact to ignore (inclusive, nm)
EVENT_WLS = [619.8, 713.5]           # Center wavelengths of expected anomalies (nm)
EVENT_WINDOW_NM = 20                 # Width of window (nm) around each event to analyze
SMOOTHING_KERNEL_NM = 2              # Width (nm) of smoothing window applied for target analysis (not applied to final result)
BALANCE_PENALTY_WEIGHT = 0.5         # Tradeoff weight between balance and magnitude of anomaly scores
# ---------------------------------


def get_best_target_index(wl_data, cal_data, rad_data) -> int:
    """
    Identify the best target spectrum based on balanced anomaly detection.

    This function performs a full spectral analysis pipeline to identify
    the most promising target spectrum that minimizes the combined anomaly
    features across predefined event wavelengths.

    The pipeline includes:
    1. Preprocessing of both calibration and radiance data.
    2. Conversion to reflectance by dividing radiance by calibration data.
    3. Smoothing of reflectance spectra.
    4. Continuum removal (normalization) around each event wavelength.
    5. Scoring of each spectrum using a balance-based variance metric.
    6. Selection of the target index with the lowest score.

    Parameters:
        wl_data (np.ndarray): 1D array of wavelength values (nm).
        cal_data (np.ndarray): 2D array of calibration target candidates. 
                               Shape should be (n_spectra, n_channels).
        rad_data (np.ndarray): 1D array of radiance data.

    Returns:
        int: Index of the calibration spectrum with the lowest (best) score.
    """
    # Preprocess wavelength and spectral data
    wl, cal = preprocess(wl_data, cal_data)
    _, rad = preprocess(wl_data, rad_data)

    # Convert radiance to reflectance
    refl = rad / cal

    # Smooth reflectance spectra using a moving average kernel
    refl_smoothed = smooth(wl, refl, kernel=SMOOTHING_KERNEL_NM)

    # Normalize (remove continuum) around each event wavelength
    normalized_event_spect = {
        event_wl: normalize_event_spectrum(wl, refl_smoothed, event_wl, EVENT_WINDOW_NM)
        for event_wl in EVENT_WLS
    }

    # Compute balance-based variance scores for each spectrum
    scores = compute_scores(normalized_event_spect, penalty_weight=BALANCE_PENALTY_WEIGHT)

    # Select the calibration spectrum index with the lowest score
    selected_target_index = np.argmin(scores)

    return selected_target_index


def fix_bad_wavelengths(wl) -> np.ndarray:
    """Fixes known incorrect wavelength values at indices 4420-4422."""
    wl[4420:4423] = (712.204, 712.205, 712.206)
    return wl


def preprocess(wl, spect) -> tuple[np.ndarray, np.ndarray]:
    """
    Preprocess spectral data by correcting known artifacts and restricting to the VIS region.

    This function performs the following operations:
    1. Removes a known absorption artifact between 690-698 nm.
    2. Removes all wavelengths equal to or greater than VIS_START_WL,
       keeping only the VIS region of the spectrum.

    Parameters:
        wl (np.ndarray): 1D array of wavelength values (in nm).
        spect (np.ndarray): 1D or 2D array of spectral data. Shape should be (n_channels,) or 
                            (n_spectra, n_channels).

    Returns:
        tuple:
            - np.ndarray: Wavelength array with artifacts and non-VIS data removed.
            - np.ndarray: Corresponding processed spectral data.
    """
    # Handle 1D or 2D spect array automatically (squeezed later to remove singleton dimensions)
    spect = np.atleast_2d(spect)

    # Drop known absorption artifact for analysis
    artifact_mask = (wl >= ABSORPTION_ARTIFACT_WL[0]) & (wl <= ABSORPTION_ARTIFACT_WL[1])  # mask (nm)
    wl = wl[~artifact_mask]
    spect = spect[:, ~artifact_mask]

    # Drop non-VIS data for analysis
    keep_vis = wl >= VIS_START_WL  # mask (nm)
    wl = wl[keep_vis]
    spect = spect[:, keep_vis]

    return wl, spect.squeeze()


def smooth(wl, spect, kernel: float) -> np.ndarray:
    """
    Apply a moving average smoothing filter to spectral data using a wavelength-based window.

    For each wavelength value, this function computes the mean spectral value over a window
    centered at that wavelength with total width equal to `kernel` (in nm). If the window 
    range does not include any neighbors, the original value is retained.

    Parameters:
        wl (np.ndarray): 1D array of wavelength values (in nm).
        spect (np.ndarray): 1D or 2D array of spectral data. Shape should be (n_channels,) or 
                            (n_spectra, n_channels).
        kernel (float): Width of the smoothing window in nm. Smoothing is performed over the 
                        range [w - kernel/2, w + kernel/2] for each wavelength `w`.

    Returns:
        np.ndarray: Smoothed spectral data with the same shape as input.
    """
    # Handle 1D or 2D spect array automatically (squeezed later to remove singleton dimensions)
    spect = np.atleast_2d(spect)

    # Compute half of the smoothing window width
    half_width = kernel / 2

    # Apply smoothing (moving average)
    smoothed = np.zeros_like(spect)
    for i, w in enumerate(wl):
        # Determine the index range of wavelengths within the smoothing window
        left = np.searchsorted(wl, w - half_width, side="left")
        right = np.searchsorted(wl, w + half_width, side="right")

        # Compute the mean over the window for each spectrum (if window has width)
        smoothed[:, i] = (np.mean(spect[:, left:right], axis=1) if right > left else spect[:, i])

    return smoothed.squeeze()


def normalize_event_spectrum(wl, spect, event_wl: float, event_window: float) -> np.ndarray:
    """
    Extract and normalize a spectral segment around a specified event wavelength
    by removing a linear continuum baseline.

    This function selects a window of spectral data centered on a target event
    wavelength, fits a linear continuum using the spectral values just outside
    the window, and subtracts the continuum from the original data to isolate
    local spectral features.

    Parameters:
        wl (np.ndarray): 1D array of wavelength values (nm).
        spect (np.ndarray): 2D array of spectral reflectance data with shape 
                            (n_spectra, n_channels).
        event_wl (float): Center wavelength of the event (nm).
        event_window (float): Total width (nm) of the analysis window centered on event_wl.

    Returns:
        np.ndarray: 2D array of continuum-removed (normalized) spectral values 
                    within the event window. Shape is (n_spectra, n_window_channels).
    """
    # Compute half-width of the window around the event center
    half_width = event_window / 2

    # Find indices corresponding to the bounds of the event window
    left_idx = np.searchsorted(wl, event_wl - half_width, side="left")
    right_idx = np.searchsorted(wl, event_wl + half_width, side="right")

    # Identify indices just outside the window to define continuum anchor points
    left_cont_idx = max(0, left_idx - 1)
    right_cont_idx = min(len(wl) - 1, right_idx + 1)

    # Wavelengths and spectral values at the continuum endpoints
    wl_left, wl_right = wl[left_cont_idx], wl[right_cont_idx]
    refl_left, refl_right = spect[:, left_cont_idx], spect[:, right_cont_idx]

    # Compute slope of the linear continuum between the two anchor points
    slope = (refl_right - refl_left) / (wl_right - wl_left)

    # Extract wavelength values within the event window
    segment_wl = wl[left_idx:right_idx]

    # Construct the continuum line across the window for each spectrum
    continuum = refl_left[:, None] + slope[:, None] * (segment_wl - wl_left)

    # Extract the original spectral data in the event window
    original = spect[:, left_idx:right_idx]

    # Subtract the continuum to normalize the event region
    return original - continuum


def compute_scores(event_spect: dict, penalty_weight: float = 0.5) -> np.ndarray:
    """
    Compute a balance-based anomaly score for each spectrum across multiple event regions.

    For each input spectrum, this function calculates the variance of the signal within 
    each event window, then aggregates these variances across all events using a balance 
    score defined as:

        score = std + penalty_weight x mean

    This score rewards spectra with consistent variance across events while 
    penalizing both high average variance and high variability.

    Parameters:
        event_spect (dict): A dictionary mapping event wavelengths to 2D arrays
                            of continuum-removed spectral data. Each array should 
                            have shape (n_spectra, n_channels).
        penalty_weight (float): A scalar weight applied to the mean variance 
                                when computing the balance score. Default is 0.5.

    Returns:
        np.ndarray: 1D array of variance balance scores, one per input spectrum.
    """

    # Compute variance across wavelengths for each event window
    per_event_vars = {key: np.var(segment, axis=1) for key, segment in event_spect.items()}
    var_stack = np.stack(list(per_event_vars.values()))

    # Compute balance score: std + weight * mean (per spectrum)
    avg = np.mean(var_stack, axis=0)
    std = np.std(var_stack, axis=0)
    scores = std + penalty_weight * avg

    return scores


def find_files(path: Path, glob_pattern: str, target_dir_name: str = None) -> list[Path]:
    """
    Find all files matching a glob pattern in a directory or its subdirectories.
    - If target_dir_name is given, only search in subdirectories named target_dir_name.
    - If the provided path is itself named target_dir_name, search only within that directory.
    - If target_dir_name is None, search all subdirectories.

    Args:
        path (Path): Directory to search.
        glob_pattern (str): Pattern to match files (e.g., "*.txt").
        target_dir_name (str, optional): Name of subdirectory to search within.

    Returns:
        list[Path]: Sorted list of matching Path objects.

    Raises:
        FileNotFoundError: If no matching files are found.
    """
    files = []

    if target_dir_name:
        # If path is the target directory, search only within it
        if path.is_dir() and path.name == target_dir_name:
            files = list(path.glob(glob_pattern))
        else:
            # Search all subdirectories named target_dir_name
            for subdir in path.rglob(target_dir_name):
                if subdir.is_dir():
                    files.extend(subdir.glob(glob_pattern))
    else:
        # If no target_dir_name, search all directories
        files = list(path.rglob(glob_pattern))

    if not files:
        msg = f"No files matching '{glob_pattern}' found under '{path}'"
        if target_dir_name:
            msg += f" in '{target_dir_name}' directories"
        raise FileNotFoundError(msg)

    return natsort.natsorted(files)


def load_spect(files: list[Path]) -> np.ndarray:
    """ Load spectral data from text files.

    Args:
        files (list[Path]): List of Path objects pointing to spectral data files.

    Returns:
        np.ndarray: 2D array of spectral data with shape (n_channels, n_spectra).
    """
    spectra = np.stack([np.loadtxt(file) for file in files])
    return spectra.transpose((2, 0, 1))


def apply_calibration(wl, cal, rad) -> np.ndarray:
    """ Apply calibration to radiance data to convert it to reflectance, and smooth 
    the VIO region using a 51-channel savgol filter.

    Args:
        wl (np.ndarray): 1D array of wavelengths corresponding to the spectral data.
        cal (np.ndarray): 1D array of calibration data.
        rad (np.ndarray): 1D array of radiance data.

    Returns:
        np.ndarray: 1D array of reflectance data after applying calibration and 
                    smoothing the VIO region.
    """
    refl = rad / cal
    refl = smooth_vio(wl, refl, VIO_SAVGOL_WINDOW_LENGTH, VIO_SAVGOL_POLYORDER)
    return refl


def smooth_vio(wl, spect, savgol_window_length: int, savgol_polyorder: int) -> np.ndarray:
    """ Smooth the VIO region of the spectral data using a Savitzky-Golay filter.

    Args:
        wl (np.ndarray): 1D array of wavelengths corresponding to the spectral data.
        spect (np.ndarray): 1D or 2D array of spectral data. Shape should be (n_channels,) or 
                           (n_spectra, n_channels).
        savgol_window_length (int): Window length for the Savitzky-Golay filter. Must be odd.
        savgol_polyorder (int): Polynomial order for the Savitzky-Golay filter.

    Returns:
        np.ndarray: 1D or 2D array of smoothed spectral data with the same shape as input.
    """
    # Handle 1D or 2D spect array automatically (squeezed later to remove singleton dimensions)
    spect = np.atleast_2d(spect)

    # Apply savgol filter to VIO data
    if np.min(wl) < VIO_END_WL:
        vio_mask = wl < VIO_END_WL
        spect[:, vio_mask] = savgol_filter(
            spect[:, vio_mask],
            window_length=savgol_window_length,
            polyorder=savgol_polyorder,
        )
    else:
        print(f"No VIO data to smooth, skipping. VIO_END_WL set to: {VIO_END_WL}")

    return spect.squeeze()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Select best target index from calibration candidates and apply correction.")
    parser.add_argument("cal", type=Path, help="Path to directory containing calibration data")
    parser.add_argument("rad", type=Path, help="Path to radiance data (directory or singular file)")
    parser.add_argument("--out", default=None, help=f"Output directory name to be created in the parent directory of the rad file. Default is '{OUT_DIR_NAME}'")
    args = parser.parse_args()

    print(f"[INFO] Calibration directory: {args.cal}")
    print(f"[INFO] Radiance input: {args.rad}")
    if args.out:
        print(f"[INFO] Output directory name specified: {args.out}")
        out_dir_name = args.out
    else:
        print(f"[INFO] Output directory name not specified; will use default '{OUT_DIR_NAME}'")
        out_dir_name = OUT_DIR_NAME
    
    # If args.rad is a directory, find all radiance data files to process
    if args.rad.is_dir():
        print(f"[INFO] Searching for radiance files in directory: {args.rad}")
        rad_files = find_files(args.rad, RAD_FILE_MATCH, RAD_DIR_MATCH)
        print(f"[INFO] Found {len(rad_files)} radiance file(s)")
    else:
        print(f"[INFO] Using single radiance file: {args.rad}")
        rad_files = [args.rad]

    # If COSINE_CORRECTION is enabled, ensure that the corresponding FITS files exist
    if COSINE_CORRECTION:
        raise Exception("COSINE_CORRECTION is not yet implemented in this version of the script. Please set COSINE_CORRECTION to False.")
        print(f"[INFO] COSINE_CORRECTION enabled; searching for corresponding fits files...")
        fits_files = []
        for file in rad_files:
            # Ensure the corresponding FITS file exists
            fits_file = file.stem[:-4] + '.fits'
            fits_file = file.parents[1] / fits_file
            if not fits_file.exists():
                raise FileNotFoundError(f"Expected FITS file '{fits_file}' does not exist for cosine correction.")
            fits_files.append(fits_file)
        print(f"[INFO] Found {len(fits_files)} fits file(s)")

    # Find all calibration files in the specified directory
    print(f"[INFO] Searching for calibration files in directory: {args.cal}")
    cal_files = find_files(args.cal, CAL_FILE_MATCH)
    print(f"[INFO] Found {len(cal_files)} calibration file(s)")

    # Load calibration and radiance data
    print("[INFO] Loading calibration spectra...")
    wl_data, cal_data = load_spect(cal_files)
    print("[INFO] Loading radiance spectra...")
    _, rad_data = load_spect(rad_files)

    # Fix known bad wavelengths
    if FIX_BAD_WLS:
        print("[INFO] Fixing known bad wavelengths...")
        wl = fix_bad_wavelengths(wl_data[0])
    else:
        wl = wl_data[0]

    # Create output directories for results
    out_dirs = [filename.parents[1] / out_dir_name for filename in rad_files]
    for out_dir in set(out_dirs):
        try:
            print(f"[INFO] Creating output directory: {out_dir}")
            out_dir.mkdir(parents=True, exist_ok=False)
        except FileExistsError:
            print(f"[INFO] Output directory '{out_dir}' already exists. Skipping creation.")

    # Process each radiance file to select the best calibration target and apply calibration
    for idx, (filename, rad, out_dir) in enumerate(zip(rad_files, rad_data, out_dirs)):
        print(f"[INFO] Processing radiance file {idx+1}/{len(rad_files)}: {filename}")
        best_target = get_best_target_index(wl, cal_data, rad)
        print(f"[INFO] Best calibration target index for {filename.name}: {best_target}")
        refl = apply_calibration(wl, cal_data[best_target], rad)

        # Save wl, refl to file
        out_filename = filename.stem[:-3] + f"R{best_target:02d}.csv"
        out_path = out_dir / out_filename
        print(f"[INFO] Saving reflectance data to: {out_path}")
        np.savetxt(out_path, np.column_stack((wl, refl)), fmt="%.29f", delimiter="  ")
