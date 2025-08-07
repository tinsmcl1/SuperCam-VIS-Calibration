# SuperCam-VIS-Calibration

This repository provides code to select the best calibration target for SuperCam VIS spectral data and apply the calibration to radiance data, saving the result to file.

## Requirements
- Python 3.12 (recommended)
- Conda (for environment management)

## Setup Instructions

### 1. Clone the Repository
```
git clone https://github.com/tinsmcl1/SuperCam-VIS-Calibration.git
cd SuperCam-VIS-Calibration
```

### 2. Create a Conda Environment
It is recommended to use a conda environment to manage dependencies and avoid conflicts.

```
conda create -n supercam-cal python=3.12
conda activate scam-vis-cal
```

### 3. Install Required Packages
Install the required packages using pip (inside your conda environment):

```
pip install numpy, scipy, natsort
```

Or, to install the specific versions used in development:
```
pip install -r requirements.txt
```

## Usage
### Configuration
The are a number of constants that can be configured at the top of the calibration script. You can modify these are needed. Ensure they are set appropriately for your use case before running. These are the defaults:
```
OUT_DIR_NAME = "REF_BEST"            # Name of output directory for selected target reflectance files
RAD_DIR_MATCH = "RAD"                # Name of directory containing radiance files (used to recursively find radiance files)
RAD_FILE_MATCH = "*RAD.csv"          # Pattern to match radiance files (used to recursively find all radiance files in a given directory)
CAL_FILE_MATCH = "*.txt"             # Pattern to match calibration files (used to recursively find all calibration files in a given directory)
COSINE_CORRECTION = True             # Whether to apply cosine correction in calibration
FIX_BAD_WLS = True                   # Whether to fix known bad wavelengths in the dataset ( wl[4420:4423] = (712.204, 712.205, 712.206) )
VIS_START_WL = 537.28                # Start of visible spectral region (inclusive, nm)
VIO_END_WL = 465                     # End of VIO spectral region (exclusive, nm)
VIO_SAVGOL_WINDOW_LENGTH = 51        # Window length for savgol filter applied to VIO data (must be odd)
VIO_SAVGOL_POLYORDER = 1             # Polynomial order for savgol filter applied to VIO data
ABSORPTION_ARTIFACT_WL = (690, 698)  # Wavelength range of known absorption artifact to ignore (inclusive, nm)
EVENT_WLS = [619.8, 713.5]           # Center wavelengths of expected anomalies (nm)
EVENT_WINDOW_NM = 20                 # Width of window (nm) around each event to analyze
SMOOTHING_KERNEL_NM = 2              # Width (nm) of smoothing window applied for target analysis (not applied to final result)
BALANCE_PENALTY_WEIGHT = 0.5         # Tradeoff weight between balance and magnitude of anomaly scores
```

### Inputs

The script is designed to work with Supercam spectral data and requires candidate calibration files and radiance data in text or CSV format. It is intended to be run from the command line with those files/directories as input.

Input calibration target data should be provided as a directory containing text files, with one file for each candidate calibration target. Each file must have two space-delimited columns: the first column for wavelength and the second for radiance.

The input radiance data to be calibrated is in similar format, but has .csv extension and should live in a directory called 'RAD'. The expected directory structure for input radiance data is as follows:

    DatasetDirectory/
    ├── RAD/
    │   ├── data1_RAD.csv
    │   ├── data2_RAD.csv
    │   └── ...
    ├── data1.fits
    ├── data2.fits
    └── ...

> Note that the .fits files are only needed when COSINE_CORRECTION = True. The headers will be read to extract the required values for that correction. The fits files are expected to be named the same as the radiance files besides the .fits extension replacing '_RAD.csv'.

The code will work if you supply a single radiance file, a path to a RAD/ directory, or a path to a directory containing one or more RAD directories at any sublevel, meaning you can process multiple datasets at once. However, the calibration target files should be in a single directory.

### Running the Calibration Script
Run the calibration script from the command line:

```
python calibrate.py /path/to/calibration/files /path/to/radiance/file(s)
```

Or:
```
python calibrate.py /path/to/calibration/files /path/containing/one/or/more/datasets
```

Optional: specify an output directory with `--out`. If not specified, a subdirectory named the value of `OUT_DIR_NAME` will be created at the same level as the parent directory of the input radiance file(s).

#### Examples
```
python calibrate.py sample_data/Sol184_VIS/radiance_text_files_1_to_50/ sample_data/Sol207_Garde_abrasion/RAD/SCAM_0207_0685310942_124_CP3_scam01207_Garde_207_scam_______01P11_RAD.csv
python calibrate.py sample_data/Sol184_VIS/radiance_text_files_1_to_50/ sample_data/Sol207_Garde_abrasion/RAD/
python calibrate.py sample_data/Sol184_VIS/radiance_text_files_1_to_50/ sample_data/
```
Outputs will be saved to a `{OUT_DIR_NAME}/` subdirectory at the appropriate level for each dataset.

## Calibration Target Selection and Scoring Algorithm

The goal of this algorithm is to select the best calibration target from a set of candidates for SuperCam VIS spectral data. The process is as follows:

1. **Preprocessing:**
    - Known artifacts (such as absorption features and bad wavelengths) are removed from the spectra.
    - Only the visible (VIS) region is retained for analysis.

2. **Reflectance Calculation:**
    - Each candidate calibration spectrum is used to convert the radiance data to reflectance.

3. **Smoothing:**
    - The reflectance spectra are smoothed using a moving average kernel to reduce noise.

4. **Event Window Normalization:**
    - For each event wavelength of interest, a window is extracted around the event.
    - The continuum (baseline) is estimated by fitting a straight line between the window edges and is subtracted from the spectrum in the window, isolating local features.

5. **Variance Scoring:**
    - For each event window, the variance of the normalized spectrum is computed for each calibration candidate. Lower variance indicates a smoother, more regular spectrum in that region.

6. **Balanced Score Aggregation:**
    - For each calibration candidate, the variances from all event windows are combined into a single score using a balance-based metric:

     ```python
     combined_score = np.std(event_variances) + w * np.mean(event_variances)
     ```
    - Here, `w` is a tradeoff parameter that balances the importance of overall quality (mean variance) and fairness (standard deviation between events).
    - The calibration target with the lowest combined score is selected as the best.

This approach ensures that the selected calibration target minimizes anomalies at key wavelengths and provides balanced performance across all events of interest.

## Calibration Algorithm

After identifying the best calibration targets to use, each radiance spectrum is converted to reflectance. This process is as follows:

1. **Load Calibration and Radiance Data:**
    - The selected calibration target spectrum and the radiance spectrum(s) are loaded.

2. **Bad Wavelength Correction:**
    - Known bad wavelengths are corrected as specified in the configuration.

3. **Cosine Correction (Optional):**
    - If enabled, a cosine correction is applied to account for illumination geometry.

4. **Reflectance Calculation:**
    - Reflectance is computed by dividing the radiance spectrum by the calibration target spectrum at each wavelength:
    ```
    reflectance = radiance / calibration_target
    ```

5. **VIO Region is Smoothed:**
    - The VIO region of the spectrum is smoothed with a 51-channel Savitzky-Golay filter.

6. **Output:**
    - The calibrated reflectance spectra are saved to the specified output directory in two-column space-deliminated .csv files, where the first column is wavelength and second column is reflectance.

This process ensures that the resulting reflectance data are accurately calibrated and ready for scientific analysis.

## Contact
For questions, contact Calley.Tinsman@jhuapl.edu or Jeffrey.R.Johnson@jhuapl.edu
https://github.com/tinsmcl1/SuperCam-VIS-Calibration
