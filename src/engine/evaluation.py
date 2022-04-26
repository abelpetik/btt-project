"""
Automatically evaluate a group of ERG measurements.

Notes
-----
Standard deviation cannot be used everywhere, because some data are not normally distributed.
We always use median instead of mean to calculate reference values.
Reference values should be adjusted to age, but we don't have a large enough cohort.
"""

import os
import warnings
from pathlib import Path
from data_import import importer


def evaluate_in_folder(folder_path: str):
    folder_path = Path(folder_path)
    files_in_folder = os.listdir(folder_path)
    file_list = [fname for fname in files_in_folder if fname[-4:] == '.csv']
    for file in file_list:
        evaluate_single_patient(folder_path / file)


def evaluate_single_patient(filepath: str = None):
    """

    Parameters
    ----------
    filepath

    Returns
    -------

    """
    patient_data = importer(filepath)
    print(f'Number of recorded protocols: {len(patient_data)}')

    # Sorting the individual recordings based on their protocol
    dark_adapted_001_indices = []
    dark_adapted_3_indices = []
    dark_adapted_30_indices = []
    light_adapted_3_indices = []
    light_adapted_30_hz_indices = []
    uncategorized_indices = []

    for i, protocol in enumerate(patient_data):
        if protocol['Flash (cd·s/m² or cd/m²)'] == '0.01':
            dark_adapted_001_indices.append(i)
        elif protocol['Flash (cd·s/m² or cd/m²)'] == '3' and float(protocol['Stimulus Frequency']) <= 0.1:
            dark_adapted_3_indices.append(i)
        elif protocol['Flash (cd·s/m² or cd/m²)'] == '3' and 1.5 < float(protocol['Stimulus Frequency']) < 2.5:
            light_adapted_3_indices.append(i)
        elif protocol['Flash (cd·s/m² or cd/m²)'] == '3' and 28 < float(protocol['Stimulus Frequency']):
            light_adapted_30_hz_indices.append(i)
        elif protocol['Background (cd/m²)'] == '30':
            dark_adapted_30_indices.append(i)
        else:
            uncategorized_indices.append(i)

    print(f"Dark 0.01: {dark_adapted_001_indices}")
    print(f"Dark 3: {dark_adapted_3_indices}")
    print(f"Dark 30: {dark_adapted_30_indices}")
    print(f"Light 3: {light_adapted_3_indices}")
    print(f"Light 30 Hz: {light_adapted_30_hz_indices}")
    if uncategorized_indices:
        warnings.warn(f"There were {len(uncategorized_indices)} recordings that do not correspond to a protocol.")


if __name__ == '__main__':
    evaluate_in_folder(r'C:\Abel\Egyetem\MSc\3.felev\BTT\Data from first visit')
    # evaluate_single_patient(r'C:\Abel\Egyetem\MSc\3.felev\BTT\Data from first visit\CB09_981015_220221122100_Small.csv')
    # evaluate_single_patient(r'C:\Abel\Egyetem\MSc\3.felev\BTT\Data from first visit\CB09_981015_220221124321_Normal.csv')
    # evaluate_single_patient()
