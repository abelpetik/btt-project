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
import re
from pathlib import Path
import matplotlib.pyplot as plt
from data_import import importer, save, load
import numpy as np
from scipy import sparse
import scipy.stats
from scipy.sparse.linalg import spsolve


def evaluate_single_patient(uid, folder_path: str):
    """
    Evaluate ERG waveforms of a single patient compared to other patients' data in the same provided folder.

    Parameters
    ----------
    uid: Patient ID eg.: 'CB07'
    folder_path: full path to folder containing patient's data

    Returns
    -------
    evaluation: dict holding the scores of all recordings addressable in an electrode/protocol/eye hierarchy.

    """
    folder_path = Path(folder_path)
    try:
        all_waveform_distances_from_median = load(folder_path / 'evaluation_results.pkl')
    except FileNotFoundError:
        print("Couldn't find already existing evaluation! Calculating now.")
        distances_in_folder(folder_path)
        all_waveform_distances_from_median = load(folder_path / 'evaluation_results.pkl')

    electrodes = ['Small', 'Normal']
    protocols = ['dark_001', 'dark_3', 'light_3', 'light_30']
    eyes = ['LeftEye', 'RightEye']

    evaluation = {}

    for electrode in electrodes:
        evaluation[electrode] = {}
        for protocol in protocols:
            evaluation[electrode][protocol] = {}
            for eye in eyes:
                evaluation[electrode][protocol][eye] = get_z_score_for_patient(all_waveform_distances_from_median, uid, electrode, protocol, eye)

    return evaluation


def distances_in_folder(folder_path: str):
    """
    Calculate the euclidean distance between each recorded ERG waveform in a folder from the median
    of all other corresponding recordings. Results are saved into the folder to evaluation_results.pkl.

    Parameters
    ----------
    folder_path: full path to folder containing ERG data

    Returns
    -------
    None
    """
    all_raw_waveforms = extract_all_patient_waveform_data(folder_path)

    regexp = r'(.*)_(\d{6})_(\d{12})_(.*).csv'

    folder_path = Path(folder_path)
    files_in_folder = os.listdir(folder_path)

    all_waveform_distances_from_median = {}

    for fname in files_in_folder:
        if fname[-4:] != '.csv':
            continue
        parsed_filenames = re.findall(regexp, fname)
        uid = parsed_filenames[0][0]
        electrode = parsed_filenames[0][-1]
        print(uid, electrode)

        if uid not in all_waveform_distances_from_median.keys():
            all_waveform_distances_from_median[uid] = {}
        if electrode not in all_waveform_distances_from_median.keys():
            all_waveform_distances_from_median[uid][electrode] = {}
        all_waveform_distances_from_median[uid][electrode] = compare_patient_to_others(uid, electrode, all_raw_waveforms)

    save(folder_path / 'evaluation_results.pkl', all_waveform_distances_from_median)


def get_z_score_for_patient(all_distances, uid, electrode, protocol, eye):
    """
    Return the z-score for 1 measurement, calculated as the distance from the median response in standard deviations.

    Parameters
    ----------
    uid: patient ID
    electrode: electrode type {'Normal', 'Small'}
    protocol: ERG protocol {'dark_001', 'dark_3', 'light_3', 'light_30'}
    eye: {'RightEye', 'LeftEye'}

    Returns
    -------
    z-score for specified recording.
    """
    distances = get_all_distances_for_protocol(all_distances, protocol, electrode)
    patient_distance = all_distances[uid][electrode][protocol][eye]
    return patient_distance / distances.std()


def get_all_distances_for_protocol(all_distances_dict: dict, protocol: str, electrode: str):
    """
    Get all the Euclidean distance values corresponding to a chosen protocol and electrode type.

    Parameters
    ----------
    all_distances_dict: dictionary containing all recording distances from median responses
    protocol: ERG protocol {'dark_001', 'dark_3', 'light_3', 'light_30'}
    electrode: electrode type to calculate for {'Normal', 'Small'}

    Returns
    -------
    Array containing extracted distances.
    """
    all_distances = []
    for patient in all_distances_dict.keys():
        try:
            all_distances.append(all_distances_dict[patient][electrode][protocol]['RightEye'])
            all_distances.append(all_distances_dict[patient][electrode][protocol]['LeftEye'])
        except KeyError:
            print(f"Missing data for patient: {patient}, electrode: {electrode}, protocol: {protocol}.")

    return np.array(all_distances)


def compare_patient_to_others(uid, electrode, all_raw_waveforms, show_plots=False):
    """
    Calculate Euclidean distance from median response calculated from all other waveforms with a leave-one-out paradigm.

    Parameters
    ----------
    uid
    electrode
    all_raw_waveforms
    show_plots

    Returns
    -------
    Patient's results in a dictionary.
    """
    patient_waveforms = all_raw_waveforms[uid][electrode]

    protocols = list(patient_waveforms.keys())

    patient_results = {}

    for protocol in protocols:
        patient_results[protocol] = {}
        others_waveforms = []
        for patient in all_raw_waveforms.keys():
            if patient == uid:
                continue
            for eye in ['RightEye', 'LeftEye']:
                try:
                    others_waveforms.append(all_raw_waveforms[patient][electrode][protocol][eye][1])

                except KeyError:
                    print(f"Missing data for patient: {patient}, electrode: {electrode}, protocol: {protocol}.")

        others_waveforms = np.stack(others_waveforms, axis=0)
        others_median = np.median(others_waveforms, axis=0)
        if show_plots:
            plt.plot(others_median-others_median[0], color='k', label='Median')
            plt.plot(patient_waveforms[protocol]['RightEye'][1], label='Right eye')
            plt.plot(patient_waveforms[protocol]['LeftEye'][1], label='Left eye')
            plt.xlabel('ms')
            plt.ylabel('uV')
            plt.legend()
            plt.show()

        patient_results[protocol]['RightEye'] = np.linalg.norm(others_median-(patient_waveforms[protocol]['RightEye'][1]-patient_waveforms[protocol]['RightEye'][1][0]))
        patient_results[protocol]['LeftEye'] = np.linalg.norm(others_median-(patient_waveforms[protocol]['RightEye'][1]-patient_waveforms[protocol]['LeftEye'][1][0]))

    return patient_results


def extract_all_patient_waveform_data(folder_path: str, reimport=False, preprocessed=True):
    """
    Extract waveform data from all recordings in a folder.

    Parameters
    ----------
    folder_path: full path to folder containing ERG data
    reimport: Whether to reimport all waveforms if they are already present in an 'all_raw_waveforms.pkl' file.
    preprocessed: Extract pre-processed "Reported waveform" instead of raw.

    Returns
    -------
    Extracted waveofrm data in a dictionary.
    """
    folder_path = Path(folder_path)
    files_in_folder = os.listdir(folder_path)
    if 'all_raw_waveforms.pkl' in files_in_folder and not reimport:
        print("Found already extracted waveforms in all_raw_waveforms.pkl. Using that instead of reimport.")
        return load(folder_path / 'all_raw_waveforms.pkl')
    all_raw_waveforms = {}

    regexp = r'(.*)_(\d{6})_(\d{12})_(.*).csv'

    for fname in files_in_folder:
        if fname[-4:] != '.csv':
            continue
        parsed_filenames = re.findall(regexp, fname)
        uid = parsed_filenames[0][0]
        electrode = parsed_filenames[0][-1]

        if uid not in all_raw_waveforms.keys():
            all_raw_waveforms[uid] = {}

        if preprocessed:
            all_raw_waveforms[uid][electrode] = extract_single_patient_waveform_data(folder_path / fname, 'Reported Waveform')
        else:
            all_raw_waveforms[uid][electrode] = extract_single_patient_waveform_data(folder_path / fname, 'Raw Waveform')

    print('Done with extracting waveforms from all files.')
    save(folder_path / 'all_raw_waveforms.pkl', all_raw_waveforms)
    return all_raw_waveforms


def extract_single_patient_waveform_data(filepath: str = None, field=None):
    patient_data = importer(filepath)

    # ---Sorting the individual recordings based on their protocol---
    dark_001_indices = {}
    dark_3_indices = {}
    dark_30_indices = {}
    light_3_indices = {}
    light_30_indices = {}
    uncategorized_indices = []

    for i, protocol in enumerate(patient_data):
        if protocol['Flash (cd·s/m² or cd/m²)'] == '0.01':
            dark_001_indices[protocol['TestedEye']] = i
        elif protocol['Flash (cd·s/m² or cd/m²)'] == '3' and float(protocol['Stimulus Frequency']) <= 0.1:
            dark_3_indices[protocol['TestedEye']] = i
        elif protocol['Flash (cd·s/m² or cd/m²)'] == '3' and 1.5 < float(protocol['Stimulus Frequency']) < 2.5:
            light_3_indices[protocol['TestedEye']] = i
        elif protocol['Flash (cd·s/m² or cd/m²)'] == '3' and 28 < float(protocol['Stimulus Frequency']):
            light_30_indices[protocol['TestedEye']] = i
        elif protocol['Background (cd/m²)'] == '30':
            dark_30_indices[protocol['TestedEye']] = i
            # These have to be dropped, they don't have waveform data
        else:
            uncategorized_indices.append(i)

    # ---Printing found indices for all protocols---
    print(f"Dark 0.01: {dark_001_indices}")
    print(f"Dark 3: {dark_3_indices}")
    print(f"Dark 30: {dark_30_indices}")
    print(f"Light 3: {light_3_indices}")
    print(f"Light 30 Hz: {light_30_indices}")
    if uncategorized_indices:
        warnings.warn(f"There were {len(uncategorized_indices)} recordings that do not correspond to a protocol.")

    waveforms = {}
    protocols = ['dark_001', 'dark_3', 'light_3', 'light_30']

    for prot in protocols:
        waveforms[prot] = {}
        waveforms[prot]['RightEye'] = get_raw_data(patient_data[eval(f'{prot}_indices')['RightEye']], field=field)
        waveforms[prot]['LeftEye'] = get_raw_data(patient_data[eval(f'{prot}_indices')['LeftEye']], field=field)

    return waveforms


def get_params(target: dict, list_of_indices: list, patient_data: list):
    for keyword in target.keys():
        for i in list_of_indices:
            target[keyword].append(float(patient_data[i][keyword]))
    print(target)


def get_raw_data(trial_data, field):
    data = trial_data[field][1][~np.isnan(trial_data[field][1])]
    time = trial_data[field][0][~np.isnan(trial_data[field][0])]
    return time, data


def baseline_als_optimized(y, lam, p, niter=10):
    L = len(y)
    D = sparse.diags([1, -2, 1],[0, -1, -2], shape=(L, L-2))
    D = lam * D.dot(D.transpose())  # Precompute this term since it does not depend on `w`
    w = np.ones(L)
    W = sparse.spdiags(w, 0, L, L)
    for i in range(niter):
        W.setdiag(w)  # Do not create a new matrix, just update diagonal values
        Z = W + D
        z = spsolve(Z, w*y)
        w = p * (y > z) + (1-p) * (y < z)
    return z


import pywt


def calc_baseline(signal):
    """
    Calculate the baseline of signal.
    Args:
        signal (numpy 1d array): signal whose baseline should be calculated
    Returns:
        baseline (numpy 1d array with same size as signal): baseline of the signal
    """
    ssds = np.zeros((3))

    cur_lp = np.copy(signal)
    iterations = 0
    while True:
        # Decompose 1 level
        lp, hp = pywt.dwt(cur_lp, "db4")

        # Shift and calculate the energy of detail/high pass coefficient
        ssds = np.concatenate(([np.sum(hp ** 2)], ssds[:-1]))

        # Check if we are in the local minimum of energy function of high-pass signal
        if ssds[2] > ssds[1] and ssds[1] < ssds[0]:
            break

        cur_lp = lp[:]
        iterations += 1

    # Reconstruct the baseline from this level low pass signal up to the original length
    baseline = cur_lp[:]
    for _ in range(iterations):
        baseline = pywt.idwt(baseline, np.zeros((len(baseline))), "db4")

    return baseline[: len(signal)]


import scipy.signal


def highpass_filter(signal):
    b, a = scipy.signal.butter(3, 0.2, 'highpass', fs=2000)
    return scipy.signal.filtfilt(b, a, signal, axis=0)


if __name__ == '__main__':
    #distances_in_folder(r'C:\Data\ITK MSc\Info Bionics\Semester II\Brain Therapy Technologies\Data 1\Data from first visit')
    # extract_all_patient_waveform_data(r'C:\Abel\Egyetem\MSc\3.felev\BTT\Data from first visit')
    # extract_single_patient_waveform_data(r'C:\Abel\Egyetem\MSc\3.felev\BTT\Data from first visit\CB07_970322_220216113048_Normal.csv')
    # print(load(r'C:\Abel\Egyetem\MSc\3.felev\BTT\Data from first visit\CB07_970322_220216113048_Normal.pkl'))
    # evaluate_single_patient()
    print(evaluate_single_patient('CB07', r'C:\Data\ITK MSc\Info Bionics\Semester II\Brain Therapy Technologies\Data 1\Data from first visit'))
