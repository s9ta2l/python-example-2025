#!/usr/bin/env python

# Edit this script to add your team's code. Some functions are *required*, but you can edit most parts of the required functions,
# change or remove non-required functions, and add your own functions.

################################################################################
#
# Optional libraries, functions, and variables. You can change or remove them.
#
################################################################################

import joblib
import numpy as np
import os
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import sys
import pandas as pd
import neurokit2 as nk
from tqdm import tqdm
from scipy import signal
import wfdb

from scipy.stats import skew, kurtosis
from sklearn.preprocessing import StandardScaler
import pywt


from helper_code import *

################################################################################
#
# Required functions. Edit these functions to add your code, but do not change the arguments for the functions.
#
################################################################################

# Train your models. This function is *required*. You should edit this function to add your code, but do *not* change the arguments
# of this function. If you do not train one of the models, then you can return None for the model.

# Train your model.
def train_model(data_folder, model_folder, verbose):
    # Find the data files.
    if verbose:
        print('Finding the Challenge data...')

    records = find_records(data_folder)
    num_records = len(records)

    if num_records == 0:
        raise FileNotFoundError('No data were provided.')

    # Extract the features and labels from the data.
    if verbose:
        print('Extracting features and labels from the data...')

    features = np.zeros((num_records, 112), dtype=np.float64)
    labels = np.zeros(num_records, dtype=bool)
    skipped_records_list = []
    skipped_records_idx_list = []

    # Iterate over the records.
    for i in range(num_records):
        if verbose:
            width = len(str(num_records))
            print(f'- {i+1:>{width}}/{num_records}: {records[i]}...')

        record = os.path.join(data_folder, records[i])
        try:
            features[i] = extract_features(record)
            labels[i] = load_label(record)
        except Exception as e:
            print("Error:", e)
            print(f'Skipping record {i+1}: {records[i]} - error with extracting features.')
            skipped_records_list.append(records[i])
            skipped_records_idx_list.append(i)
            continue

    # Remove the rows of the skipped records
    features = np.delete(features, skipped_records_idx_list, axis=0)
    labels = np.delete(labels, skipped_records_idx_list, axis=0)

    # Train the models.
    if verbose:
        print('Training the model on the data...')

    # This very simple model trains a random forest model with very simple features.

    # Define the parameters for the random forest classifier and regressor.
    n_estimators = 50  # Number of trees in the forest.
    max_leaf_nodes = 5  # Maximum number of leaf nodes in each tree.
    random_state = 56  # Random state; set for reproducibility.

    # Fit the model.
    model = RandomForestClassifier(
        n_estimators=n_estimators, max_leaf_nodes=max_leaf_nodes, random_state=random_state).fit(features, labels)

    # Create a folder for the model if it does not already exist.
    os.makedirs(model_folder, exist_ok=True)

    # Save the model.
    save_model(model_folder, model)

    if verbose:
        print('Done.')
        print()

# Load your trained models. This function is *required*. You should edit this function to add your code, but do *not* change the
# arguments of this function. If you do not train one of the models, then you can return None for the model.
def load_model(model_folder, verbose):
    model_filename = os.path.join(model_folder, 'model.sav')
    model = joblib.load(model_filename)
    return model

# Run your trained model. This function is *required*. You should edit this function to add your code, but do *not* change the
# arguments of this function.
def run_model(record, model, verbose):
    # Load the model.
    model = model['model']

    # Extract the features.
    features = extract_features(record)
    features = features.reshape(1, -1)

    # Get the model outputs.
    binary_output = model.predict(features)[0]
    probability_output = model.predict_proba(features)[0][1]

    return binary_output, probability_output

################################################################################
#
# Optional functions. You can change or remove these functions and/or add new functions.
#
################################################################################

# Extract your features.
def extract_features(record):
    header = load_header(record)
    age = get_age(header)
    sex = get_sex(header)


    ###################################
    Signals, fields = load_signals(record)    
    
    try:
    
        sample_rate = 400

        lead_index = 6
        signal_data = Signals[:,lead_index]  

        # Remove baseline wander (high-pass filter at 0.5 Hz)
        nyquist = 0.5 * sample_rate
        high_pass_cutoff = 0.5 / nyquist
        b_hp, a_hp = signal.butter(2, high_pass_cutoff, btype='high')
        ecg_baseline_removed = signal.filtfilt(b_hp, a_hp, signal_data)

        # Remove high-frequency noise using a low-pass filter 
        signal_filtered = remove_high_frequency_noise(ecg_baseline_removed, sample_rate, cutoff_freq=40)


        # Detect R-peaks using NeuroKit2
        signals, info = nk.ecg_peaks(signal_filtered, sampling_rate=sample_rate, method='promac')#, method='martinez2004' , 'manikandan2012'
        # Extract R-peak indices
        rpeaks = info['ECG_R_Peaks']


        # Perform ECG delineation to get QRS onset and offset
        delineate_signals, delineate_info = nk.ecg_delineate(signal_filtered, rpeaks, sampling_rate=sample_rate, method="dwt")

        # Extract QRS onset and offset points
        qrs_onsets = delineate_info["ECG_Q_Peaks"]  # Q wave peaks (QRS onset)
        qrs_offsets = delineate_info["ECG_S_Peaks"]  # S wave peaks (QRS offset)

        # Filter out None values
        qrs_onsets = [onset for onset in qrs_onsets if onset is not None]
        qrs_offsets = [offset for offset in qrs_offsets if offset is not None]

        # Make sure we have matching onsets and offsets
        min_len = min(len(qrs_onsets), len(qrs_offsets))
        qrs_onsets = qrs_onsets[:min_len]
        qrs_offsets = qrs_offsets[:min_len]

        # Calculate QRS durations in seconds
        qrs_durations = []
        for onset, offset in zip(qrs_onsets, qrs_offsets):
            if onset < offset:  # Ensure valid duration (onset should be before offset)
                duration_samples = offset - onset
                duration_seconds = duration_samples / sample_rate
                qrs_durations.append(duration_seconds)

        # Convert QRS durations to binary (1 if > 110ms, 0 otherwise)
        qrs_binary = [1 if duration * 1000 > 110 else 0 for duration in qrs_durations]
        qrs_binary_percentage = sum(qrs_binary) / len(qrs_binary) * 100
        QRS_prolonged_binary = 1 if qrs_binary_percentage > 50 else 0


        # Custom T-wave detection
        # Define window for T-wave search (in samples, based on sampling rate)
        t_window_start = int(0.1 * sample_rate)  # 100 ms after R-peak
        t_window_end = int(0.4 * sample_rate)    # 400 ms after R-peak

        t_peak_indices = []
        t_peak_amplitudes = []
        t_inversion_flags = []

        print("Detecting T-waves...")
        for r_peak in rpeaks:
            # Define the window for T-wave search
            start_idx = r_peak + t_window_start
            end_idx = r_peak + t_window_end

            # Make sure indices are within signal bounds
            if start_idx >= len(signal_filtered) or end_idx >= len(signal_filtered) or start_idx >= end_idx:
                continue

            # Extract the segment for T-wave analysis
            t_wave_segment = signal_filtered[start_idx:end_idx]
            
            # Check if the segment is empty
            if len(t_wave_segment) == 0:
                continue

            # Find the index of the minimum value (for inverted T-waves)
            min_idx = start_idx + np.argmin(t_wave_segment)
            min_val = np.min(t_wave_segment)

            # Find the index of the maximum value (for normal T-waves)
            max_idx = start_idx + np.argmax(t_wave_segment)
            max_val = np.max(t_wave_segment)

            # Determine if T-wave is inverted (more negative than positive)
            if abs(min_val) > abs(max_val):
                # T-wave is likely inverted
                t_peak_indices.append(min_idx)
                t_peak_amplitudes.append(min_val)
                t_inversion_flags.append(True)  # True means inverted
            else:
                # T-wave is likely normal
                t_peak_indices.append(max_idx)
                t_peak_amplitudes.append(max_val)
                t_inversion_flags.append(False)  # False means not inverted

        # Check if T-waves were found
        if len(t_peak_indices) == 0:
            print("\T-wave Analysis: No T-waves detected. Setting T-wave statistics to NaN.")
            t_inversion_percentage = np.nan
            t_wave_binary = np.nan
        else:
            # Calculate percentage of inverted T-waves
            t_inversion_percentage = sum(t_inversion_flags) / len(t_inversion_flags) * 100
            t_wave_binary = 1 if t_inversion_percentage > 50 else 0
        


        # Custom P-wave detection
        # Define window for P-wave search preceding R-peak (e.g., 300ms to 100ms before R-peak)
        p_window_before = int(0.3 * sample_rate)  # 300 ms before R-peak
        p_window_after = int(0.1 * sample_rate)   # 100 ms before R-peak
        p_peak_indices = []
        p_peak_amplitudes = []
        p_inversion_flags = []  # True if inverted
        p_absence_flags = []    # True if P-wave is absent (amplitude below threshold)

        # Threshold for P-wave amplitude; if below, we consider it 'absent'
        p_absence_threshold = 0.05  # Adjust based on noise level

        print("Detecting P-waves...")
        for r_peak in rpeaks:
            start_idx = max(r_peak - p_window_before, 0)
            end_idx = max(r_peak - p_window_after, 0)

            if end_idx <= start_idx:
                p_peak_indices.append(None)
                p_peak_amplitudes.append(None)
                p_inversion_flags.append(False)
                p_absence_flags.append(True)
                continue

            p_wave_segment = signal_filtered[start_idx:end_idx]
            
            # Check if the segment is empty
            if len(p_wave_segment) == 0:
                p_peak_indices.append(None)
                p_peak_amplitudes.append(None)
                p_inversion_flags.append(False)
                p_absence_flags.append(True)
                continue

            # For a normal P-wave, we expect an upright deflection; we take the maximum value
            max_idx = start_idx + np.argmax(p_wave_segment)
            max_val = np.max(p_wave_segment)

            # For an inverted P-wave, we expect a downward deflection; we take the minimum value
            min_idx = start_idx + np.argmin(p_wave_segment)
            min_val = np.min(p_wave_segment)

            # Determine if P-wave is inverted or absent
            if abs(min_val) > abs(max_val):
                # P-wave is likely inverted
                p_peak_indices.append(min_idx)
                p_peak_amplitudes.append(min_val)
                p_inversion_flags.append(True)
                p_absence_flags.append(False)
            else:
                # P-wave might be normal or absent
                p_peak_indices.append(max_idx)
                p_peak_amplitudes.append(max_val)
                p_inversion_flags.append(False)

                # If the amplitude is below threshold, consider it absent
                if max_val < p_absence_threshold:
                    p_absence_flags.append(True)
                else:
                    p_absence_flags.append(False)

            ## Check if P-waves were found (non-absent)
        valid_p_waves = [i for i, absent in enumerate(p_absence_flags) if not absent]

        if len(valid_p_waves) == 0:
            print("\P-wave Analysis: No valid P-waves detected. Setting P-wave statistics to NaN.")
            num_absent = len(p_absence_flags)
            num_inverted = 0
            p_inversion_percentage = np.nan
            p_wave_binary = np.nan
            p_wave_absent_percentage = 100.0  # All are absent
            p_wave_absent_binary = 1  # All are absent
        else:
            # Print P-wave detection summary
            num_absent = sum(p_absence_flags)
            num_inverted = sum(p_inversion_flags)
            p_inversion_percentage = num_inverted / len(valid_p_waves) * 100 if valid_p_waves else np.nan
            p_wave_absent_percentage = num_absent / len(p_peak_indices) * 100 if p_peak_indices else np.nan

            # Convert P-wave inversion to binary (1 if > 50% inverted, 0 otherwise)
            p_wave_binary = 1 if p_inversion_percentage > 50 else 0

            # Convert P-wave absence to binary (1 if > 50% absent, 0 otherwise)
            p_wave_absent_binary = 1 if p_wave_absent_percentage > 50 else 0 
        

        # Create a summary dataframe with binary features
        summary_data = {
            'QRS_prolonged_binary': 1 if not np.isnan(qrs_binary_percentage) and qrs_binary_percentage > 50 else 0 if not np.isnan(qrs_binary_percentage) else np.nan,
            'T_wave_inverted_binary': t_wave_binary,
            'P_wave_inverted_binary': p_wave_binary,
            'P_wave_absent_percentage': p_wave_absent_percentage,
            'P_wave_absent_binary': p_wave_absent_binary if not np.isnan(p_wave_absent_percentage) else np.nan
        }


        # Compute RR intervals in seconds
        rr_intervals = np.diff(rpeaks) / sample_rate


        hrv_features = {}
        if len(rr_intervals) > 0:
            hrv_features['Mean_RR'] = np.mean(rr_intervals)
            hrv_features['STD_RR'] = np.std(rr_intervals)
            #hrv_features['RMSSD'] = np.sqrt(np.mean(np.square(np.diff(rr_intervals)))) if len(rr_intervals) > 1 else np.nan
            #hrv_features['STD_Successive_RR'] = np.std(np.diff(rr_intervals)) if len(rr_intervals) > 1 else np.nan
            hrv_features['Median_RR'] = np.median(rr_intervals)
            #hrv_features['MAD_RR'] = np.median(np.abs(rr_intervals - np.median(rr_intervals)))
            #hrv_features['20th_percentile_RR'] = np.percentile(rr_intervals, 20)
            #hrv_features['80th_percentile_RR'] = np.percentile(rr_intervals, 80)
            hrv_features['Min_RR'] = np.min(rr_intervals)
            hrv_features['Max_RR'] = np.max(rr_intervals)

        one_hot_encoding_sex = np.zeros(3, dtype=bool)
        if sex == 'Female':
            one_hot_encoding_sex[0] = 1
        elif sex == 'Male':
            one_hot_encoding_sex[1] = 1
        else:
            one_hot_encoding_sex[2] = 1


        num_finite_samples = np.size(np.isfinite(Signals))
        if num_finite_samples > 0:
            signal_mean = np.nanmean(Signals)
        else:
            signal_mean = 0.0
        if num_finite_samples > 1:
            signal_std = np.nanstd(Signals)
        else:
            signal_std = 0.0


        features = np.concatenate(([age], one_hot_encoding_sex, [signal_mean, 
                                                                 signal_std, 
                                                                 summary_data['QRS_prolonged_binary'], 
                                                                 summary_data['T_wave_inverted_binary'], 
                                                                 summary_data['P_wave_inverted_binary'], 
                                                                 summary_data['P_wave_absent_percentage'], 
                                                                 summary_data['P_wave_absent_binary'], 
                                                                 hrv_features['Mean_RR'], 
                                                                 hrv_features['STD_RR'], 
                                                                 hrv_features['Median_RR'], 
                                                                 hrv_features['Min_RR'], 
                                                                 hrv_features['Max_RR']]))
    
    
    except Exception as e:
        print("Error:", e)
        print("Available files:", os.listdir('.'))
        
        one_hot_encoding_sex = np.zeros(3, dtype=bool)
        if sex == 'Female':
            one_hot_encoding_sex[0] = 1
        elif sex == 'Male':
            one_hot_encoding_sex[1] = 1
        else:
            one_hot_encoding_sex[2] = 1


        num_finite_samples = np.size(np.isfinite(Signals))
        if num_finite_samples > 0:
            signal_mean = np.nanmean(Signals)
        else:
            signal_mean = 0.0
        if num_finite_samples > 1:
            signal_std = np.nanstd(Signals)
        else:
            signal_std = 0.0
            
        features = np.concatenate(([age], one_hot_encoding_sex, [signal_mean, signal_std]))
        features.resize(1,16)

    # Calculate WPD features and add to vector
    lead_index = 6
    signal_data = Signals[:,lead_index] 
    scaler = StandardScaler()
    signal_data = scaler.fit_transform(signal_data.reshape(-1, 1))
    wpd_features_vector = get_wpd_hos_vector(signal_data.squeeze())
    features = np.concatenate((features, wpd_features_vector))

    return np.asarray(features, dtype=np.float32)

# Save your trained model.
def save_model(model_folder, model):
    d = {'model': model}
    filename = os.path.join(model_folder, 'model.sav')
    joblib.dump(d, filename, protocol=0)

# Function to remove high frequency noise
def remove_high_frequency_noise(signal_data, sample_rate, cutoff_freq=40):
    nyquist = 0.5 * sample_rate
    normal_cutoff = cutoff_freq / nyquist
    order = 4
    b, a = signal.butter(order, normal_cutoff, btype='low')
    filtered_signal = signal.filtfilt(b, a, signal_data)
    return filtered_signal

def get_wpd_hos_vector(window_data, wavelet='sym3', level=5):
    """ Apply Wavelet Packets Decomposition (WPD) and return Higher Order Statistics (HOS)
        of mean, skewness and kurtosis for ECG window.
        
        Args:
            window_data (np.ndarray): ECG window data. Shape: (n_samples,). The data is already z-score scaled.
            wavelet (str): Mother wavelet (MW) to be used in the decomposition.
            level (int): The order of decomposition. 
            
        Returns:
            np.ndarray: Numpy array vector including the mean, skewness and kurtosis of the concatenated
                        decomposition coefficients from each of the sub-bands of the user-specified level.
                        The number of (mean, skew, kurt) triplets that are concetanted is given by the
                        formula: N_subbands = 2**level.
    """
    # Perform wavelet packets decomposition
    wpd = pywt.WaveletPacket(data = window_data,
                             wavelet = wavelet,
                             mode = 'symmetric',
                             maxlevel = level)    
    
    # Select the layer based on the user-specified level
    wpd_last_layer = wpd.get_level(level)
    
    # Return the concatenated array of the (mean, skew, kurt) triplets per sub-band in 'level' layer
    n_last_layer_nodes = 2**level
    wpd_hos_vector = np.empty([3*n_last_layer_nodes,])

    for node_i in range(n_last_layer_nodes):
        wpd_last_layer_node_data = wpd_last_layer[node_i].data
        wpd_hos_vector[3*node_i]   = np.mean(np.square(wpd_last_layer_node_data))

        skew_val = skew(wpd_last_layer_node_data)
        wpd_hos_vector[3*node_i+1] = skew_val if not np.isnan(skew_val) else 0
        
        kurtosis_val = kurtosis(wpd_last_layer_node_data)
        wpd_hos_vector[3*node_i+2] = kurtosis_val if not np.isnan(kurtosis_val) else 0

    return wpd_hos_vector 
