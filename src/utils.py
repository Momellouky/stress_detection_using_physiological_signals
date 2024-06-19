import pandas as pd
import numpy as np 
from scipy.fft import fft, fftfreq
from scipy.signal import butter, filtfilt
from scipy.stats import mode


class FeatureExtraction : 
    
    def __init__(self) -> None:
        pass
    
    def feature_extraction(self, data, window_size, signal_name, sampling_rate) : 
        subjects_lst = data.subject.unique() 

        temp_subject_lst = []
        temp_feats_lst = []
        for subject in subjects_lst : 
            subject_data = data[data['subject'] == subject][signal_name] 
            features = self.non_overlapping_rolling_stats(data=subject_data, window_size=window_size, signal_name=signal_name, sampling_rate=sampling_rate)
            temp_feats_lst.append(features)
            for _ in range(0, int(features.shape[0])) : 
                temp_subject_lst.append(subject)


        features = pd.DataFrame()
        for feat_df in temp_feats_lst : 
            features = pd.concat([features, feat_df], axis=0)



        features = features.reset_index()
        features = pd.concat([features, pd.DataFrame(temp_subject_lst)], axis=1)
        features = features.rename(columns={
            0 : 'subject'
        })
        
        print(f'{signal_name} feature extraction : feature set shape : {features.shape}')
        
        return features
    
    def non_overlapping_rolling_stats(self, data, window_size, signal_name, sampling_rate):
    
        peak_freq_signals = ['EMG', 'BVP']
        slope_signals = ["TEMP"]
        
        mean_vals = []
        std_vals = []
        min_vals = []
        max_vals = []
        peak_freqs = []
        slopes = []
        
        for start in range(0, len(data), window_size):
            window_data = data[start:start + window_size]
            if len(window_data) == window_size:
                # mean_vals.append(np.round(window_data.mean(), 2))
                # std_vals.append(np.round(window_data.std(ddof=0), 2))
                # min_vals.append(np.round(window_data.min(), 2))
                # max_vals.append(np.round(window_data.max(), 2))
                
                mean_vals.append(np.mean(data))
                std_vals.append(np.std(window_data, ddof=0))
                min_vals.append(np.min(data))
                max_vals.append(np.max(data))
                
                if signal_name in peak_freq_signals : 
                    # FFT to calculate peak frequency
                    yf = fft(window_data.values)
                    xf = fftfreq(window_size, 1 / sampling_rate)
                    peak_freq = xf[np.argmax(np.abs(yf))]
                    peak_freqs.append(peak_freq)
                    
                if signal_name in slope_signals : 
                    x = np.arange(window_size)
                    y = window_data
                    p = np.polyfit(x, y, 1)
                    slopes.append(p[0])
        
        if signal_name in peak_freq_signals : 
            result = pd.DataFrame({
            f'{signal_name}mean': mean_vals,
            f'{signal_name}std': std_vals,
            f'{signal_name}min': min_vals,
            f'{signal_name}max': max_vals, 
            f'{signal_name}peak_freq': peak_freqs
            })
        elif signal_name in slope_signals : 
            result = pd.DataFrame({
            f'{signal_name}mean': mean_vals,
            f'{signal_name}std': std_vals,
            f'{signal_name}min': min_vals,
            f'{signal_name}max': max_vals, 
            f'{signal_name}slope': slopes
            })
        else : 
            result = pd.DataFrame({
            f'{signal_name}mean': mean_vals,
            f'{signal_name}std': std_vals,
            f'{signal_name}min': min_vals,
            f'{signal_name}max': max_vals
        })
        return result
    
    def non_overlapping_rolling_labels(self, data, window_size):
        mode_vals = []
        
        for start in range(0, len(data), window_size):
            window_data = data[start:start + window_size]
            if len(window_data) == window_size:
                mode_vals.append(mode(window_data).mode[0])  
        
        result = pd.DataFrame({
            'label': mode_vals
        })
        return result
    
    
class Filter : 
    def __init__(self) -> None:
        pass
    
    def butter_lowpass(self, cutoff, fs, order=5):
        nyquist = 0.5 * fs
        normal_cutoff = cutoff / nyquist
        b, a = butter(order, normal_cutoff, btype='low', analog=False)
        return b, a

    def lowpass_filter(self, data, cutoff, fs, order=5):
        b, a = self.butter_lowpass(cutoff, fs, order=order)
        y = filtfilt(b, a, data)
        return y