import cvxEDA.src.cvxEDA as cvxeda
from dataclasses import dataclass
from multiprocessing import Process, Queue
import os
import time
import sys
import glob
import pandas as pd 
import numpy as np

@dataclass
class EdaSignalDecomposerConfig : 
    data_store = os.path.join('src', 'notebooks', 'data')
    eda_data_store = os.path.join(data_store, 'wesad_eda_empatica.csv')
    eda_component_save_dir = os.path.join(data_store, 'eda_batch_components')
    eda_merged_component_save_dir = os.path.join(data_store, 'eda_merged_components')
    

class EdaSignalDeconposer : 
    
    def __init__(self) -> None:
        self.eda_signal_decomposer_config = EdaSignalDecomposerConfig()
        
    def _run_cvxEDA(self, eda_signal, delta):
        """Run the cvxEDA method on EDA signal. 

        Args:
            eda_signal (Pd DataFrame): raw eda signal
            delta (float): 1 / sampling_freq
            result_queue (Multiprocessing queue): A communication channel between parallel processes
        """
        
        print("===============In run_cvxEDA function===============") 
        
        result = cvxeda.cvxEDA(y=eda_signal, delta=delta)
        temp = []
        for el in result : 
            temp.append(el) 
        
        phasic_comp =  temp[0]
        smna_comp =  temp[1] 
        tonic_comp =  temp[2]

        comps = [phasic_comp, smna_comp, tonic_comp]
        
        # if result_queue.full(): 
        #     print("Queue full. We can't push additional data. ")
        # else : 
        #     print("Queue size is sufficient")
        #     result_queue.put(comps)
        
        # print("===============End run_cvxEDA function===============")
        # return comps 
        return comps
    
    @DeprecationWarning
    def decompose(self, n_hz:int = 4, keep_list:list = None, save_file:bool = False) : 
        filtered_eda_df = pd.DataFrame()
        eda_temp_df = pd.DataFrame() 
        
        write_mode = 'OFF'
        if save_file == True : 
            write_mode = 'ON'
        
        print("=============== Loading the dataframe... ===============")
        data_df = pd.read_csv(self.eda_signal_decomposer_config.eda_data_store, sep=";")
        print("============ Dataframe Loaded ============")
        

        print("=============== Dropping subjects with invalid labels... ===============")
        eda_df = data_df
        # data_df = data_df.drop(data_df[(data_df['subject'] == 'S11') | (data_df['subject'] == 'S13') | (data_df['subject'] == 'S61')].index)
        # eda_df = data_df[data_df['EDA'].isna() == False][['EDA', 'subject', 'trial']]
        if eda_df['EDA'].dtype == str : 
            eda_df['EDA'] = eda_df['EDA'].str.replace(',', '.').astype('float')
        

        # Create a queue to store the result
        print("============ Init the Queue. ============")
        result_queue = Queue()
        
        # Start the process
        print("============ Init the Process constructor. ============")
        start_time = time.time()
        if keep_list:
            print(f"- Dropping noise subjects")
            mask = eda_df['subject'].isin(keep_list)
            eda_df = eda_df[mask]
        else:
            # filtered_df = pd.read_csv('../../data/filtered_data.csv', sep=';')
            # subject_trial_pairs = set(zip(filtered_df['Subject'], filtered_df['Trial']))
            # self.add_subject_trials(filtered_df, subject_trial_pairs, ['S11', 'S13', 'S61'])
            # filtered_eda_df = self.filter_df(eda_df, subject_trial_pairs)
            # eda_df = filtered_eda_df
            # filtered_df = pd.read_csv('../../data/filtered_eda.csv', sep=';')
            # eda_df = filtered_df
            pass
        
        i = 0 
        batch_count = 1
        step = n_hz * 60 * 2
        stop_flag = False
        
        while True : 
            if stop_flag == True : 
                break
            
            start_index = i
            end_index = min(start_index + step, len(eda_df['EDA']))
            print(f"Processing batch: {batch_count}, from {start_index} to {end_index}")
            p = Process(target=self._run_cvxEDA, args=(eda_df['EDA'][start_index  : end_index], 1 / n_hz, result_queue)) 
            
            print("============ Start the Process... ============")
            p.start()
            
            print("============ Retrieve results ============")
            result = result_queue.get()

            print("Result type: ", type(result))
            print("Result:\n", result)
            end_time = time.time()
            p.join(timeout=10)
            
            if p.is_alive() == True : 
                print("timeout exceeded. ")
                p.terminate()


            # Get EDA components 
            phasic = result[0]
            smna = result[1]
            tonic = result[2]
            
            print(f"- writing data in the file is {write_mode}")

            if save_file == True : 
            
                os.makedirs(self.eda_signal_decomposer_config.eda_component_save_dir, exist_ok=True)
                
                
                BATCH_DIR = self.eda_signal_decomposer_config.eda_component_save_dir + f"/eda_comps_{batch_count}"   
                os.makedirs(BATCH_DIR)
                
                PHASIC_DIR = BATCH_DIR + '/phasic_comp.csv' 
                TONIC_DIR = BATCH_DIR + '/tonic_comp.csv' 
                SMNA_DIR = BATCH_DIR + '/smna_comp.csv' 
                
                np.savetxt(PHASIC_DIR, phasic, delimiter=",")
                np.savetxt(TONIC_DIR, tonic, delimiter=",")
                np.savetxt(SMNA_DIR, smna, delimiter=",")

            batch_count += 1 
            i += step 

            if i >= eda_df.shape[0] : 
                stop_flag = True 
         
        print("=============== END OF THE PROGRAM ===============")
        print(f"=============== EDA Decomposition in: {end_time - start_time} ===============")
        
    @DeprecationWarning
    def merge_component(self, component : str = 'phasic', n_hz : int = 4, save_file : bool = False) : 

        
        write_mode = 'OFF' 
        if save_file == True : 
            write_mode = 'ON'
        
        eda_component = []
        
        print(
            f"=============== Merge the {component} component of the EDA signal... ==============="
        )
        
        PARAMS = {
            'batch_no': 1, 
            'first_min': n_hz * 60 * 2, 
            'step': 4 * 60 * 2, 
            'stop' : False    
        }
        
        while not PARAMS['stop'] :  
            
            FOLDER_PATH = self.eda_signal_decomposer_config.eda_component_save_dir + f"/eda_comps_{PARAMS['batch_no']}"
            PARAMS['batch_no'] += 1
            
            print(f"FOLDER_PATH : {FOLDER_PATH}")
            FILE_NAME = f'{component}_comp.csv'  # Pattern to match all CSV files
            
            print(f"Reading file : {FOLDER_PATH + '/' + FILE_NAME}")
            file_content = self._get_csv_content(FOLDER_PATH, FILE_NAME)
            print(f"file_content: {file_content}")
            
            if file_content is None  : 
                PARAMS['stop'] = True
                continue
            
            
            eda_component.append(file_content)  
        
        ####################### Merging the EDA compoent ended #######################
        
        print(
            f"=============== Merging the {component} component Ended ==============="
        )
        
        print(
            f"=============== Phasic component content: ==============="
        )
        print(f"{eda_component}")
        
        eda_component = np.concatenate(eda_component)
        
        print(f"=============== Saving EDA {component} component  ===============")
        
        print(f"- writing data in the file is {write_mode}")
        if save_file == True : 
            os.makedirs(self.eda_signal_decomposer_config.eda_merged_component_save_dir, exist_ok=True)
                
            COMPONENT_DIR = self.eda_signal_decomposer_config.eda_merged_component_save_dir + f"/eda_{component}_comp.csv"
            np.savetxt(COMPONENT_DIR, eda_component, delimiter=",")  
        
        print(f"Program Ended WITHOUT errors. ")
        print(f"=============== Program Ended WITHOUT errors.  ===============")
    
    @DeprecationWarning
    def _get_csv_content(self, folder_path, file_name_pattern):
        """
        Get the content of a specific CSV file from a folder as a NumPy array.
        
        Parameters:
            folder_path (str): Path to the folder containing CSV files.
            file_name_pattern (str): Pattern to match the desired CSV file.
                                    Example: '*.csv' will match all CSV files.
        
        Returns:
            np.ndarray or None: Content of the matched CSV file as a NumPy array,
                                or None if the file is not found or cannot be read.
        """
        search_pattern = os.path.join(folder_path, file_name_pattern)
        
        matching_files = glob.glob(search_pattern)
        
        if matching_files:
        
            try:
                csv_content = np.genfromtxt(matching_files[0], delimiter=',')
                return csv_content
            except Exception as e:
                print(f"Error reading CSV file: {e}")
                return None
        else:
            
            return None
        
    def decompose_eda(self, frequency:int = 4) : 
        
        res = {
            'subject' : [],
            'phasic' : [], 
            'tonic' : [], 
            'smna' : []
        }
        
        eda_df = pd.read_csv(self.eda_signal_decomposer_config.eda_data_store, sep=';') 
        subjects_lst = eda_df.subject.unique() 
        print(f'subjects_lst  : {subjects_lst}')
        for subject in subjects_lst : 
            data = eda_df[eda_df['subject'] == subject] 
            eda_comps = self._run_cvxEDA(data['EDA'], 1 / frequency)
            for phasic in eda_comps[0] : 
                res['phasic'].append(phasic)
                
            for smna in eda_comps[1] : 
                res['smna'].append(smna)
                
            for tonic in eda_comps[2] : 
                res['tonic'].append(tonic)
                
            for _ in range(0, len(eda_comps[0])) : 
                res['subject'].append(subject)
                
        eda_comps_df = pd.DataFrame(res) 
        
        return eda_comps_df
        
if __name__ == '__main__' : 
    
    decomposer = EdaSignalDeconposer() 
    eda_comps_df = decomposer.decompose_eda() 
    eda_comps_df.to_csv(f'{decomposer.eda_signal_decomposer_config.data_store}/eda_comps.csv', sep=';', index=False)
    del decomposer