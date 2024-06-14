import os 
import sys 
from src.exception import CustomException
from src.logger import logging 
from dataclasses import dataclass
import pickle
import pandas as pd 
import numpy as np

@dataclass
class DataIngestionConfig : 
    data_source = os.path.join('..', 'WESAD')
    local_data_store = os.path.join('src', 'notebooks', 'data')
    wesad_bvp_data_store_location = os.path.join('src', 'notebooks', 'data', 'wesad_bvp_empatica.csv') 
    wesad_eda_data_store_location = os.path.join('src', 'notebooks', 'data', 'wesad_eda_empatica.csv')
    wesad_temp_data_store_location = os.path.join('src', 'notebooks', 'data',  'wesad_temp_empatica.csv')
    wesad_acc_data_store_location = os.path.join('src', 'notebooks', 'data', 'wesad_acc_empatica.csv') 
    wesad_resp_data_store_location = os.path.join('src', 'notebooks', 'data', 'wesad_resp_empatica.csv') 
    wesad_emg_data_store_location = os.path.join('src', 'notebooks', 'data', 'wesad_emg_empatica.csv') 
    wesad_ecg_data_store_location = os.path.join('src', 'notebooks', 'data', 'wesad_ecg_empatica.csv') 
    wesad_label_data_store_location = os.path.join('src', 'notebooks', 'data', 'wesad_label_empatica.csv') 
    
    
class DataIngestion : 
    def __init__(self) -> None:
        self.ingestion_config = DataIngestionConfig() 
    
    def read_pkl_file(self, subject_id:str) : 
        pkl_dir=os.path.join(self.ingestion_config.data_source, subject_id)
        pkl_file=os.path.join(pkl_dir, f'{subject_id}.pkl')
        with open(pkl_file, 'rb') as file : 
            lines = pickle.load(file, encoding="latin1")
            
        return lines
    
    def get_subjects_data(self) : 
        
        logging.info('Read pkl files')
        subjects_data = {}
        subjects_lst = os.listdir(self.ingestion_config.data_source)
        subjects_lst = [item for item in subjects_lst if item.startswith('S') and item[1:].isdigit()]
        for subject in subjects_lst : 
            subject_file_path = os.path.join(self.ingestion_config.data_source, subject, f'{subject}.pkl')
            temp_data = pd.read_pickle(subject_file_path)
            subjects_data[subject] = temp_data
            
        return subjects_data
    
    def construct_dataframes(self, subjects_data) : 
        
        logging.info('COPYING DATA INTO DATAFRAMES...')
        subjects_lst = os.listdir(self.ingestion_config.data_source)
        subjects_lst = [item for item in subjects_lst if item.startswith('S') and item[1:].isdigit()]
        bvp_dict = {
            'subject' : [], 
            'BVP' : []
        }
        eda_dict = {
            'subject' : [], 
            'EDA' : []
        }
        temp_dict = {
            'subject' : [], 
            'TEMP' : []
        }
        acc_dict = {
            'subject' : [], 
            'x_axis' : [], 
            'y_axis' : [], 
            'z_axis' : [], 
        }
        resp_dict = {
            'subject' : [], 
            'RESP' : []
        }
        
        emg_dict = {
            'subject' : [], 
            'EMG' : []
        }
        
        ecg_dict = {
            'subject' : [], 
            'ECG' : []
        }
        
        label_dict = {
            'label' : []
        }
        
        for subject in subjects_lst : 

            x_acc_data = subjects_data[subject]['signal']['wrist']['ACC'].T[0]
            y_acc_data = subjects_data[subject]['signal']['wrist']['ACC'].T[1]
            z_acc_data = subjects_data[subject]['signal']['wrist']['ACC'].T[2]
            temp_data = subjects_data[subject]['signal']['wrist']['TEMP']
            eda_data = subjects_data[subject]['signal']['wrist']['EDA']
            bvp_data = subjects_data[subject]['signal']['wrist']['BVP']
            resp_data = subjects_data[subject]['signal']['chest']['Resp']
            emg_data = subjects_data[subject]['signal']['chest']['EMG']
            ecg_data = subjects_data[subject]['signal']['chest']['ECG']
            labels_data = subjects_data[subject]['label']
            
            temp_subject = [subject] * temp_data.shape[0]
            eda_subject = [subject] * eda_data.shape[0]
            bvp_subject = [subject] * bvp_data.shape[0]
            acc_subject = [subject] * x_acc_data.shape[0]
            resp_subject = [subject] * resp_data.shape[0]
            emg_subject = [subject] * emg_data.shape[0]
            ecg_subject = [subject] * ecg_data.shape[0]
            
            for i in range(0, len(temp_data)): 
                temp_dict['TEMP'].append(temp_data[i][0])
                temp_dict['subject'].append(temp_subject[i])
            for i in range(0, len(eda_data)): 
                eda_dict['EDA'].append(eda_data[i][0])
                eda_dict['subject'].append(eda_subject[i])
                
            for i in range(0, len(bvp_data)): 
                bvp_dict['BVP'].append(bvp_data[i][0])
                bvp_dict['subject'].append(bvp_subject[i])
            
            for i in range(0, len(x_acc_data)): 
                acc_dict['subject'].append(acc_subject[i])
                acc_dict['x_axis'].append(x_acc_data[i])
                acc_dict['y_axis'].append(y_acc_data[i])
                acc_dict['z_axis'].append(z_acc_data[i])
                
            for i in range(0, len(resp_data)): 
                resp_dict['RESP'].append(resp_data[i][0])
                resp_dict['subject'].append(resp_subject[i])
            
            for i in range(0, len(emg_data)): 
                emg_dict['EMG'].append(emg_data[i][0])
                emg_dict['subject'].append(emg_subject[i])
        
            for i in range(0, len(ecg_data)): 
                ecg_dict['ECG'].append(ecg_data[i][0])
                ecg_dict['subject'].append(ecg_subject[i])
                
            for i in range(0, len(labels_data)): 
                label_dict['label'].append(labels_data[i])

        bvp_df = pd.DataFrame(bvp_dict)
        eda_df = pd.DataFrame(eda_dict)
        temp_df = pd.DataFrame(temp_dict)
        acc_df = pd.DataFrame(acc_dict)
        resp_df = pd.DataFrame(resp_dict)
        emg_df = pd.DataFrame(emg_dict)
        ecg_df = pd.DataFrame(ecg_dict)
        labels_df = pd.DataFrame(label_dict)
        
        return (
                bvp_df, eda_df, temp_df, acc_df, 
                resp_df, emg_df, 
                ecg_df, labels_df)
        
    def initiate_data_ingestion(self) : 
        logging.info("Start data ingestion operation. ")
        try : 
            # Read pkl files
            subjects_data = self.get_subjects_data() 
            
            # construct a dataframe
            bvp_df, eda_df, temp_df, acc_df, resp_df, emg_df, ecg_df, label_df = self.construct_dataframes(subjects_data=subjects_data)
            
            # save the dataframe as csv
            os.makedirs(self.ingestion_config.local_data_store, exist_ok=True)
            logging.info('SAVING BVP DATA')
            bvp_df.to_csv(self.ingestion_config.wesad_bvp_data_store_location, index=False, sep=';')
            logging.info('SAVING EDA DATA')
            eda_df.to_csv(self.ingestion_config.wesad_eda_data_store_location, index=False, sep=';')
            logging.info('SAVING TEMP DATA')
            temp_df.to_csv(self.ingestion_config.wesad_temp_data_store_location, index=False, sep=';')
            logging.info('SAVING ACC DATA')
            acc_df.to_csv(self.ingestion_config.wesad_acc_data_store_location, index=False, sep=';')
            logging.info('SAVING RESP DATA')
            resp_df.to_csv(self.ingestion_config.wesad_resp_data_store_location, index=False, sep=';')
            logging.info('SAVING EMG DATA')
            emg_df.to_csv(self.ingestion_config.wesad_emg_data_store_location, index=False, sep=';')
            logging.info('SAVING ECG DATA')
            ecg_df.to_csv(self.ingestion_config.wesad_ecg_data_store_location, index=False, sep=';')
            logging.info('SAVING LABEL DATA')
            label_df.to_csv(self.ingestion_config.wesad_label_data_store_location, index=False, sep=';') 
            logging.info('SAVE COMPLETED')
     
        except Exception as e: 
            raise CustomException(e, sys)
        
        

if __name__ == "__main__":
    data_ingestor = DataIngestion() 
    data_ingestor.initiate_data_ingestion()