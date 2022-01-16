import torch
import torchaudio
import torchaudio.transforms as T
import numpy as np
import pandas as pd
import h5py
import os
import logging

from glob import glob
from itertools import chain
from tqdm import tqdm

from pcen import PCENTransform

log_fmt = '%(asctime)s - %(module)s - %(levelname)s - %(message)s'
logging.basicConfig(level=logging.INFO, format=log_fmt)
logger = logging.getLogger(__name__)

def create_labels(label, feature, train_file, seg_len, hop_seg, fps):

    '''Chunk the time-frequecy representation to segment length and store in h5py dataset
    Args:
        -df_pos : dataframe
        -log_mel_spec : log mel spectrogram
        -glob_cls_name: Name of the class used in audio files where only one class is present
        -file_name : Name of the csv file
        -train_file: h5py object
        -seg_len : fixed segment length
        -fps: frame per second
    Out:
        - label_list: list of labels for the extracted mel patches'''

    label_list = []
    if len(train_file['features'][:]) == 0:
        file_index = 0
    else:
        file_index = len(train_file['features'][:])

    str_ind = 0
    end_ind = feature.shape[1]

    if (end_ind - str_ind) > seg_len:
        shift = 0
        while end_ind - (str_ind + shift) > seg_len:

            feature_patch = feature[int(str_ind + shift):int(str_ind + shift + seg_len)]
            
            train_file['features'].resize((file_index + 1, feature_patch.shape[0], feature_patch.shape[1]))
            train_file['features'][file_index] = feature_patch
            label_list.append(label)
            file_index += 1
            shift = shift + hop_seg

        feature_patch_last = feature[end_ind - seg_len:end_ind]

        
        train_file['features'].resize((file_index + 1 , feature_patch.shape[0], feature_patch.shape[1]))
        
        train_file['features'][file_index] = feature_patch_last
        label_list.append(label)
        file_index += 1

    elif (end_ind - str_ind) < seg_len:

        # If patch length is less than segment length 
        # then tile the patch multiple times 

        feature_patch = feature[str_ind:end_ind]
        #if feature_patch.shape[0] == 0:
        #    logger.warning("The patch is of 0 length")

        repeat_num = int(seg_len / (feature_patch.shape[0])) + 1
        feature_patch_new = np.tile(feature_patch, (repeat_num, 1))
        feature_patch_new = feature_patch_new[0:int(seg_len)]
        train_file['features'].resize((file_index+1, feature_patch_new.shape[0], feature_patch_new.shape[1]))
        train_file['features'][file_index] = feature_patch_new
        label_list.append(label)
        file_index += 1
    
    #logger.info("Total files created : {}".format(file_index))
    return label_list

def time2frame(df, fps):
    'Margin of 25 ms around the onset and offsets'
    df.loc[:,'Starttime'] = df['Starttime'] - 0.025
    df.loc[:,'Endtime'] = df['Endtime'] + 0.025

    'Converting time to frames'
    start_time = [int(np.floor(start * fps)) for start in df['Starttime']]
    end_time = [int(np.floor(end * fps)) for end in df['Endtime']]

    return start_time, end_time

def melSpectFeature(conf, audio_path, aug):
    """
    Function to make Mel Spectrogram features with optional augmentation
    """
    data, sr = torchaudio.load(audio_path)
    resample = T.Resample(sr, conf.features.sample_rate)
    data = resample(data)
    #data = (data - torch.mean(data)) / torch.std(data)
    chunk_samples = int(conf.features.sample_rate * 0.25)
    feature = torch.Tensor([])
    
    for idx in range(0, data.shape[1]-chunk_samples, chunk_samples):
        chunk_feature = T.MelSpectrogram(
            n_fft=conf.features.n_fft,
            hop_length=conf.features.hop,
            power=1,
            f_min=conf.features.fmin,
            f_max=conf.features.fmax,
            n_mels=conf.features.n_mels,
        )(data[0][idx:idx+chunk_samples])
        if aug:
            if torch.rand(1) > 0.5:
                chunk_feature = T.FrequencyMasking(freq_mask_param=conf.features.freq_mask)(chunk_feature)
            if torch.rand(1) > 0.5:  
                chunk_feature = T.TimeMasking(time_mask_param=conf.features.time_mask)(chunk_feature)
        feature = torch.cat((feature, chunk_feature), dim=1)

    if conf.features.frontend == 'PCEN':
        feature = torch.unsqueeze(feature, dim=0)
        feature = PCENTransform(conf=conf)(feature)
    elif conf.features.frontend == 'log':
        feature = torch.log(feature + 1E-6)
    else:
        feature = feature

    feature = torch.squeeze(feature)
    feature = torch.transpose(feature, 0, 1) 
    return feature

def featureExtract(conf=None,mode=None):
    '''
    Training:

    Evaluation:
        Currently using the validation set for evaluation.
        
        For each audio file, extract time-frequency representation and create 3 subsets:
        a) Positive set - Extract segments based on the provided onset-offset annotations.
        b) Negative set - Since there is no negative annotation provided, we consider the entire
                        audio file as the negative class and extract patches of length conf.seg_len
        c) Query set - From the end time of the 5th annotation to the end of the audio file.
                        Onset-offset prediction is made on this subset.
    Args:
    - config: config object
    - mode: train/valid

    Returns:
    - Num_extract_train/Num_extract_valid - Number of samples in training/validation set
    '''
    label_tr = []
    fps = conf.features.sample_rate / conf.features.hop

    #Converting fixed segment legnth to frames
    seg_len = int(round(conf.features.seg_len * fps))
    hop_seg = int(round(conf.features.hop_seg * fps))
    
    aug = False

    if mode == 'train':
        aug = conf.features.aug_train
        logger.info("=== Processing training set ===")
        csv_files = [file for path, _, _ in os.walk(conf.path.train_dir) 
                     for file in glob(os.path.join(path, '*0.wav')) ]
        train_file_dir = os.path.join(conf.path.train_feat,'train.h5')
        train_file = h5py.File(train_file_dir,'w')
        train_file.create_dataset('features', shape=(0, seg_len, conf.features.n_mels),
                          maxshape=(None, seg_len, conf.features.n_mels))
        num_extract = 0
        for file in tqdm(csv_files):
            split_list = file.split('/')
            cls_name = split_list[-2]
            breakpoint()
            file_name = split_list[-1]
            #logger.info("Processing file name {}".format(file_name))
            feature = melSpectFeature(conf, file, aug)
            label_list = create_labels(
                cls_name, feature,
                train_file, seg_len,
                hop_seg, fps
            )
            label_tr.append(label_list)
        
        logger.info("Feature extraction for training set complete")
        
        num_extract = len(train_file['features'])
        flat_list = [item for sublist in label_tr for item in sublist]
        
        train_file.create_dataset('labels', data=[s.encode() for s in flat_list], dtype='S20')
        data_shape = train_file['features'].shape
        train_file.close()
        
        return num_extract, data_shape
    else:
        logger.info("=== Processing Test set ===")
        csv_files = [file for path, _, _ in os.walk(conf.path.val_dir) 
                    for file in glob(os.path.join(path, '*.csv')) ]
        num_extract = 0

        for file in csv_files:
            positive_idx = 0
            
            negative_idx = 0
            negative_start = 0
            negative_hop = 0

            query_idx = 0
            query_start = 0
            query_hop = 0

            start_index = 0
            end_index = 0

            split_list = file.split('/')
            filename = str(split_list[-1].split('.')[0]) + '.h5'
            
            # path to audio file
            audio_path = file.replace('csv', 'wav')

            # actually create val hdf5 file
            test_file_dir = os.path.join(conf.path.test_feat, filename)
            test_file = h5py.File(test_file_dir, 'w')

            # create datasets for positive set, negative set, and query set
            test_file.create_dataset('feat_positive', shape=(1, seg_len, conf.features.n_mels),
                          maxshape=(None, seg_len, conf.features.n_mels))
            test_file.create_dataset('feat_negative', shape=(1, seg_len, conf.features.n_mels),
                          maxshape=(None, seg_len, conf.features.n_mels))
            test_file.create_dataset('feat_query', shape=(1, seg_len, conf.features.n_mels),
                          maxshape=(None, seg_len, conf.features.n_mels))

            test_file.create_dataset('query_index_start',shape=(1,),maxshape=(None))

            df_test = pd.read_csv(file, header=0, index_col=False)
            Q_list = df_test['Q'].to_numpy()              
            start_time, end_time = time2frame(df_test, fps)

            index_sup = np.where(Q_list == 'POS')[0][:conf.train.n_shot]
            feature, df_test = melSpectFeature(conf, audio_path, df_test, aug=False)

            query_idx_start = end_time[index_sup[-1]]
            negative_idx_end = feature.shape[0] - 1
            test_file['query_index_start'][:] = end_time[index_sup[-1]]

            logger.info('=== Processing Test File: {} ==='.format(filename))
            logger.info('=== Creating negative dataset ===')
            while negative_idx_end - (start_index + negative_hop) > seg_len:
                negative_patch = feature[int(start_index + negative_hop):
                                         int(start_index + negative_hop + seg_len)]
                
                test_file['feat_negative'].resize((negative_idx + 1, 
                                                  negative_patch.shape[0],
                                                  negative_patch.shape[1]))
                test_file['feat_negative'][negative_idx] = negative_patch
                
                negative_idx += 1
                negative_hop += hop_seg

            negative_patch = feature[negative_idx_end - seg_len:negative_idx_end]
                
            test_file['feat_negative'].resize((negative_idx + 1, 
                                              negative_patch.shape[0],
                                              negative_patch.shape[1]))
            test_file['feat_negative'][negative_idx] = negative_patch
            
            logger.info('=== Creating positive dataset ===')
            for index in index_sup:
                start_index = int(start_time[index])
                end_index = int(end_time[index])

                if (end_index - start_index) > seg_len:
                    shift = 0
                    while end_index - (start_index + shift) > seg_len:
                        positive_patch = feature[int(start_index + shift):
                                                int(start_index + shift + seg_len)]

                        test_file['feat_positive'].resize((positive_idx + 1,
                                                          positive_patch.shape[0],
                                                          positive_patch.shape[1]))
                        test_file['feat_positive'][positive_idx] = positive_patch
                        positive_idx += 1
                        shift += hop_seg
                    
                    positive_patch = feature[end_index - seg_len:end_index]
                    test_file['feat_positive'].resize((positive_idx + 1, 
                                              positive_patch.shape[0],
                                              positive_patch.shape[1]))
                    test_file['feat_positive'][positive_idx] = positive_patch   
                    positive_idx += 1

                else:
                    positive_patch = feature[start_index:end_index]

                    if positive_patch.shape[0] == 0:
                        logger.warning("The patch is of 0 length")
                        continue
                    
                    repeat_num = int(seg_len / (positive_patch.shape[0])) + 1

                    patch_new = np.tile(positive_patch, (repeat_num, 1))
                    patch_new = patch_new[0:int(seg_len)]
                    test_file['feat_positive'].resize((positive_idx + 1, patch_new.shape[0], patch_new.shape[1]))
                    test_file['feat_positive'][positive_idx] = patch_new
                    positive_idx += 1
            
            logger.info('=== Creating query dataset ===')
            while negative_idx_end - (query_idx_start + query_hop) > seg_len:
                query_patch = feature[int(query_idx_start + query_hop):
                                      int(query_idx_start + query_hop + seg_len)]
                test_file['feat_query'].resize((query_idx + 1,
                                               query_patch.shape[0],
                                               query_patch.shape[1]))
                test_file['feat_query'][query_idx] = query_patch
                query_idx += 1
                query_hop += hop_seg

            query_patch = feature[negative_idx_end - seg_len:negative_idx_end]
            test_file['feat_query'].resize((query_idx + 1,
                                          query_patch.shape[0],
                                          query_patch.shape[1]))
            test_file['feat_query'][query_idx] = query_patch
            
            num_extract += len(test_file['feat_query'])
            test_file.close()
