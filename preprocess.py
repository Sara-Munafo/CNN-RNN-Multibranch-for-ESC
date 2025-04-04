import numpy as np
import soundfile as sf
from data_helper import Clip

import pickle


######################################################################################################################
#                                        Block 1: Single preprocessing functions 
######################################################################################################################     

def create_labels_array(dataset):
    """
    Create one-hot encoded labels array from clip.category (str)
    Parameters:
        - dataset(list): list of Clip objects
    Returns:
        - labels(np.array) : array of shape (n_samples,n_classes) of one-hot encoded vectors
    """
    # Get the list of all classes
    categories = np.unique([clip.category for clip in dataset])

    y_data = np.zeros((len(dataset),categories.shape[0]))
    for i,clip in enumerate(dataset):
        # Get the index of specific label from 'categories'
        index = int(np.argwhere(categories==clip.category)[0][0])
        # Assing 1 at selected position in the y_data one hot encoded array
        y_data[i,index]=1
    return y_data



def standardize_features(features):
    '''
    Standardize all features to have zero mean and variance 1
    Parameters:
        - features(array): array of features computed from a Clip object;
    Returns:
        - features(array): standardized features
    '''
    features = (features - np.mean(features)) / np.std(features+1e-6)
    return features



def segmentation(dataset, y_data, architecture, mode):
    '''
    Segment the audio features into partially superimposed smaller segments.
    Parameters:
        - dataset(list) : list of Clips objects;
        - y_data(np.array) : array of one-hot encoded labels
        - architecture(str) : 'multibranch' or 'sequential'; if 'multibranch', computes also MFCC.
        - mode(str) : 'train' or 'test'; determines whether to also return folds (train/validation) or not; in both cases, returns also ids
                        identifying to which clip each segment belongs.
    Returns:
        - segments_cnn2d(np.array) : 2-channel array with segments of log-scaled mel spectrograms, stacked with their deltas:
                                    shape = [segments_per_clip*n_clips, mel_bands=60, segment_length=101, 2]
        - segments_rnn(np.array) : array with segments of MFCC
                                    shape = [segments_per_clip*n_clips,segment_length=101, n_coefficients=13]
        - labels(np.array) : array with labels corresponding to each segment;
        - folds(np.array) : array with c-v folds corresponding to each segment.
        - clip_id(np.array) : uniquely clip identifying ids.
    '''
    segments_cnn2d = []
    segments_rnn = []
    labels = []
    folds = []
    clip_id = []

    # Segmentation specifics
    segment_length = 101
    hop_size = int(0.5*segment_length)

    for i in range(len(dataset)):
    
        clip = dataset[i]
        label = y_data[i]
        fold = clip.fold

        # Get (standardized) log-mel-spectrogram and deltas for the whole Clip
        mel_spec = standardize_features(clip.logamplitude)
        delta = standardize_features(clip.delta)
        clip_length = mel_spec.shape[1]

        if architecture=='multibranch': #Compute (standardized) MFCC for RNN branch
            mfcc = standardize_features(clip.mfcc)
            clip_length = mfcc.shape[0]

    
        #Generate overlapping segments
        for start in range(0, clip_length - segment_length + 1, hop_size):
            #Stack segment features
            # Mel-spectrograms and their deltas
            seg_spec = mel_spec[:, start:start + segment_length]  #(60, 101)
            seg_delta = delta[:, start:start + segment_length]
            combined_spatial = np.stack((seg_spec, seg_delta), axis=0)
            #Reshape for tf: (channels, height, width) --> (height, width, channels)
            combined_spatial = np.transpose(combined_spatial, (1, 2, 0))
            #Add to the dataset
            segments_cnn2d.append(combined_spatial)

            if architecture=='multibranch':
                seg_mfcc = mfcc[start:start+segment_length, :]  #(101,13) (for rnn shape is (T,feat))
                segments_rnn.append(seg_mfcc)
                                
            labels.append(label)

            # Append clip ID for the segment
            clip_id.append(i)

            # Append fold only in the train/validation phase
            if mode=='train':
                folds.append(int(fold))

    segments_cnn2d = np.array(segments_cnn2d)
    segments_rnn = np.array(segments_rnn)
    labels = np.array(labels)
    clip_id = np.array(clip_id)
    
    if mode=='test':
        return(segments_cnn2d, segments_rnn, labels, clip_id)
    else:
        return(segments_cnn2d, segments_rnn, labels, folds, clip_id)
    


def divide_folds(seg_feature, seg_labels, clip_id, seg_folds):
    '''
    Organize the dataset and labels into folds for cross-validation
    Parameters:
        - seg_feature(np.ndarray): computed features over segments;
        - seg_labels(np.ndarray): array of one-hot encoded labels;
        - seg_folds(list): list of folds
    Returns:
        - X(np.ndarray) : (n_folds,n_segments_per_folds,(segment_shape)), features divided by folds;
        - Y(np.array) : (n_folds,n_segments_per_folds,n_classes=50), segment labels divided by folds.
    '''
    folds = [1,2,3,4,5]
    # Set fold_size (each fold has the same size)
    fold_size = int(seg_feature.shape[0] / 5)
    Y = np.zeros((5, fold_size, 50))

    seg_length = 101

    # Division is based on the original dimension of the feature segment, which changes if we are working with Mel spec. or MFCC
    
    # For Mel-spectrograms:
    if seg_feature.shape[-1]==2:
        X = np.zeros((5, fold_size, 60, seg_length, 2))
        for fold in folds:
            indexes = np.argwhere([seg_folds[j]==fold for j in range(len(seg_folds))])
            for k,ind in enumerate(indexes):
                X[fold-1,k,:,:,:] = seg_feature[ind,:,:,:]
                Y[fold-1,k,:] = seg_labels[ind,:]
    
    # For MFCCs:
    elif seg_feature.shape[-1]==13:
        X = np.zeros((5, fold_size, seg_length, 13))
        for fold in folds:
            indexes = np.argwhere([seg_folds[j]==fold for j in range(len(seg_folds))])
            for k,ind in enumerate(indexes):
                X[fold-1,k,:,:] = seg_feature[ind,:,:]
                Y[fold-1,k,:] = seg_labels[ind,:]
    else:
        raise ValueError('Shape of feature segment array does not match expected shape.')
    
    # Also divide the clips_id, useful for validation
    clip_id_folds = np.zeros((5, fold_size))
    for fold in folds:
        indexes = np.argwhere([seg_folds[j]==fold for j in range(len(seg_folds))])
        for k,ind in enumerate(indexes):
                clip_id_folds[fold-1,k] = clip_id[ind]
    return(X,Y,clip_id_folds)




######################################################################################################################
#                                        Block 2: Preprocessing pipeline function 
######################################################################################################################     

def preprocess_pipeline(mode, architecture, name_file):
    '''
    Preprocess the dataset by sequentially applying the above functions.
    Parameters:
        - mode(str): 'train' or 'test';
        - architecture(str): 'multibranch' or 'sequential';
        - name_file(str): list of folds
    Returns:
        if mode=='train':
            - X_cnn, X_rnn (np.arrays): inputs for CNN and RNN organized into folds (only X_cnn for architecture='sequential');
            - Y (np.array): one-hot segment labels organized into folds;
            - folds (list): list of segments' folds.
            - clip_id (np.array): array of clips ID for each segment, identifying to which Clip it belongs to;
            - y_clips (np.array): one-hot Clips labels.
        if mode=='test':
            - cnn2d_seg, rnn_seg (np.arrays): inputs for CNN and RNN, not organized into folds (only cnn2d_seg if architecture='sequential');
            - seg_labels (np.array): one-hot segment labels, not organized into folds;
            - folds_or_id (np.array): array of clips ID for each segment, identifying to which Clip it belongs to;
            - y_clips (np.array): one-hot Clips labels.
    '''
    
    # Load Clips and create one-hot labels
    if name_file.endswith('.pkl'):
        with open(name_file, "rb") as fp:   # Unpickling
            clips = pickle.load(fp)
        y_clips = create_labels_array(clips)
    else:
        raise TypeError('Pickle file expected as input.')

    
    # Final data (split into folds if mode==train, which is the same also used for validation)
    if mode=='train':
        # Segmentation
        cnn2d_seg, rnn_seg, seg_labels, folds, clip_id = segmentation(clips, y_clips, architecture, mode)

        if architecture=='multibranch':
            # Get segments for CNN and RNN branch
            X_cnn, Y, clip_id_folds = divide_folds(cnn2d_seg, seg_labels, clip_id, folds)
            X_rnn, _ , _ = divide_folds(rnn_seg, seg_labels, clip_id, folds)
            return(X_cnn, X_rnn, Y, folds, clip_id_folds, y_clips)
        elif architecture=='sequential':
            # Get segments only for CNN
            X_cnn, Y, clip_id_folds  = divide_folds(cnn2d_seg, seg_labels, clip_id, folds)
            return(X_cnn, Y, folds, clip_id_folds, y_clips)
        else:
            raise ValueError('Architecture not recognized.')
        
    elif mode=='test':
        # Segmentation
        cnn2d_seg, rnn_seg, seg_labels, clip_id = segmentation(clips, y_clips, architecture, mode)
        
        if architecture=='multibranch':
            return(cnn2d_seg, rnn_seg, seg_labels, clip_id, y_clips)
        elif architecture=='sequential':
            return(cnn2d_seg, seg_labels, clip_id, y_clips)
        else:
            raise ValueError('Architecture not recognized.')
    else:
            raise ValueError('Mode not recognized.')
        

