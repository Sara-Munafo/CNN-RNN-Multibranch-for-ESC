import os
import shutil
import urllib
import zipfile
import glob
import pandas as pd
import soundfile as sf
import IPython.display
import librosa
import numpy as np
import pydub

#####################################################################################################################################
#                                              Block 0: Define Clips class to handle data
#####################################################################################################################################
# Class to compute features from raw audio files.
# The following class was implemented by Piczak in the provided ipynb. 
# It was slightly change to account for what was needed for my implementation (e.g. adding MFCC computation).
class Clip:
    """A single 5-sec long recording."""
    
    RATE = 22050   # All recordings in ESC are 44.1 kHz, will be resample to RATE
    FRAME = 512    # Frame size in samples
    
    class Audio:
        """The actual audio data of the clip.
        
            Uses a context manager to load/unload the raw audio data. This way clips
            can be processed sequentially with reasonable memory usage.
        """
        
        def __init__(self, path):
            self.path = path
        
        def __enter__(self):
            # Load audio file and resample to the target RATE
            y, sr = librosa.load(self.path, sr=None)  # Load with original sampling rate
            y_resampled = librosa.resample(y, orig_sr=sr, target_sr=Clip.RATE)  # Resample
            
            # Trim or pad to exactly 5 seconds after resampling
            target_length = int(Clip.RATE * 5)  # 5 seconds at target sampling rate
            if len(y_resampled) > target_length:
                y_resampled = y_resampled[:target_length]
            elif len(y_resampled) < target_length:
                y_resampled = np.pad(y_resampled, (0, target_length - len(y_resampled)))
            
            # Store raw audio data and Pydub segment
            self.raw = y_resampled
            self.data = pydub.AudioSegment(
                (y_resampled * (0x7FFF + 0.5)).astype("int16").tobytes(),
                frame_rate=Clip.RATE,
                sample_width=2,  # 16-bit PCM
                channels=1
            )
            return self
        
        def __exit__(self, exception_type, exception_value, traceback):
            if exception_type is not None:
                print(exception_type, exception_value, traceback)
            del self.data
            del self.raw

        
    def __init__(self, filename):
        self.filename = os.path.basename(filename)
        self.path = os.path.abspath(filename)        
        self.directory = os.path.dirname(self.path)
        self.category = self.directory.split('/')[-1]
        self.fold = self.filename.split('-')[0]
        
        self.audio = Clip.Audio(self.path)
        
        with self.audio as audio:
            self._compute_features(audio)    
            
    def _compute_features(self, audio):
        # Melspec computation with 1024 FFT window length, 512 hop length, 60 bands
        self.melspectrogram = librosa.feature.melspectrogram(y=audio.raw, sr=Clip.RATE, hop_length=Clip.FRAME, n_mels=60)
        # Log scaling of melspec
        self.logamplitude = librosa.power_to_db(self.melspectrogram) 
        # MFCC computation
        self.mfcc = librosa.feature.mfcc(S=self.logamplitude, n_mfcc=13).transpose() 
        # Delta of log-melspec
        self.delta = librosa.feature.delta(self.logamplitude)  
            

            
    @classmethod
    def _get_frame(cls, audio, index):
        if index < 0:
            return None
        return audio.raw[(index * Clip.FRAME):(index+1) * Clip.FRAME]
    
    def __repr__(self):
        return '<{0}/{1}>'.format(self.category, self.filename)








#####################################################################################################################################
#                                              Block 1: Download and organize dataset 
#####################################################################################################################################

def download_dataset(name):
    '''Downloads the original ESC dataset from the github repository.

    Parameters:
        - name(str): name of the original repository
    Raturns: 
        - saves Esc-dataset directory: containing metadata, audio files'''
    
    if not os.path.exists(name):
        os.mkdir(name)
        # Download repository
        urllib.request.urlretrieve('https://github.com/karoldvl/{0}/archive/master.zip'.format(name), '{0}/{0}.zip'.format(name))
        #print("Download successful")
    
        # Extract content and delete .zip file
        with zipfile.ZipFile('{0}/{0}.zip'.format(name)) as package:
            package.extractall('{0}/'.format(name))
        #print("Extraction successful")
        os.unlink('{0}/{0}.zip'.format(name))
        #print("ZIP file deleted")
        
        # Move files to selected directory
        for src in glob.glob('{0}/{0}-master/*'.format(name)):
            shutil.move(src, '{0}/{1}'.format(name, os.path.basename(src)))
        os.rmdir('{0}/{0}-master'.format(name))
        #print("File reorganization complete")
        os.rename(name, 'ESC-dataset')



def organize_dataset(input_dir, dataset):
    """
    Organize the dataset into subfolders by class.

    Parameters:
        - input_dir(str): original name of the directory where files are stored;
        - dataset(str): "esc10" or "esc50", specifies what dataset we are working on.
    Returns:
        - 'dataset' directory, containing the audio files organized in sub-folders based on their label.
    """

    # Load metadata
    metadata_path = os.path.join(input_dir,"meta/esc50.csv")
    metadata = pd.read_csv(metadata_path)

    if dataset=='esc10':
        esc10_classes = []
        for _,row in metadata.iterrows():
            if row['esc10']==True:
                esc10_classes.append(row['category'])
        esc10_classes = list(set(esc10_classes))

        # Filter metadata for ESC-10 classes
        metadata = metadata[metadata['category'].isin(esc10_classes)]

    # Create subfolders for each class
    for label in metadata['category'].unique():
        class_dir = os.path.join(dataset, label)
        os.makedirs(class_dir, exist_ok=True)

    # Move audio files into respective class folders
    for _, row in metadata.iterrows():
        src = os.path.join(input_dir, "audio", row['filename'])
        dst = os.path.join(dataset, row['category'], row['filename'])
        shutil.copy(src, dst)
        



def load_dataset(name):
    """
    Load all dataset recordings into a  list.

    Parameters:
        - name(str): name of the folder we want to load (either 'esc10' or 'esc50')
    Returns:
        - clips(list) : nested list of loaded Clip objects
    """

    clips = []

    # For each directory in the 'name' folder, selects .wav files and appends them as Clip objects to clips list
    for directory in sorted(os.listdir('{0}/'.format(name))):
        directory = '{0}/{1}'.format(name, directory)
        if os.path.isdir(directory):
            for clip in sorted(os.listdir(directory)):
                if clip[-3:] == 'wav':
                    clips.append(Clip('{0}/{1}'.format(directory, clip)))
            
    IPython.display.clear_output()
    print('All {0} recordings loaded.'.format(name))            
    
    return clips





#####################################################################################################################################
#                                              Block 2: split train/test and augment training data
#####################################################################################################################################


def create_test_dataset(name, test_dir):
    """
    Load all dataset recordings into a list and hold out files for testing.

    Parameters:
        - name (str): Name of the folder to load (either 'esc10' or 'esc50').
        - test_dir (str): Directory to move held-out test files to.
        - num_folds (int): Number of folds in the dataset (default: 5).
    Returns:
        - clips (list): List of loaded Clip objects for training/validation.
        - test_clips (list): List of loaded Clip objects for testing.
    """
    # Ensure the test directory exists
    os.makedirs(test_dir, exist_ok=True)

    test_clips = []
    num_folds = 5
    fold_files = {f"fold{fold}": [] for fold in range(1, num_folds + 1)}

    for category in sorted(os.listdir(f'{name}/')):
        category_dir = f'{name}/{category}'
        if os.path.isdir(category_dir):
            print(f'Processing category: {category}')

            # Collect files by fold
            for clip in sorted(os.listdir(category_dir)):
                if clip.endswith('.wav'):
                    fold = int(clip.split('-')[0])  # Extract fold number from file name
                    fold_files[f"fold{fold}"].append(f'{category_dir}/{clip}')

            # Hold out two files per fold for the test set
            for fold in range(1, num_folds + 1):
                if fold_files[f"fold{fold}"]:
                # Select two random files from the fold
                    for _ in range(2):  # Select two files
                        r = np.random.randint(0, len(fold_files[f"fold{fold}"]))
                        test_file = fold_files[f"fold{fold}"].pop(r)  # Get a random file from the fold
                        test_category_dir = os.path.join(test_dir, category)
                        os.makedirs(test_category_dir, exist_ok=True)  # Create category folder in test directory
                        dest = os.path.join(test_category_dir, os.path.basename(test_file))
                        shutil.move(test_file, dest)  # Move file to the test directory
                        test_clips.append(Clip(dest))  # Load it as a Clip for the test set


            # Clear fold files for the next category
            fold_files = {f"fold{fold}": [] for fold in range(1, num_folds + 1)}

    IPython.display.clear_output()
    #print(f'All {name} recordings loaded.')
    print(f'Test files moved to: {test_dir}')




def reshape_clip(augmented, original):
    """
    Reshape augmented waveforms to match original clips' length.
    
    Parameters:
        - augmented (numpy.ndarray): The augmented audio waveform.
        - original (Clip): The original Clip object.
    Returns:
        - augmented (Clip): the reshaped augmented sample
    """
    target_shape = original.RATE*5
    aug_shape = augmented.shape[0]

    if aug_shape>target_shape:
        augmented = augmented[:target_shape]  # Cut
    elif aug_shape<target_shape:
        augmented = np.pad(augmented, (0, target_shape - aug_shape))  # Pad
    return(augmented)



def save_augmented_clip(augmented_audio, original_clip, output_dir, suffix):
    """
    Save an augmented audio sample as a new WAV file.
    
    Parameters:
        - augmented_audio (numpy.ndarray): The augmented audio waveform.
        - original_clip (Clip): The original Clip object.
        - output_dir (str): Directory to save the augmented file.
        - suffix (str): Suffix to append to the original filename for the augmented sample.
    Returns:
        - output_path (str): Path of the saved augmented clip
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Construct the new filename
    augmented_filename = f"{os.path.splitext(original_clip.filename)[0]}_{suffix}.wav"
    output_path = os.path.join(output_dir, augmented_filename)
    
    # Save the augmented audio
    sf.write(output_path, augmented_audio, Clip.RATE)
    
    return output_path



def augment_dataset(dataset, is_10):
    """
    Augment the selected dataset through time shift, time delay and pitch shift; 
    saves the augmented samples in the original directory.
    
    Parameters:
        - dataset (list): The list of original Clip objects.
    
    """
    
    for clip in dataset:

        if is_10==True:
            #Number of pitch shifting augmentations and shifting steps (in semitones)
            n_shift = 4
            steps = [-2, -1, 1, 2]

            # Number of time stretching augmentations and rates
            n_stretch = 2
            stretch = [0.9, 1.1]

            #Number of time delaying augmentations and the (random) time delays
            n_delay = 4
            t_delay = np.random.randint(-5000,5000,n_delay)
        else:
            #Number of pitch shifting augmentations and shifting steps (in semitones)
            n_shift = 1
            steps = [2]

            # Number of time stretching augmentations and rates
            n_stretch = 1
            stretch = [0.9]

            #Number of time delaying augmentations and the (random) time delays
            n_delay = 2
            t_delay = np.random.randint(-5000,5000,n_delay)


        with clip.audio as audio:

            for i in range(n_shift):
                shifted_clip = librosa.effects.pitch_shift(y=audio.raw, sr =22050,  n_steps=steps[i])
                shifted_clip = reshape_clip(shifted_clip, clip)
                suffix = 'aug_shift_{0}'.format(i)
                save_augmented_clip(shifted_clip, clip, clip.directory, suffix)

            for j in range(n_stretch):
                stretched_clip = librosa.effects.time_stretch(y=audio.raw, rate=stretch[j])
                stretched_clip = reshape_clip(stretched_clip, clip)
                suffix = 'aug_stretch_{0}'.format(j)
                save_augmented_clip(stretched_clip, clip, clip.directory, suffix)

            for k in range(n_delay):
                delayed_clip = np.roll(audio.raw,t_delay[k])
                delayed_clip = reshape_clip(delayed_clip, clip)
                suffix = 'aug_delay_{0}'.format(k)
                save_augmented_clip(delayed_clip, clip, clip.directory, suffix)