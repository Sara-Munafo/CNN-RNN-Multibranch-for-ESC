import os
import librosa
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
from data_helper import Clip
from collections import defaultdict
import pickle



#####################################################################################################################################
#                                              Block 0: Data visualization functions
#####################################################################################################################################
# The first two functions from this block were implemented by Piczak in the ipynb related to their ESC implementation.
# They were slightly modified accordingly to what I needed to display.



def add_subplot_axes(ax, position):
    """
    Add a subplot within an existing axis.

    Parameters:
        - ax (matplotlib.axes.Axes): The parent axis where the subplot will be added.
        - position (list): A list of four values [x, y, width, height] defining the 
          relative position and size of the subplot within the parent axis.

    Returns:
        - matplotlib.axes.Axes: The newly created subplot axis.
    """
    box = ax.get_position()
    position_display = ax.transAxes.transform(position[0:2])
    position_fig = plt.gcf().transFigure.inverted().transform(position_display)
    x = position_fig[0]
    y = position_fig[1]

    return plt.gcf().add_axes([x, y, box.width * position[2], box.height * position[3]])  # axisbg='w')


def plot_clip_overview(clip, ax, feature):
    """
    Plot an overview of an audio clip, including its waveform and spectrogram.

    Parameters:
        - clip (Clip object): The audio clip object containing waveform and spectrogram data.
        - ax (matplotlib.axes.Axes): The parent axis where the overview will be plotted.
        - feature (str): The feature type to display in the spectrogram 
          ('melspec' for mel spectrogram, anything else for MFCC).

    Returns:
        - None: Displays the waveform and spectrogram on the given axis.
    """
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax_waveform = add_subplot_axes(ax, [0.0, 0.7, 1.0, 0.3])
    ax_spectrogram = add_subplot_axes(ax, [0.0, 0.0, 1.0, 0.7])

    with clip.audio as audio:
        ax_waveform.plot(np.arange(0, len(audio.raw)) / float(Clip.RATE), audio.raw)
        ax_waveform.get_xaxis().set_visible(False)
        ax_waveform.get_yaxis().set_visible(False)
        ax_waveform.set_title('{0} \n {1}'.format(clip.category, clip.filename), 
                              {'fontsize': 8}, y=1.03)

        if feature == 'melspec':
            librosa.display.specshow(librosa.power_to_db(clip.melspectrogram), sr=Clip.RATE, 
                                     x_axis='time', y_axis='mel', cmap='RdBu_r')
        else:
            librosa.display.specshow(clip.mfcc.T, x_axis="time", sr=Clip.RATE, cmap="plasma")
            plt.ylabel("MFCC Coeff.")
            plt.yticks(np.arange(clip.mfcc.shape[1]), labels=np.arange(clip.mfcc.shape[1]))

        ax_spectrogram.get_xaxis().set_visible(False)
        ax_spectrogram.get_yaxis().set_visible(False)


def clips_overview(feature, categories, clips_shown, clips_list):
    """
    Generate an overview of multiple audio clips, displaying their waveforms and spectrograms.

    Parameters:
        - feature (str): The feature type to display in the spectrogram ('melspec' or 'mfcc').
        - categories (int): The number of categories (rows) to display.
        - clips_shown (int): The number of clips (columns) to show per category.
        - clips_list (list): List of clip objects from which the displayed clips will be selected.

    Returns:
        - None: Displays the overview of the selected clips in a grid layout.
    """
    f, axes = plt.subplots(categories, clips_shown, 
                           figsize=(clips_shown * 2, categories * 2), 
                           sharex=True, sharey=True)
    f.subplots_adjust(hspace=0.35)

    for c in range(categories):
        clips_toplot = []
        categories_name = os.listdir('esc50')

        for clip in clips_list:
            if clip.category == categories_name[c]:
                clips_toplot.append(clip)
                if len(clips_toplot) == clips_shown:
                    break

        for i in range(clips_shown):
            plot_clip_overview(clips_toplot[i], axes[c, i], feature)





#####################################################################################################################################
#                                              Block 1: Augmented data visualization functions
#####################################################################################################################################



def create_augmented_dict(clips_list):
    """
    Create a dictionary to store augmented clips by original clip name and type of augmentation.

    Parameters:
        - clips_list (list): list of (augmented+original) Clips objects
    Returns:
        - augmented_clips_dict (dict): dictionary of organized augmented clips
    """
    # Create a dictionary to store augmented clips by original clip name and type
    augmented_clips_dict = defaultdict(lambda: {"shift": [], "delay": [], "stretch": []})

    # Populate the dictionary
    for clip in clips_list:
        # Extract the base name without the augmentation suffix
        base_name = clip.filename.split('_aug_')[0]

        # Determine the augmentation type (shift, del, stretch)
        if '_aug_shift_' in clip.filename:
            augmented_clips_dict[base_name]["shift"].append(clip)
        elif '_aug_delay_' in clip.filename:
            augmented_clips_dict[base_name]["delay"].append(clip)
        elif '_aug_stretch_' in clip.filename:
            augmented_clips_dict[base_name]["stretch"].append(clip)
    return augmented_clips_dict


def get_augmented_names(original_clip_name, augmented_clips_dict):
    """
    Retrieve one augmented clip of each type for a given original clip name.

    Parameters:
        - original_clip_name (str): The name of the original clip.
        - augmented_clips_dict (dict): Dictionary of augmented clips.

    Returns:
        - dict: A dictionary with one clip per augmentation type (or None if not found).
    """
    if original_clip_name in augmented_clips_dict:
        return {
            "shift": augmented_clips_dict[original_clip_name]["shift"] if augmented_clips_dict[original_clip_name]["shift"] else None,
            "delay": augmented_clips_dict[original_clip_name]["delay"] if augmented_clips_dict[original_clip_name]["delay"] else None,
            "stretch": augmented_clips_dict[original_clip_name]["stretch"] if augmented_clips_dict[original_clip_name]["stretch"] else None,
        }
    else:
        return {"shift": None, "delay": None, "stretch": None}
    


def plot_augmentations(clip, augmented_clips_dict, fig_size, colors):
    """
    Plot the raw (normalized) waveform of a Clip with its augmentations superimposed.

    Parameters:
        - clip (Clip object): clip to plot
        - augmented_clips_dict (dict): dictionary of augmented clips
        - fig_size (tuple): figsize for the plot
        - colors (list): colors for plot
    """
    original_clip_name = clip.filename.split('.wav')[0]
    # Get augmented samples from the original
    augmented_samples = get_augmented_names(original_clip_name, augmented_clips_dict)
    # Select one augmentation per type
    clips_to_plot = [augmented_samples["shift"][0], augmented_samples["delay"][0], augmented_samples["stretch"][0] ]

    fig, ax = plt.subplots(1, 3, figsize=fig_size)

    #plt.figure(figsize=(15,20))
    for i in range(3):
            clip_2 = clips_to_plot[i]
            with clip.audio as audio:
                with clip_2.audio as audio_2:
                    ax[i].plot(np.arange(0, len(audio.raw)) / 22050.0, audio.raw, color=colors[0], label = 'original')
                    ax[i].plot(np.arange(0, len(audio_2.raw)) / 22050.0, audio_2.raw, color=colors[1], label='augmented')
                    ax[i].set_title('{0} : {1} + {2}'.format(clip.category, clip.filename, clip_2.filename.split('_')[2]))
                    ax[i].set_xlabel('Time(s)')
                    ax[i].set_ylabel("Amplitude (normalized)")
                    ax[i].legend(loc="upper right")







#####################################################################################################################################
#                                              Block 2: Per class accuracy visualization function
#####################################################################################################################################


def plot_class_accuracy_2(model_names, class_accuracy, fig_size, sb_palette):
    """
    Plot the accuracy per class of the selected models.

    Parameters:
        - model_names (list): list of names of the models for plot title
        - class_accuracy (list): list of accuracy per class of the models
        - fig_size (tuple): figsize for the plot
        - sb_palette (str): palette for theplot
    """
    # Load labels
    with open('classes.pkl', 'rb') as c:
        classes = pickle.load(c)
    
    # Number of plots to make
    n = len(model_names)
    fig, ax = plt.subplots(n, 1, sharex=True, figsize=(fig_size[0], fig_size[1] * n))

    for i in range(0,n):
        sb.barplot(x=classes, y=class_accuracy[i], palette=sb_palette, ax=ax[i])
        ax[i].set_ylabel("Accuracy (%)", fontsize = 14)
        ax[i].grid(axis="y", linestyle="--", alpha=0.7)
        ax[i].set_title(model_names[i], fontsize=16)

    ax[-1].set_xlabel("Class", fontsize=14)  # Set x-label only for the last subplot
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()
