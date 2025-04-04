import tensorflow as tf
import time
import numpy as np



#########################################################################################################################
#                                             Block 1: Training functions
#########################################################################################################################


def save_metrics(history, trainable_params, name_file, train_time):
  '''
  Saves training metrics of model.

  Parameters:
    - history: training history of the model
    - trainable params: trainable parameters of the model
    - name_file(str): name of the file in which to save results
    - train_time(float64): time of training (in seconds)
   '''
  # Extract training history
  train_loss = history.history['loss']
  val_loss = history.history['val_loss']
  train_acc = history.history['accuracy']
  val_acc = history.history['val_accuracy']


  # Save as a NumPy file with the metrics of the model
  name_file = f"{name_file}_metrics.npz"
  np.savez(name_file, train_loss=train_loss, val_loss=val_loss, train_acc=train_acc, val_acc=val_acc)

  # Save to a text file
  name_txt = f"{name_file}_complexity.txt"
  with open(name_txt, "w") as f:
      f.write(f"Trainable Parameters: {trainable_params}\n")
      f.write(f"Training time: {train_time}\n")





def train_sequential(model_name, input_shape, train_dataset, val_dataset, num_epochs, n_units, save_model):
  '''
  Train sequential model.

  Parameters:
    - model_name(function): name of the model function;
    - input_shape(tuple): shape of input;
    - train_dataset(tf.data.Dataset): training dataset;
    - val_dataset(tf.data.Dataset): validation dataset;
    - num_epochs(int): training epochs;
    - n_units(int): number of GRU units;
    - save_model(bool): boolean to decide whether to save the model;
   '''

  # Set lr scheduler
  lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_accuracy',  
    factor=0.5,          
    patience=5,         
    min_lr=1e-6,         
    min_delta=0.001)


  optimizer = tf.keras.optimizers.SGD(learning_rate=0.004, momentum=0.9, nesterov=True)
  model = model_name(input_shape, n_units)  
 
  model.compile(optimizer=optimizer, loss="categorical_crossentropy",  metrics=["accuracy"])
  trainable_params = sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])

  print("Training model with {0} GRU units.  Trainable parameters: {1}".format(n_units, trainable_params))

  start_time = time.time()
  history = model.fit(train_dataset,
                  epochs=num_epochs,
                  validation_data=val_dataset,
                  callbacks=[lr_scheduler])
  end_time = time.time()
  train_time = end_time-start_time
  print("Training time: {0}".format(train_time))
  
  #Keep track of: losses, accuracy, complexity in time and memory
  name_file = f"{model_name.__name__}_{n_units}"
  save_metrics(history, trainable_params, name_file, train_time)

  if save_model==True:
    model.save(f"{name_file}.h5")

  del model
  tf.keras.backend.clear_session()





def train_multibranch(model_name, input_shape_cnn, input_shape_rnn, train_dataset, val_dataset, num_epochs, n_units, bidirectional, dense_neurons, save_model):
  '''
  Train multibranch model.

  Parameters:
    - model_name(function): name of the model function;
    - input_shape_cnn(tuple): shape of CNN input;
    - input_shape_rnn(tuple): shape of RNN input;
    - train_dataset(tf.data.Dataset): training dataset;
    - val_dataset(tf.data.Dataset): validation dataset;
    - num_epochs(int): training epochs;
    - n_units(int): number of GRU units;
    - bidirectional(bool): determines if the GRU layer is bidirectional;
    - dense_neurons(int): number of neurons in Dense layers;
    - save_model(bool): boolean to decide whether to save the model;
   '''

  # Set lr scheduler
  lr_scheduler = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_accuracy',  
    factor=0.5,          
    patience=5,          
    min_lr=1e-6,         
    min_delta=0.001)


  optimizer = tf.keras.optimizers.SGD(learning_rate=0.004, momentum=0.9, nesterov=True)
  model = model_name(input_shape_cnn, input_shape_rnn, n_units, bidirectional, dense_neurons)  #Try num_layers layers, with and without GRU

  #tf.config.run_functions_eagerly(True)
  model.compile(optimizer=optimizer, loss="categorical_crossentropy",  metrics=["accuracy"])
  trainable_params = sum([tf.keras.backend.count_params(w) for w in model.trainable_weights])

  print("Training model with {0} GRU units, {1} Dense hidden units, bidirectional = {2}.  Trainable parameters: {3}".format(n_units, dense_neurons, bidirectional, trainable_params))

  start_time = time.time()
  history = model.fit(train_dataset,
                  epochs=num_epochs,
                  validation_data=val_dataset,
                  callbacks=[lr_scheduler])
  end_time = time.time()
  train_time = end_time-start_time
  print("Training time: {0}".format(train_time))

  #Keep track of: losses, accuracy, complexity in time and memory
  name_file = f"{model_name.__name__}_{n_units}_{dense_neurons}_{str(bidirectional)[0]}"
  save_metrics(history, trainable_params, name_file, train_time)

  if save_model==True:
    model.save(f"{name_file}.h5")

  del model
  tf.keras.backend.clear_session()







#########################################################################################################################
#                                             Block 1: Testing functions
#########################################################################################################################


def aggregate_predictions(predictions, clip_id):
    """
    Aggregate segment predictions by averaging probabilities for each full clip.

    Parameters:
      - predictions (np.array): array of probabilities predicted by the model;
      - clip_id (list): list of ids (for each segment, id of the clip it belongs to)
    Returns:
      - final_preds (np.array): array of final predictions, obtained by averaging predictions over all segments of the same clip.
    """
    unique_clips = np.unique(clip_id)  # Get unique clip identifiers
    final_preds = []

    for clip in unique_clips:
        clip_indices = np.where(clip_id == clip)[0]  # Find all segments of the clip
        clip_probs = predictions[clip_indices]  # Extract probabilities of those segments
        avg_probs = np.mean(clip_probs, axis=0)  # Average over all segments
        final_preds.append(np.argmax(avg_probs))  # Take the class with highest probability

    return np.array(final_preds)




def accuracy_per_class(Y_true, Y_pred):
  """
    Compute accuracy per single class.

    Parameters:
      - Y_true (np.array): array of not one-hot encoded true labels (integers from 0 to 49);
      - Y_pred (np.array): array of not one-hot encoded predicted labels (integers from 0 to 49)
    Returns:
      - class accuracy (np.array): array of per-class accuracies (shape (50,))
    """
  # Initialize counters
  correct_per_class = np.zeros(50)
  clips_per_class = int(len(Y_true)/50)
  # Loop through all predictions
  for true_label, pred_label in zip(Y_true, Y_pred):
      if true_label == pred_label:
          correct_per_class[true_label] += 1  # Count correct predictions

  # Compute per-class accuracy
  class_accuracy = (correct_per_class / clips_per_class) * 100  
  return class_accuracy
