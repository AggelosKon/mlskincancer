#!/usr/bin/env python
# coding: utf-8


import pandas as pd
import numpy as np

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt # Used to plot images

from os import listdir # Used to access images folders
from os.path import isfile, join # Used to access images files
from tensorflow.keras import backend as K # Keras backed to de-allocate memory
from tensorflow.keras.callbacks import ReduceLROnPlateau # Reduce learning rate epoch by epoch
from keras.preprocessing.image import ImageDataGenerator # Used to populate images

# Stochastic Gradient Descent
from sklearn.linear_model import SGDClassifier
#Support Vector Machine
from sklearn.svm import SVC


def y_actual_labels(folder_path = 'augmented_dataset/test/test_images'):
    """ Returns the actual labels of the test set """
    y_actual = []
    test_counters = pd.DataFrame([[0,0]], columns=['benign', 'malignant'], index=['counter'])

    for f in listdir(folder_path):
        image_path = join(folder_path, f)
    
        if isfile(image_path):
            if f.startswith('benign'):
                y_actual.append('benign')
                test_counters['benign'] += 1
            elif f.startswith('malignant'):
                y_actual.append('malignant')
                test_counters['malignant'] += 1
                
    return y_actual, test_counters


# Confusion matrix function definition
def confusion_matrix(y_actual, y_pred):
    """ Creates a confusion matrix """
    conf_mat = pd.DataFrame([[0, 0], [0, 0]],                            columns=['true_benign', 'true_malignant'],                            index=['predicted_benign', 'predicted_malignant'])
    

    for i in range(len(y_pred)):
        conf_mat['true_' + y_actual[i]]['predicted_' + y_pred[i]] += 1
        
    return conf_mat


def classification_report(y_actual, y_pred):
    from sklearn.metrics import classification_report
    cr = classification_report(y_actual, y_pred, output_dict=True)

    cr['accuracy'] = {'f1-score': cr['accuracy']}
    for k, v in cr['macro avg'].items():
        if k == 'f1-score':
            continue
        elif k == 'support':
            cr['accuracy'][k] = v
        else:
            cr['accuracy'][k] = 'NA'
            
    return pd.DataFrame(cr).transpose()[['precision', 'recall', 'f1-score', 'support']]


def plot_model(train_history, validation_history, what = '', where = 'upper left'):
    """ Plots model's metrics from its trainning and validation """
    epochs = len(train_history)
    plt.plot([x+1 for x in range(epochs)], train_history)
    plt.plot([x+1 for x in range(epochs)], validation_history)
    plt.title('model ' + what)
    plt.ylabel(what)
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc = where)
    plt.show()


def run_nn(model, epochs = 10, batch_size = 64, rescale_factor = 1./255, lr = 1e-5, name = ''):
    """ Trains and evaluates a model """
    np.random.seed(123)
    result = {}
    # Set a learning rate annealer
    # This will reduce the learning rate of the ann to the half
    # of the learning rate of the previous epoch up to the lr/100 ((1e-5)/100 = 1e-7)
    learning_rate_reduction = ReduceLROnPlateau(monitor = 'accuracy', patience = 3,
                                                verbose = 1, factor = 0.5,
                                                min_lr = lr / 100)
    
    # Train the model
    train_datagen = ImageDataGenerator(rescale = rescale_factor)
    valid_datagen = ImageDataGenerator(rescale = rescale_factor)

    train_generator = train_datagen.flow_from_directory('augmented_dataset/train',
                                                        batch_size = batch_size,
                                                        class_mode = 'categorical')
    validation_generator = valid_datagen.flow_from_directory('augmented_dataset/valid',
                                                             batch_size = batch_size,
                                                             class_mode = 'categorical')

    # Number of batches in train set:
    STEP_SIZE_TRAIN = train_generator.n // train_generator.batch_size
    # Number of batches in validation set:
    STEP_SIZE_VALID = validation_generator.n // validation_generator.batch_size

    # Set a learning rate annealer
    # This will reduce the learning rate of the ann (depending on its accurcay) to the half
    # of the 'factor' * (learning rate of the previous epoch) up to the 'min_lr'
    # every 'patience' epochs.
    # 'Verbose = 1' means that we want to print a message whenever a lr reduction happens
    learning_rate_reduction = ReduceLROnPlateau(monitor = 'accuracy', patience = 5,
                                                verbose = 1, factor = 0.5,
                                                min_lr = lr / 100)
    
    # Fit the model
    print('Start training process for the ' + name + ' model...')
    history = model.fit(train_generator, steps_per_epoch = STEP_SIZE_TRAIN, epochs = epochs,
                        validation_data = validation_generator, validation_steps = STEP_SIZE_VALID,
                        callbacks = [learning_rate_reduction])
    result['history'] = history
    print(name + ' model training process was successful!')
    
    # Evaluate the model over validation set
    print('Start evaluating the ' + name + ' model over validation set...')
    metrics = model.evaluate(validation_generator, steps = STEP_SIZE_VALID)
    result['metrics'] = {k: v for k, v in zip(model.metrics_names, metrics)}
    print(name + ' model was evaluated on validation set successfully!')
    
    # Evaluate the model over test set
    print('Start evaluating the ' + name + ' model on test set...')
    test_datagen = ImageDataGenerator(rescale = rescale_factor)

    test_generator = test_datagen.flow_from_directory(directory = 'augmented_dataset/test',
                                                      batch_size = 1, color_mode="rgb",
                                                      class_mode = None,
                                                      shuffle = False,
                                                      seed = 42)

    STEP_SIZE_TEST = test_generator.n // test_generator.batch_size

    test_generator.reset()
    y_pred = model.predict(test_generator, steps = STEP_SIZE_TEST, verbose = 1)
    
    predicted_class_indices = np.argmax(y_pred, axis = 1)
    labels = (train_generator.class_indices)
    labels = dict((v,k) for k, v in labels.items())
    predictions = [labels[k] for k in predicted_class_indices]
    result['predictions'] = predictions
    print(name + ' model was evaluated on test set successfully!')
    
    result['model'] = model # The model is now trained
    
    return result





def run_not_nn(epochs = 10, input_dim=(256, 256, 3), batch_size = 64, alpha = 1e-5, rescale_factor = 1./255, name = ''):
    import time
    np.random.seed(123)
    
    train_datagen = ImageDataGenerator(rescale = 1./255)
    valid_datagen = ImageDataGenerator(rescale = 1./255)

    train_generator = train_datagen.flow_from_directory('augmented_dataset/train',
                                                        batch_size = batch_size,
                                                        class_mode = 'categorical')
    validation_generator = train_datagen.flow_from_directory('augmented_dataset/valid',
                                                             batch_size = batch_size,
                                                             class_mode = 'categorical')
    if name=='SGD' :
        clf = SGDClassifier(alpha = alpha)
    else:
        clf = SVC(random_state = 123, kernel = 'rbf')
        
   
    STEP_SIZE_TRAIN = train_generator.n // batch_size
    STEP_SIZE_VALID = validation_generator.n // validation_generator.batch_size
    flat_dim = np.prod(input_dim)
    history = []

    print('Start training process for the ' + name + ' model...')
     
    for e in range(1, epochs + 1):
        print('Epoch ' + str(e) + '/' + str(epochs) + ' is evaluating...', end='\r')
    
        batch_num = 0
        start = time.time()
        for x_batch, y_batch in train_generator:
            batch_num += 1
    
            batch_x = [x.reshape(flat_dim,) for x in x_batch]
            batch_y = [np.argmax(y) for y in y_batch]
        
            clf.fit(batch_x, batch_y)
    
            if batch_num > STEP_SIZE_TRAIN:
                break
        
        valid_predictions = 0
        predictions_made = 0
        validation_batch_num = 0
        
        for x_batch_validation, y_batch_validation in validation_generator:
            validation_batch_num += 1
            
            batch_x_validation = [x.reshape(np.prod(input_dim),) for x in x_batch_validation]
            batch_y_validation = [np.argmax(y) for y in y_batch_validation]
            
            epoch_predictions = clf.predict(batch_x_validation)
            
            for i in range(len(epoch_predictions)):
                predictions_made += 1
                if epoch_predictions[i] == batch_y_validation[i]:
                    valid_predictions += 1
                    
            if validation_batch_num > STEP_SIZE_VALID:
                break
                    
        history.append(valid_predictions / float(predictions_made))
                
        print('Epoch {}/{} ({:.2f} seconds)'.format(e, epochs, (time.time() - start)), end='\n')
    
     
    print('Training process for the ' + name + ' model was sucessful!')
    
    # Evaluate the model on test set
    print('Start evaluating the ' + name + ' model on test set...')
    test_datagen = ImageDataGenerator(rescale = 1./255)

    test_generator = test_datagen.flow_from_directory(directory = 'augmented_dataset/test',                                                      batch_size = 1, color_mode="rgb",                                                      class_mode = None,                                                      shuffle = False,                                                      seed = 42)

    STEP_SIZE_TEST = test_generator.n // test_generator.batch_size

    test_generator.reset()
    y_pred = []
    image_num = 0

    for x_batch in test_generator:
        image_num += 1
        batch_x = [x.reshape(flat_dim) for x in x_batch]
        y_pred.append(clf.predict(batch_x))

        if image_num % 1000 == 0:
            print(image_num, 'images processed')

        if image_num >= test_generator.n:
            break

    print(image_num, 'images processed')    
    print(name +' model was evaluated on test set successfully!')
        
    labels = (train_generator.class_indices)
    labels = dict((v,k) for k, v in labels.items())
    
    return {'predictions' : [labels[k] for k in [x[0] for x in y_pred]], 'history': history}


def tf_version():
    import tensorflow as tf
    return tf.__version__
