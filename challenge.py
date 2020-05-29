#!/usr/bin/python3
# Example challenge entry

import sys
import os 
import scipy.io
import numpy as np
import keras
import util

def run_ECG_classifier(data, model, preproc):

    # Use your classifier here to obtain a label.
    
    # Prior is the proportion of samples of three categories (AF, Normal, Other) in the training set
    # For Noisy, we set it's prior as 1.0. 
    # The order of categories is AF, Normal, Other, Noisy
    prior = np.array([0.0908, 0.6111, 0.2981, 1.0])
    
    data = data.squeeze()
    if data.shape[0] > 18000:
        data = data[:18000]

    x = preproc.process_x([data])
    probs = model.predict(x)
    label = np.argmax(probs/prior, axis=1)
    return label

def load_ECG_model():
    # load the model from disk
    filename='./model/classify_model.hdf5'
    preproc = util.load(os.path.dirname(filename))
    loaded_model = keras.models.load_model(filename)

    return loaded_model, preproc


if __name__ == '__main__':

    # Parse arguments.
    if len(sys.argv) != 3:
        raise Exception('Include the input and output directories as arguments, e.g., python challenge.py input output.')

    input_directory = sys.argv[1]
    output_directory = sys.argv[2]

    # input_directory = 'E:/Physionet2017_seq/dev'
    # output_directory = 'dev_result'

    # Find input files.
    input_files = []
    for f in os.listdir(input_directory):
        if os.path.isfile(os.path.join(input_directory, f)) and not f.lower().startswith('.') and f.lower().endswith('mat'):
            input_files.append(f)
    
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    
    # remove answer.txt before prediction.
    if os.path.exists(os.path.join(output_directory, "answers.txt")):
        os.remove(os.path.join(output_directory, "answers.txt"))
        
    # Load model.
    print('Loading ECG model...')
    model, preproc = load_ECG_model()

    # Iterate over files.
    num_files = len(input_files)

    for i, f in enumerate(input_files):
        print(' {}/{}...'.format(i+1, num_files))
        tmp_input_file = os.path.join(input_directory,f)
        # Read waveform samples (input is in WFDB-MAT format)
        record = f.split('.')[0]
        mat_data = scipy.io.loadmat(tmp_input_file)
        samples = mat_data['val']
        # Your classification algorithm goes here...
        if samples[0][0] < 0:
            answer = "N"
        else:
            answer = "A"
        
        answer = run_ECG_classifier(samples, model, preproc)
        answer = preproc.int_to_class[answer[0]]
        # Save results.
        # Write result to answers.txt
        answers_file = open(os.path.join(output_directory, "answers.txt"), "a")
        answers_file.write("%s,%s\n" % (record, answer))
        answers_file.close()

    print('Done.')
