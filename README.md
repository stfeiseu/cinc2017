# Example Python classifier for the PhysioNet/CinC Challenge 2017

## Contents

This classifier uses three scripts:

* `util.py` contains two functions named `load` and `save`, which are used to load and save processing as a binary file.

* `load.py` contains the processing function of data before input to model.

* `challenge.py` contains `load_ECG_model` and `run_ECG_classifier` functions. It calls another two scripts. In `challenge.py`,    `load_ECG_model` was called once to load model and `run_ECG_classifier` was called many times to predict for all files. This script output a `answer.txt` file to save the predict result. 

## Use

Here are instructions for testing the code in Linux.  
 First, create a folder, `docker_test`, in your home directory. Then, put the code in docker_test/cinc2017_test. Finally, build a Docker image and run the example code using the following steps:  

 First, switch the current directory to the `~/docker_test/cinc2017_test`.   

 Then, execute the following command to build a docker image. 

    sudo docker build -t image .

    sudo docker run -it -v ~/docker_test/validation:/physionet/input_directory -v ~/docker_test/cinc2017_test:/physionet/output_directory image bash

**Note:** In the last command, the directory (`~/docker_test/validation`) followed by the first `-v` is the directory where the dataset is located on my computer, you can change `~/docker_test/validation` to the directory of data set on your computer.   

At last, you can run this classifier by running

    python challenge.py input_directory output_directory

where `input_directory` is a directory for input data files and `output_directory` is a directory for output classification files.    

After completing the above step, a `answer.txt` file will be generated in the folder `~/docker_test/cinc2017_test/output_directory`. 

The PhysioNet/CinC 2017 webpage (https://physionet.org/content/challenge-2017/1.0.0/) provides a training database with data files
and a description of the contents and structure of these files.

**How do I install Docker?**  
Go to https://docs.docker.com/install/ and install the Docker Community Edition. For troubleshooting, see https://docs.docker.com/config/daemon/


**Reference**: This file refers to https://physionetchallenges.github.io/2020/submissions#docker 