# EfficientParsing
Semantic parser for spreadsheet applications which is able to generalize new instances by decomposition.   
Main project of a bachelors thesis in computer science. 

## Requirements:
- Python 3.10
- Ubuntu or Windows Subsystem for Linux

## Cloning the project:
To clone this project you require Git LFS. 
An installation guide can be found [here](https://github.com/git-lfs/git-lfs/wiki/Installation).

## Installation:
To use this project, we recommend creating a python virtual environment.  
To create the environment copy the following inside the terminal in the project base folder ```./setup_venv.sh```. 
The script will ask for your ```sudo``` password and should install all required python modules as well as 
do the required setup work for you. 

## Usage:
Under ```notebooks```, you can experiment with components of the semantic parser and see usage examples.  
Furthermore, you can create plots which contain information about the quality of certain aspects of the semantic parser.  
  
In addition, you can use ```do_measurements.py``` in ```efficient_parsing``` to compute the measurements done in the 
[bachelors thesis](thesis.pdf).  You can execute a single measurement or all at once. 
Refer to the help text of the script for further information.