#Author: Srinath Sibi, Email: ssibi@stanford.edu
#Purpose: To plot the markers and see how we can change the event markers
import glob, os, sys, shutil,re
import matplotlib as plt
import numpy as np
import csv
import matplotlib.pyplot as plt
import math
from scipy import interpolate
import numpy as np
import pandas as pd
from statistics import mean
LOGFILE = os.path.abspath('.') + '/CurrentOutputFile.csv'
MAINPATH = os.path.abspath('.')#Always specify absolute path for all path specification and file specifications
LISTOFPARTICIPANTS =[]#This is the list of all participants in the Data folder
LISTOFSECTIONS =[]# List of all sections for a participant
DEBUG = 0# To print statements for debugging
