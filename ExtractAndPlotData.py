#Author: Srinath Sibi, Email: ssibi@stanford.edu
#Purpose: To plot the markers and see how we can change the event markers to something sane
#Overall Important columns:
# Time : Use column 10 and split and convert the time into seconds using the iMotionsTimeConverter converter. Also compare to TImestamp CAL (mSecs) in column 20
# Details: 1) UTC TImestamp -> Col 10 ; Timestamp -> Col 20
# Markers are different from sim markers.
# Details : 1) SimulatorEvent (0,0) -> Col 15
#First, we read and convert the input the columns that are of interest to us. We store them alongside the time and the markers in each file
#The following are the files that need to be made:
# 1. HR
# 2. GSR
# 3. PPG
# 4. Drive Data
#First the HR file.
# ECG Details: 1) For CAT 1: The ECG LL-RA CAL (mVolts) (ShimmerECG) data , ECG LL-RA RAW (no units) (ShimmerECG) and the ECG LA-RA CAL (mVolts) (ShimmerECG) columns work well.
# The corresponding columns are cols 45, 46 , 47 , 48.
# 2) For CAT 2: The ECG values are under the first two columns mislabelled as EMG and are present in Columns 37 , 38 , 39 , 40.
# 3) For CAT3 (participant 12): The ECG values are in the columns 32 , 33 , 34 , 35
# Drive Details: 1) Brake -> Col 14 ; Speed -> 16 ; Steer -> 17 ; Throttle -> 18
# PPG Details : 1) PPG heart beat-> Col 28 ; 2) IBI (msecs) -> Col 29
# GSR Details : 1) GSR CAL (micro siemens)-> Col 30
# Listed below are the events we are looking for :
# Events = [ 'GoAroundRocks' , 'CurvedRoads' , 'Failure1' , 'HighwayExit' , 'RightTurn1' , 'RightTurn2' , 'PedestrianEvent' , 'TurnRight3' , 'BicycleEvent' , 'TurnRight4' , 'TurnRight5'\
#, 'RoadObstructionEvent' , 'HighwayEntryEvent' , 'HighwayIslandEvent' ]
import os, sys, csv , re
import matplotlib as plt
import numpy as np
import csv
import matplotlib.pyplot as plt
import math
from scipy import interpolate
import numpy as np
import pandas as pd
import shutil
from statistics import mean
from PlottingFunctions import *
from biosppy.signals import ecg, eda
import biosppy
LOGFILE = os.path.abspath('.') + '/CurrentOutputFile.csv'#Using one output file for all the scripts from now to avoid flooding the working folder
MAINPATH = os.path.abspath('.')#Always specify absolute path for all path specification and file specifications
DEBUG = 0# To print statements for debugging
TOPELIMINATION = 100
READANDEXTRACTINDIVIDUALFILES = 0# This is the flag to make sure that files are read from the start
MAKEPLOTS = 0#Make the individual plots from the files of all the data streams
PROCESSMARKERS = 0#Analyze the markers and make abridged version of the markers file for event processing
REWRITEABRIDGEDMARKERFILE = 0#This Flag is to rewrite the marker file.
SEGMENTDATA = 0#This is to cut the data into windows for all the data in the study.
GSRSEGMENTATION = 0# This is the subsegment for GSR data segmentation in the SEGMENTDATA section
HRSEGMENTATION = 0# This is the subsegment for HR data segmentation in the SEGMENTDATA section
PPGSEGMENTATION = 0# This is the subsection for PPG data separation in the SEGMENTDATA section
DRIVESEGMENTATION = 1# This is the subsection for DRIVE sata separation in the SEGMENTDATA section
SIGNALPROCESSING = 1# This is the flag to signal the
HRPROCESSING = 1#This is to process the HR Data only
BIOSPPYVSHEARTPY = 0# This flag is to determine if we want to use biosppy or heartpy ( 0 - biosppy vs 1 - heartpy)
GSRPROCESSING = 0#This is to process the GSR Data only
PPGPROCESSING =0# This is to process the PPG Data only
BACKUPDATA = 0#This is to backup files that are important or are needed for later.
REMOVEFILE = 0# We are using this marker to remove a file with the same name across all participatns in similar locations.
#The 3 categories are defined here. Check here before adding the
#We use this function as a key to sorting the list of folders (participants).
CAT1 = [ 'P002', 'P004' , 'P005' , 'P007' , 'P008' , 'P009' , 'P010' , 'P011' , 'P013' , 'P016' , 'P021' , 'P023' , 'P024' , 'P025' , 'P028' , 'P032' ]
CAT2 = [ 'P022' , 'P037' , 'P040' , 'P042' , 'P045' , 'P046' , 'P047' , 'P055' , 'P057' , 'P058' , 'P060' , 'P064' , 'P066' , 'P067' , 'P069' , 'P071' , \
'P076' , 'P077' , 'P085' , 'P086' , 'P087' , 'P088' , 'P089' , 'P090' , 'P090' , 'P093' , 'P094' , 'P095' , 'P096' , 'P097' ,'P098']
CAT3 = ['P012']
# Window sizes are defined here :
GoAroundRocksWiNDOW = 20
CurvedRoadsWINDOW = 20
Failure1WINDOW = 20
HighwayExitWINDOW = 20
TURNWINDOW = 20
PedestrianEventWINDOW = 20
BicycleEventWINDOW = 20
RoadObstructionEventWINDOW = 20
HighwayEntryEventWINDOW = 20
HighwayIslandEventWINDOW = 20
# Function to return the index of the element nearest to a value
def find_nearest(array, value):
    ''' Find nearest value is an array '''
    idx = (np.abs(array-value)).argmin()
    return idx
#Function to sort the participant folder based on the number
def SortFunc(e):
    return int( re.sub('P','',e) )
#Function to estimate the sampling rate and plot it. it is not needed for later.
def CalculateSamplingRate(time, participant, section):
    if DEBUG ==1:   print " Function to estimate the sampling rate using the time list. "
    try:
        samplingrates =[]
        for i in range(len(time) - 1):
            try:
                samplingrates.append( (1 / (time[i+1] - time[i]) ) )
            except ZeroDivisionError:
                pass
        #Plotting the sampling rate to get a better picture
        #starting the plot
        fig = plt.figure()
        fig.tight_layout()
        plt.title('Raw Sampling Rate (Hz)')
        plt.plot(range(len(samplingrates[0:300])), samplingrates[0:300], 'r-', label =  'Sampling Rate', linewidth = 0.1)
        if DEBUG == 1:
            print "First few elements of the x and y data are : ", x_data[0:3] ,'\n', y_data[0:3]
        plt.xlabel('Index')
        plt.ylabel('Sampling Rate')
        plt.legend(loc = 'upper right')
        plt.savefig(MAINPATH+'/Data/'+participant+'/'+section+'/SamplingFrequency.pdf', bbox_inches = 'tight', dpi=900 , quality = 100)
        plt.close()
        if DEBUG==1:    print " Average sampling rate : " , mean(samplingrates)
        return mean(samplingrates)
    except Exception as e:
        print "Sampling rate calculation exception for participant: " , participant , ' in seciton: ' , section , 'Exception recorded: ' , e
        print 'Error on line {}'.format(sys.exc_info()[-1].tb_lineno)
        file = open(LOGFILE,'a')
        writer = csv.writer(file)
        writer.writerow(['Sampling rate calculation exception for participant: ', participant , 'in section' , section , 'Exception recorded: ', e , 'Error on line {}'.format(sys.exc_info()[-1].tb_lineno) ])
        file.close()
#Function to resample the data into the data with a target sampling frequency
def Resample( time, signal , participant , section , target_fs = 1000.0 , type = 'ecg' ):
    # NOTE: type is not used yet. But might be in the future when we add GSR and IBI data
    try:
        if DEBUG ==1:   print " The resampling function for " , participant , " section : " , section
        # First we need to eliminate the repeating data points
        time_1 = [time[0]]# Intermediate to final resampling
        signal_1= [signal[0]]
        for i in range(len(time)):
            if i >=1:#Making sure that we start the searching for repeat time points on the second index
                if time[i] == time[i-1]:
                    pass# Do not append the data to the new array if it's a repeat time step
                elif time[i] != time[i-1]:
                    time_1.append(time[i])
                    signal_1.append(signal[i])
        if DEBUG==1:    print "Raw Time sample : " , time[0:150] , "\n\n\nTime sample after repeat elimination: " , time_1[0:150]
        # The above section works
        # Before we make sure that we can samples in the data , we need to round to third decimal
        for i in range(len(time_1)):
            time_1[i] = round(time_1[i] , 3)
        # Now we create a linearly spaced time array between the start and the end points based on the sampling frequency privided
        N = ( 1 / target_fs )# Calculating number of samples based on
        time_2 = np.arange( time_1[0] , time_1[-1] , N ).tolist()
        for i in range(len(time_2)):
            time_2[i] = round(time_2[i] , 3)
        if DEBUG==1:    print "Time sample after repeat elimination : " , time_1[0:150] , "\n\n\nTime sample with target fs intervals: " , time_2[0:150]
        signal_2 = [np.nan]*len(time_2)
        # We need to create a list with
        # Now we need to make sure that we input the values in the time_1 array that matches the values in the time_2 array
        for i in range(len(time_2)):
            if time_2[i] in time_1:
                signal_2[i] = signal_1[ time_1.index(time_2[i]) ]
        if DEBUG==1:    print "Raw Signal sample:" , signal_1[0:100] , "\n\n\n Upsampled Signal: " , signal_2[0:100]
        ###################################################################################
        # The signal has been upsampled and the signal_2 and time_2 now have the target_fs and can be interpolated
        # BEGIN INTERPOLATION
        signal_pd = pd.Series(signal_2)
        signal_pd = signal_pd.interpolate(kind = 'cubic', inplace = False)
        # CONVERTING THE UPSAMPLED PANDAS SERIES TO THE OUTPUT LIST
        signal_resampled = signal_pd.tolist()
        time_resampled = time_2
        if DEBUG==1:    print "\n\n\n Upsampled Signal: " , signal_2[0:100] , "\n\n\nInterpolated Signal: " , signal_resampled[0:100]
        # I have confirmed by verification of data that the signal interpolate worked. Now we ensure by plotting some of the samples
        # USE THIS SECTION ONLY TO VERIFY THE ACCURACY OF THE RESAMPLING PROCESS
        #fig = plt.figure()
        #fig.tight_layout()
        #plt.title('Comparison of raw and resampled signal')
        #plt.subplot(2,1,1)
        #plt.plot(time_1[0:1500], signal_1[0:1500], label = 'Raw Signal' , linewidth = 0.1)
        #plt.subplot(2,1,2)
        #plt.plot(time_resampled[0:1500], signal_resampled[0:1500], label = 'Resampled Signal' , linewidth = 0.1)
        #plt.show()
        return time_resampled , signal_resampled
    except Exception as e:
        print "Resampling exception for participant: " , participant , ' in section: ' , section , 'Exception recorded: ' , e
        print 'Error on line {}'.format(sys.exc_info()[-1].tb_lineno)
        file = open(LOGFILE,'a')
        writer = csv.writer(file)
        writer.writerow(['Resampling exception for participant: ', participant , 'in section' , section , 'Exception recorded: ', e , 'Error on line {}'.format(sys.exc_info()[-1].tb_lineno) ])
        file.close()
#Function to process the ECG data with the biosppy.
def ProcessingHR(participant, section):
    if DEBUG ==0:   print " Processing the HR data for : " , participant,  "in section : " , section
    try:
        # we operate in the example folder in P002, but we no longer have to do this for later iters
        # opening the file
        file = open(MAINPATH+'/Data/'+participant+ '/' + section + '/HR.csv' , 'r')
        reader = csv.reader(file)
        header = next(reader)
        data = list (reader)
        file.close()
        #We are going to set the limits to see if we can identify the peaks effectively with the algorithm for biosppy
        time = [ float(row[0]) for row in data ]
        ECG1 = [ float(row[2]) for row in data ]
        ECG2 = [ float(row[3]) for row in data ]
        ECG3 = [ float(row[4]) for row in data ]
        ECG4 = [ float(row[5]) for row in data ]
        #ECGData = [ [float(row[2]), float(row[3]), float(row[4]), float(row[5])] for row in data ]
        # We need to calculate sampling rate as an average of the interval between all the time points. If the value is too far from the 1024 Hz, then we need to look at plots.
        fs = CalculateSamplingRate(time, participant, section)#We plot the raw sampling rate.
        # Sampling rate needs to firmly fixed at 1000 Hz
        time_resampled , ECG1_resampled = Resample( time , ECG1 , participant , section , 1000.0 , 'ecg' )
        time_resampled , ECG2_resampled = Resample( time , ECG2 , participant , section , 1000.0 , 'ecg' )
        time_resampled , ECG3_resampled = Resample( time , ECG3 , participant , section , 1000.0 , 'ecg' )
        time_resampled , ECG4_resampled = Resample( time , ECG4 , participant , section , 1000.0 , 'ecg' )
        # Write the resampled data to a file in case we might need it later
        file = open( MAINPATH+'/Data/'+participant+ '/' + section + '/HR_resampled.csv' , 'wb')
        writer = csv.writer(file)
        writer.writerow( [ 'Time' , 'ECG1' , 'ECG2' , 'ECG3' , 'ECG4' ])
        for i in range(len(time_resampled)):
            writer.writerow([ time_resampled[i] , ECG1_resampled[i] , ECG2_resampled[i] , ECG3_resampled[i] , ECG4_resampled[i] ])
        file.close()
        # Resampled File written.
        # MARKER IS NOT WRITTEN TO THE RESAMPLED FILE. ASSUMED NOT NEEDED FOR NOW.
        ############################################################
        # Biosppy calculates the ecg data based on the new resampled
        try:
            out = ecg.ecg(signal=ECG1, sampling_rate=1000 , show=False)
        except:
            try:
                out = ecg.ecg(signal=ECG2, sampling_rate=1000 , show=False)
            except:
                try:
                    out = ecg.ecg(signal=ECG3, sampling_rate=1000 , show=False)
                except:
                    try:
                        out = ecg.ecg(signal=ECG4, sampling_rate=1000 , show=False)
                    except Exception as e:
                        print "Biosppy Processing Exception in ecg.ecg for participant: " , participant , ' in seciton: ' , section , 'Exception recorded: ' , e
                        print 'Error on line {}'.format(sys.exc_info()[-1].tb_lineno)
                        file = open(LOGFILE,'a')
                        writer = csv.writer(file)
                        writer.writerow([' Biosppy Processing Exception in ecg.ecg for participant: ', participant , 'in section' , section , 'Exception recorded: ', e , 'Error on line {}'.format(sys.exc_info()[-1].tb_lineno) ])
                        file.close()
        # Output columns : ['ts', 'filtered', 'rpeaks', 'templates_ts', 'templates', 'heart_rate_ts', 'heart_rate']
        #Keep in mind the time is relative to the start of the interval used in the data subset and is NOT absolute time.
        # Also the indices at which the r-peaks occur is given in the r-peaks interval, which is convenient for r-r interval calculation if needed
        # NOTE : there are 'n+1' r-peaks for every 'n' sample points of heart rate calculated [ out[-1] , out[-2] have n columns, while out[2] has n+1 columns ].
        # The components of the ECG output are explained here:
        # ts in output is the same the resampled time interval. NOTE: STARTS AT 0. WE NEED TO ADD THE START VALUE OF THE INTERVAL FROM TIME
        # filtered is the filtered ECG signal
        # heart_rate is the output heart rate calculated after the estimation of the r-peaks.
        # heart_rate_ts is the time for the heart rates calculated. NOTE: THE MISSED PEAK DETECTIONS ARE IGNORED FOR HEART RATE DETECTION
        # NOTE: HEART_RATE_TS STARTS AT 0. NEED TO ADD THE START OF THE HEART_RATE_TS
        # r peaks are the indices in the 'filtered' ECG signal at which there are peaks
        # NOTE: We don't know what templates_ts or templates are. They seem to be same length.
        if DEBUG==1:    print " The output of the biosppy processing: " , "\n\n\n Output keys : " , out.keys()#, " Output as dict : " , out.as_dict(), '\n\n\n' , len(out[-1]) , '\n\n\n' , len(out[-2]) , 'r-peaks' , len(out[2])
        if DEBUG==1:    print "The length of the heart rate signal is : " , len( out.as_dict()['heart_rate'] ) , "\n\nThe heart rate time is : " , out.as_dict()['heart_rate']
        # Data is separated here.
        ts = out.as_dict()['ts']
        filtered = out.as_dict()['filtered']
        rpeaks = out.as_dict()['rpeaks']
        heartrate_ts = out.as_dict()['heart_rate_ts']
        heartrate = out.as_dict()['heart_rate']
        # We now plot and save the figure for later
        fig = plt.figure()
        fig.tight_layout()
        fig.suptitle( ' This is the output of the Biosppy library for participant ' + participant + ' in ' + section)
        plt.subplot(2,1,1)
        plt.title(' Plot of the filtered ECG output of the ecg.ecg function ')
        plt.plot( ts , filtered , 'r--' , label = 'Filtered ECG signal' , linewidth =0.1)
        plt.subplots_adjust(hspace = 0.3)# To prevent the overlap pf the text on the 2 subplots
        verticallines = [ ts[i] for i in rpeaks ]
        for x in verticallines:
            plt.axvline( x, linewidth = '1')
        plt.xlabel( 'Time (Ts output of ecg.ecg() in sec)')
        plt.ylabel( ' Filtered ECG signal (output of ecg.ecg()) ')
        plt.legend(loc = 'upper right')
        plt.subplot(2,1,2)
        plt.title( ' Plot of the resultant heart rate ')
        plt.plot( heartrate_ts , heartrate , 'g--' , label = 'Instantaneous output heart rate (beats per min)' , linewidth = 0.1 )
        #plt.subplots_adjust(hspace = 1)
        plt.xlabel( 'Time (Heartrate_ts output of ecg.ecg() in sec)')
        plt.ylabel( ' Output Heart rate (heart_rate output of ecg.ecg()) ')
        plt.legend(loc = 'upper right')
        plt.savefig( MAINPATH+'/Data/'+participant+'/' + section + '/FilteredECGSignal.pdf', bbox_inches = 'tight', dpi=900 , quality = 100)
        plt.close()
        #Plotted and saved.
    except Exception as e:
        print "HR Processing Exception Catcher for participant: " , participant , 'Exception recorded: ' , e
        print 'Error on line {}'.format(sys.exc_info()[-1].tb_lineno)
        file = open(LOGFILE,'a')
        writer = csv.writer(file)
        writer.writerow([' HR Processing Function Exception Caught for participant: ', participant , 'Exception recorded: ', e , 'Error on line {}'.format(sys.exc_info()[-1].tb_lineno) ])
        file.close()
#Function to analyze the GSR data with biosppy eda function
def ProcessingGSR(participant, section):
    print " Processing the GSR data for : ", participant
    try:
        #Opening the GSR File. For now, we operate in the Example file for now
        file = open(MAINPATH+'/Data/'+participant+'/Example/GSR.csv' , 'r')
        reader = csv.reader(file)
        header = next(reader)
        data = list (reader)
        file.close()
        # Loading the data
        time = [ float(row[2]) for row in data[100:50000] ]
        gsr = [ float(row[4]) for row in data[100:50000] ]
        # We need to calculate sampling rates here as well
        out = eda.eda(signal=gsr , sampling_rate=1024 , show=True , min_amplitude=0.1)
        print " The output for the GSR file is : " , out.as_dict() ,'\n\n\n' , out.keys()
        # The output of the EDA function is similar to the output of the ecg function
        # The lists included are:  ['ts', 'filtered', 'onsets', 'peaks', 'amplitudes']
        # The filtered GSR is included in the 2 column and the onsets , peaks contains the index of the SCR onsets(index) and peak values (amplitude).
        # The best way to analyze the data when we mark the data is to to identify the onset, time to the onset , and the peak amplitude
        # It is in the same tuple form as the ecg output. But this can work to out ad
    except Exception as e:
        print "HR Processing Exception Catcher for participant: " , participant , 'Exception recorded: ' , e
        print 'Error on line {}'.format(sys.exc_info()[-1].tb_lineno)
        file = open(LOGFILE,'a')
        writer = csv.writer(file)
        writer.writerow([' HR Processing Function Exception Caught for participant: ', participant , 'Exception recorded: ', e , 'Error on line {}'.format(sys.exc_info()[-1].tb_lineno) ])
        file.close()
# Function to analyze the PPG data. This is handwritten since biosppy doesn't have functions for this. A
#The time converter from iMotions data. Use the column Timestamp (col 10) from the iMotions file.
def ProcessingPPG(participant, section):
    if DEBUG==1:    print " Processing the PPG values for participant " , participant , " in section " , section
def iMotionsTimeConverter(inputcode):
    decimal = float(inputcode%1000)/1000
    inputcode = (inputcode - 1000*decimal)/1000
    secs = float(inputcode%100)
    inputcode = (inputcode -secs)/100
    mins = float(inputcode%100)
    inputcode = (inputcode-mins)/100
    hours = float(inputcode)
    time_abs = hours*3600 + mins*60 + secs + decimal
    return time_abs
#Main Function
if __name__ == '__main__':
    #Opening the Logging file
    file = open(LOGFILE , 'wb')
    writer = csv.writer(file)
    writer.writerow(['The output for the Plotting and Sampling the Data.'])
    file.close()
    try:
        #print " The Main Function for the Plotting and Sampling Function"
        listoffolders = os.listdir(MAINPATH+'/Data/')
        #Sort the Folders
        listoffolders.sort(key=SortFunc)
        if DEBUG == 1:  print "\nThe list of folder:", listoffolders
        for participant in listoffolders:
            print "\n\nAnalyzing the data for participant: " , participant
            #path to the contents of the folder.
            folderpath = MAINPATH+'/Data/'+participant+'/'
            #opening the iMotions File in each folder
            MAINPATH+'/Data/'+participant+'/PPG.csv'#Output File Names for witing the relevant data
            GSR_FILE_NAME = MAINPATH+'/Data/'+participant+'/GSR.csv'
            HR_FILE_NAME = MAINPATH+'/Data/'+participant+'/HR.csv'
            DRIVE_FILE_NAME = MAINPATH+'/Data/'+participant+'/DRIVE.csv'
            PPG_FILE_NAME = MAINPATH+'/Data/'+participant+'/PPG.csv'
            MARKER_FILE_NAME = MAINPATH+'/Data/'+participant+'/MARKERS.csv'
            try:
                if READANDEXTRACTINDIVIDUALFILES == 1:
                    file = open(folderpath+ participant + '.txt','r')
                    reader = csv.reader(file)
                    #print "First and Second lines in file: " , next(reader)
                    #We need to take the first 6 lines of the data out to record in a new info file, should we need it later.
                    InfoList =[]
                    for i in range(6):
                        if i==5:
                            row = next(reader)[0]
                            headerrow = row.split('\t')
                            InfoList.append(headerrow)
                        elif i<5:
                            InfoList.append(next(reader))
                    #If the process worked, the last row should be the header column
                    if DEBUG == 1:
                        print "\n\n\nInfoList :\n" , InfoList
                    data = list(reader)#Super List of lists containing all the data in one location.
                    file.close()
                    #Write the info file
                    file = open(folderpath+participant+'iMotionsInfo.txt','wb')
                    writer = csv.writer(file)
                    writer.writerows(InfoList)
                    file.close()
                    #To ensure that there are no empty cells in the data uber-list , we delete the first 30 rows.
                    for i in range(TOPELIMINATION):
                        data.remove( data[0] )#We always remove the first element
                    if DEBUG==0 and participant in CAT1:    print "\n\n\nParticipant in CAT1 , " , participant
                    if DEBUG==0 and participant in CAT2:    print "\n\n\nParticipant in CAT2 , " , participant
                    if DEBUG==0 and participant in CAT3:    print "\n\n\nParticipant in CAT3 , " , participant
                    #Now to make sure that the rows are properly divided
                    #re_data = []
                    # We are abandoning the process of compiling the data into a single array for each participant. It seems too large and almost always ends in the code crashing. Instead, we
                    # are going to write the data as it is being read from the file into respective, GSR, HR, Drive and PPG data files. This would be better since we don't have to store the variables in a uber-large
                    # array.
                    #First we make the files
                    #open the files
                    GSR_FILE = open(GSR_FILE_NAME, 'wb')
                    HR_FILE = open(HR_FILE_NAME, 'wb')
                    DRIVE_FILE = open(DRIVE_FILE_NAME, 'wb')
                    PPG_FILE = open(PPG_FILE_NAME , 'wb')
                    #Declaring writers
                    GSR_WRITER = csv.writer(GSR_FILE)
                    HR_WRITER = csv.writer(HR_FILE)
                    DRIVE_WRITER = csv.writer(DRIVE_FILE)
                    PPG_WRITER = csv.writer(PPG_FILE)
                    #Writing the header rows
                    GSR_WRITER.writerow(['Timestamp1' , 'Timestamp2' , 'AbsoluteTime' , 'Markers' , 'GSR'])
                    HR_WRITER.writerow(['Timestamp1' , 'Timestamp2' , 'AbsoluteTime' , 'Markers' , 'ECG1' , 'ECG2' , 'ECG3' , 'ECG4'])
                    DRIVE_WRITER.writerow(['Timestamp1' , 'Timestamp2' , 'AbsoluteTime' , 'Markers' , 'Brake' , 'Speed' , 'Steer' , 'Throttle'])
                    PPG_WRITER.writerow(['Timestamp1' , 'Timestamp2' , 'AbsoluteTime' , 'Markers' , 'PPG' , 'IBI'])
                    # Files are all written and we plot from them instead of loading and keeping all of the data
                    # Now plotting the data, we try to use the functions in the PlottingFunctions.py to make the plots.
                    # Steps :
                    # 1. We make the plots of just the markers in all participant folders.
                    # 2. We make the vertical indices at the points at which markers need to be placed.
                    # 3. We make the plots of the individual data streams without the markers as vertical lines for verification.
                    # 4. We then rewrite over the plots with the vertical indices. Bear in mind the step 3 plots will not be available , they will be overwritten
                    markers = [float(row[0].split('\t')[14]) for row in data]
                    convertedtime = [ iMotionsTimeConverter(float(row[0].split('\t')[9].split('_')[1])) for row in data ]
                    convertedtime = [ (convertedtime[i] - convertedtime[0]) for i in range(len(convertedtime)) ]
                    # The markers plot with the time from the iMotions UTC timestamp
                    Plot2Data( convertedtime , markers , 'iMotions Marker' , 'Plot of Markers in Participant '+participant  , 'MarkerPlot.pdf' , LOGFILE , participant , folderpath )
                    # I am making a plot of the time from the iMotions and the second source of time stamps. This way we know that the times match and that our times are accurate.
                    iMotionsTimeStamp = [ float(row[0].split('\t')[19]) for row in data ]
                    iMotionsTimeStamp = [ (iMotionsTimeStamp[i] - iMotionsTimeStamp[0])/1000 for i in range(len(iMotionsTimeStamp)) ]
                    Plot2Data( convertedtime, iMotionsTimeStamp , 'Time Stamp Comparison' , 'Plot of Time Stamps in iMotions for '+participant , 'TimeComparison.pdf' , LOGFILE , participant , folderpath )
                    ##########
                    # NOTE : I tested the imotions time stamp and the time stamps from the Shimmer devices. For most of the participants , the two data streams are fine. But for a couple of participants
                    # the Shimmer time stamp has a jump in the data stamp. For this reason, we have the iMotions time stamps are put in the data over the Shimmer device time stamps
                    for i,row in enumerate(data):
                        #Order of data :
                        #GSR FILE : < Timestamp 1 , TImestamp 2 , Markers, GSR >
                        #DRIVE FILE :  < Timestamp 1 , TImestamp 2 , Markers , Brake , Speed , Steer , Throttle >
                        #PPG FILE : <Timestamp 1 , Timestamp 2 , Markers , PPG , IBI(msecs)>
                        ########
                        #ECG File written after the if elif block
                        #ECG FILE < Timestamp 1 , TImestamp 2 , Markers , ECG(1-4) >
                        ########
                        #Writing the GSR File
                        GSR_WRITER.writerow([ float(row[0].split('\t')[9].split('_')[1]) , float(row[0].split('\t')[19]) , convertedtime[i] , float(row[0].split('\t')[14]) , float(row[0].split('\t')[29]) ])
                        ########
                        #Writing the DRIVE File
                        DRIVE_WRITER.writerow([ float(row[0].split('\t')[9].split('_')[1]) , float(row[0].split('\t')[19]) , convertedtime[i] , float(row[0].split('\t')[14]) ,  float(row[0].split('\t')[13]) , float(row[0].split('\t')[15]) , float(row[0].split('\t')[16]) , float(row[0].split('\t')[17]) ])
                        ########
                        #Writing the PPG File
                        PPG_WRITER.writerow([ float(row[0].split('\t')[9].split('_')[1]) , float(row[0].split('\t')[19]) , convertedtime[i] , float(row[0].split('\t')[14]) , float(row[0].split('\t')[27]) , float(row[0].split('\t')[28]) ])
                        ########
                        if participant in CAT1:
                            HR_WRITER.writerow([ float(row[0].split('\t')[9].split('_')[1]) , float(row[0].split('\t')[19]) , convertedtime[i] , float(row[0].split('\t')[14]) , float(row[0].split('\t')[44]) , float(row[0].split('\t')[45]) , float(row[0].split('\t')[46]) , float(row[0].split('\t')[47]) ])
                            #re_data.append( [ row[0].split('\t')[9] , row[0].split('\t')[19] , row[0].split('\t')[14] , row[0].split('\t')[44] , row[0].split('\t')[45] , row[0].split('\t')[46] , row[0].split('\t')[47] , row[0].split('\t')[13] \
                            #, row[0].split('\t')[15] , row[0].split('\t')[16] , row[0].split('\t')[17] , row[0].split('\t')[27] , row[0].split('\t')[28] , row[0].split('\t')[29] ] )
                        elif participant in CAT2:
                            HR_WRITER.writerow([ float(row[0].split('\t')[9].split('_')[1]) , float(row[0].split('\t')[19]) , convertedtime[i] , float(row[0].split('\t')[14]) , float(row[0].split('\t')[36]) , float(row[0].split('\t')[37]) , float(row[0].split('\t')[38]) , float(row[0].split('\t')[39]) ])
                            #re_data.append( [ row[0].split('\t')[9] , row[0].split('\t')[19] , row[0].split('\t')[14] , row[0].split('\t')[36] , row[0].split('\t')[37] , row[0].split('\t')[38] , row[0].split('\t')[39] , row[0].split('\t')[13] \
                            #, row[0].split('\t')[15] , row[0].split('\t')[16] , row[0].split('\t')[17] , row[0].split('\t')[27] , row[0].split('\t')[28] , row[0].split('\t')[29] ] )
                        elif participant in CAT3:
                            HR_WRITER.writerow([ float(row[0].split('\t')[9].split('_')[1]) , float(row[0].split('\t')[19]) , convertedtime[i] , float(row[0].split('\t')[14]) , float(row[0].split('\t')[31]) , float(row[0].split('\t')[32]) , float(row[0].split('\t')[33]) , float(row[0].split('\t')[34]) ])
                            #re_data.append( [ row[0].split('\t')[9] , row[0].split('\t')[19] , row[0].split('\t')[14] , row[0].split('\t')[31] , row[0].split('\t')[32] , row[0].split('\t')[33] , row[0].split('\t')[34] , row[0].split('\t')[13] \
                            #, row[0].split('\t')[15] , row[0].split('\t')[16] , row[0].split('\t')[17] , row[0].split('\t')[27] , row[0].split('\t')[28] , row[0].split('\t')[29] ] )
                    ########
                    #Closing the files
                    GSR_FILE.close()
                    HR_FILE.close()
                    DRIVE_FILE.close()
                    PPG_FILE.close()
                ##################################################################################################################################################
                ##################################################################################################################################################
                # End of the RAW FILE READ if block
                #From this point on, we are going to re-read the data in the separated files rather than use the data uber array every single time
                # We can do the plot iteratively using the an array of all the files and the plotting functions
                if MAKEPLOTS == 1:
                    #Reading and plotting data :
                    filelist = [GSR_FILE_NAME, HR_FILE_NAME, DRIVE_FILE_NAME, PPG_FILE_NAME]
                    for item in filelist:
                        file = open(item, 'r')
                        reader = csv.reader(file)
                        headerrow = next(reader)
                        data_ = list(reader)
                        if DEBUG == 1:  print "\n Header row: " , headerrow
                        data = list(reader)
                        if DEBUG == 1:  print "\n First Row of Data: ", data_[0]
                        length = len(data_[0])# This is the length of the arrays in data_
                        length = length - 3# We discount the first three columns since they are timestamps
                        t = [ float(data_[i][2]) for i in range(len(data_)) ]
                        for k in range(length):#k is the index for the columns.
                            signal = [ float(data_[i][3+k]) for i in range(len(data_)) ]# columns 3 to the end of (3+k) are read here. We use the titles we read from the
                            signalname = headerrow[3+k]
                            #No need to plot the Markers again. They have already been plotted before
                            #if signalname not in ['Markers']:
                            Plot2Data( t , signal , signalname , 'Plot of '+signalname+' for participant '+participant , signalname+'.pdf' , LOGFILE , participant , folderpath )
                #####################
                if PROCESSMARKERS == 1:
                    # Writing the section to reprocess the markers for all the participants.
                    # After this we need to create a dictionary of which 'ones' matter for which participants. Should be a table of sorts that needs to be made.
                    ###############
                    #Load Markers from ECG file
                    file = open(HR_FILE_NAME, 'r')#Using the HR file name
                    reader = csv.reader(file)
                    headerrow = next(reader)
                    data_ = list(reader)
                    markers = [ float(row[3]) for row in data_ ]
                    time = [ float(row[2]) for row in data_ ]
                    if DEBUG==1:    print "Time and Markers: ", time[1:10], markers[1:10]
                    file.close()
                    #################
                    # Marker Noise elimination section:
                    # Populate a second array for modification of values separate from the markers array
                    markers_filtered = markers
                    # Search the array for changes in markers for increases to 20 and set it to the previous value to eliminate the jumps in marker data
                    for i in range(len(markers_filtered)):
                        if markers_filtered[i] == 20:
                            markers_filtered[i] = markers_filtered[i-1]
                    # Second, We rewrite the marker file with a third column of where the markers chage values
                    # We are also writing a second file with a list of just the points at which the marker changes values combined with the purpose of change in value as best understood
                    # Analyzing the changes in the marker values
                    changesinmarkers = [0]*len(markers_filtered)
                    for i in range(len(markers_filtered)-1):
                        if abs( markers_filtered[i+1] - markers_filtered[i] ) > 0:
                            changesinmarkers[i+1] = 1
                    # Section to write the marker file
                    file = open(MARKER_FILE_NAME, 'wb')
                    writer = csv.writer(file)
                    writer.writerow(['AbsoluteTime' , 'Markers', 'Filtered Markers' , 'Changes in Markers'])
                    for i in range(len(time)):
                        writer.writerow([ time[i] , markers[i] , markers_filtered[i] , changesinmarkers[i] ])
                    file.close()
                    if DEBUG ==1:   print "\nMarker File Written for participant: ", participant
                    ###########
                    #Plotting the Filtered Markers
                    Plot2Data( time, markers_filtered , 'Filtered Markers' , 'Plot of Filtered Markers for participant '+participant , 'FilteredMarkers.pdf' , LOGFILE , participant , folderpath )
                    ###########
                    if REWRITEABRIDGEDMARKERFILE == 1:
                        # I am going to write a new file with < Abs Time , Count , Event , Marker Start , Marker End >. This is the abridged key for processing the data
                        MARKER_FILE_SHORT_NAME = MAINPATH+'/Data/'+participant+'/MARKERS_SHORT.csv'
                        file = open(MARKER_FILE_SHORT_NAME,'wb')
                        writer = csv.writer(file)
                        writer.writerow([' Abridged file for Markers '])
                        writer.writerow(['AbsoluteTime', 'Count' , 'Filtered Marker Old' , 'Filtered Marker New' , 'Event'])
                        sum = 0
                        for i in range(len(changesinmarkers)):
                            if changesinmarkers[i] == 1:
                                sum = sum+1
                                writer.writerow([time[i] , sum , markers_filtered[i-1] , markers_filtered[i] , ' '])
                        file.close()
                        if DEBUG ==1:   print "\nAbridged Marker File Written for participant: ", participant
                ##################################################################################################
                ##################################################################################################
                if SEGMENTDATA ==1:
                    try:
                        # We read the MARKERS_SHORT.csv file and get the times for the event and save them in subfolders with the HR and GSR data
                        print " Segmenting data for participant : ", participant
                        #Open Abridged marker file :
                        file = open(folderpath+'MARKERS_SHORT.csv' , 'r')
                        reader = csv.reader(file)
                        ignore = next(reader)
                        data = list(reader)
                        file.close()
                        Events = [ 'GoAroundRocks' , 'CurvedRoads' , 'Failure1' , 'HighwayExit' , 'RightTurn1' , 'RightTurn2' , 'PedestrianEvent' , 'TurnRight3' , 'BicycleEvent' , 'TurnRight4' , 'TurnRight5'\
                        , 'RoadObstructionEvent' , 'HighwayEntryEvent' , 'HighwayIslandEvent' ]
                        WindowSizes = [ GoAroundRocksWiNDOW , CurvedRoadsWINDOW , Failure1WINDOW , HighwayExitWINDOW , TURNWINDOW , TURNWINDOW , PedestrianEventWINDOW , TURNWINDOW , BicycleEventWINDOW , TURNWINDOW , TURNWINDOW\
                         , RoadObstructionEventWINDOW , HighwayEntryEventWINDOW , HighwayIslandEventWINDOW ]
                        MarkerTimes = []#This is the corresponding times for the Events in the above list
                        for row in data:
                            if row[4] in Events:
                                MarkerTimes.append(float(row[0]))
                        if DEBUG==0:    print " Marker Times for the events are : " , MarkerTimes
                        #############
                        # Marker Times are recorded. We now move on to creating folders
                        for eventfolder in Events:
                            if not os.path.exists(folderpath+eventfolder):
                                os.makedirs(folderpath+eventfolder)
                        if DEBUG==1:    print " Folder for the events are created : " , os.listdir(folderpath)
                        #############
                        # Folders are created, we now move on to the clipping the data.
                        # We segment the 4 files one at a time.
                        # We have to divide the data one file at a time, this seems to be best way to segment data.
                        if GSRSEGMENTATION == 1:
                            # Starting with the GSR data segmentation
                            # Load the GSR data
                            file = open( folderpath+'GSR.csv','r')
                            reader = csv.reader(file)
                            ignore = next(reader)#This is the header
                            data = list(reader)
                            file.close()
                            time = [ float(row[2]) for row in data ]
                            gsr = [ float(row[4]) for row in data ]
                            marker = [ float(row[3]) for row in data ]
                            # data loadded.
                            # For every event , we find the index of the element in time, then we subtract and add WindowSizes. Then we use the find_nearest function
                            # to get nearest value's index for the edges of the windows. These indices can then be used to extract the
                            for i,event in enumerate(Events):
                                windowstart = MarkerTimes[i] - WindowSizes[i]
                                windowend = MarkerTimes[i] + WindowSizes[i]
                                #Now we need to find the index of the windowstart and windowend times in the time list and then seek the corresponding values in the gsr data too
                                index_Marker = find_nearest( np.asarray(time) , MarkerTimes[i] )
                                index_windowstart = find_nearest( np.asarray(time) , windowstart )
                                index_windowend = find_nearest( np.asarray(time) , windowend )
                                if DEBUG == 1:  print " Verification of marker index using find_nearest function for : ", event , " is " , time[index_Marker]
                                # Need to create a file for writing the data
                                file = open (folderpath+event+'/GSR.csv' ,'wb')
                                writer = csv.writer(file)
                                writer.writerow(['Time' , 'Marker' , 'GSR'])
                                # Now that we have the indices, we can clip the relevant information and write to the file
                                for i in range(index_windowstart , index_windowend):
                                    writer.writerow([ time[i] , marker[i] , gsr[i] ])
                                file.close()
                        if HRSEGMENTATION == 1:
                            # Next up is the HR data segmentation
                            # Load the HR data
                            file = open (folderpath+'HR.csv' , 'r')
                            reader = csv.reader(file)
                            ignore = next(reader)
                            data = list(reader)
                            file.close()
                            time = [ float(row[2]) for row in data ]
                            markers = [ float(row[3]) for row in data ]
                            hr1 = [ float(row[4]) for row in data ]
                            hr2 = [ float(row[5]) for row in data ]
                            hr3 = [ float(row[6]) for row in data ]
                            hr4 = [ float(row[7]) for row in data ]
                            # Data Loaded
                            for i,event in enumerate(Events):
                                windowstart = MarkerTimes[i] - WindowSizes[i]
                                windowend = MarkerTimes[i] + WindowSizes[i]
                                #Find the indices of the markers and the windows start and window ends
                                index_Marker = find_nearest( np.asarray(time) , MarkerTimes[i] )
                                index_windowstart = find_nearest( np.asarray(time) , windowstart )
                                index_windowend = find_nearest( np.asarray(time) , windowend )
                                if DEBUG == 1:  print " Verification of marker index using find_nearest function for : " , event , " is " , time[index_Marker]
                                # Need to create a file for writing the data
                                file = open ( folderpath+event+'/HR.csv' , 'wb')
                                writer = csv.writer(file)
                                writer.writerow([ 'Time' , 'Marker' , 'ECG1' , 'ECG2' , 'ECG3' , 'ECG4' ])
                                # Now that we have the indices, we can clip the relevant information and write to the file
                                for i in range(index_windowstart , index_windowend):
                                    writer.writerow([ time[i] , marker[i] , hr1[i] , hr2[i] , hr3[i] , hr4[i] ])
                                file.close()
                        if PPGSEGMENTATION == 1:
                            # PPG data segmentation
                            file = open( folderpath + 'PPG.csv' , 'r' )
                            reader = csv.reader(file)
                            ignore = next(reader)
                            data = list(reader)
                            file.close()
                            time = [ float(row[2]) for row in data ]
                            marker = [ float(row[3]) for row in data ]
                            ppg = [ float(row[4]) for row in data ]
                            ibi = [ float(row[5]) for row in data ]
                            # Data loaded
                            for i, event in enumerate(Events):
                                windowstart = MarkerTimes[i] - WindowSizes[i]
                                windowend = MarkerTimes[i] + WindowSizes[i]
                                # Find the indices of the markers and the window start and the window ends
                                index_Marker = find_nearest( np.asarray(time) , MarkerTimes[i] )
                                index_windowend = find_nearest( np.asarray(time) , windowend )
                                index_windowstart = find_nearest( np.asarray(time), windowstart )
                                if DEBUG == 1: print " Verification of marker index using find_nearest function for : ", event , " is " , time[index_Marker]
                                # Output for the PPG file
                                file = open( folderpath+event+'/PPG.csv' , 'wb' )
                                writer = csv.writer(file)
                                writer.writerow( [ 'Time' , 'Marker' , 'PPG' , 'IBI' ] )#Header
                                for i in range(index_windowstart, index_windowend):
                                    writer.writerow([ time[i] , marker[i] , ppg[i] , ibi[i] ])
                                file.close()
                        if DRIVESEGMENTATION == 1:
                            # DRIVE data segmentation
                            file = open( folderpath+'DRIVE.csv' ,'r')
                            reader = csv.reader(file)
                            ignore = next(reader)
                            data = list(reader)
                            file.close()
                            time = [ float(row[2]) for row in data ]
                            marker = [ float(row[3]) for row in data ]
                            brake = [ float(row[4]) for row in data ]
                            speed = [ float(row[5]) for row in data ]
                            steer = [ float(row[6]) for row in data ]
                            throttle = [ float(row[7]) for row in data ]
                            #Data Loaded
                            for i , event in enumerate(Events):
                                windowstart = MarkerTimes[i] - WindowSizes[i]
                                windowend = MarkerTimes[i] + WindowSizes[i]
                                #locating the indices
                                index_Marker = find_nearest(np.asarray(time) , MarkerTimes[i] )
                                index_windowstart = find_nearest(np.asarray(time) , windowstart )
                                index_windowend = find_nearest(np.asarray(time) , windowend )
                                if DEBUG == 1: print " Verification of marker index using find_nearest function for : ", event , " is " , time[index_Marker]
                                #output file writing
                                file = open(folderpath+event+'/DRIVE.csv' , 'wb' )
                                writer= csv.writer(file)
                                writer.writerow([ 'Time' , 'Marker' , 'Brake' , 'Speed' , 'Steer' , 'Throttle' ])#Header
                                for i in range(index_windowstart , index_windowend):
                                    writer.writerow([ time[i] , marker[i] , brake[i] , speed[i] , steer[i] , throttle[i] ])
                                file.close()
                    except Exception as e:
                        print " Exception recorded for participant in the Segmentation process : " , participant , " Error : ", e
                        print  'Error on line {}'.format(sys.exc_info()[-1].tb_lineno)
                        ile = open(LOGFILE, 'a')
                        writer = csv.writer(file)
                        writer.writerow([' Exception for participant ', participant , 'Exception : ', e , 'on line {}'.format(sys.exc_info()[-1].tb_lineno)])
                        file.close()
                ##################################################################################################
                ##################################################################################################
                if SIGNALPROCESSING == 1:
                    ############# Signal Processing for raw HR, PPG, IBI and GSR signals ############################
                    # 1) Next we use the functions and structure below to process and analyze the data in each segment. Basically,
                    # run the script for each section in current participant
                    # 2) The ECG analysis (biosppy) will output the 2 streams of data - > HR , IBI and HRV
                    # 3) The GSR signal analysis (biosppy) will output [ onset time, peak amplitude ] . This should also help with identifying
                    # how many peaks are present.
                    # 4) The PPG signal analysis (to be done by hand) will give a measurement of heart rate and the IBI (inter-beat interval).
                    # These are two additional measures for the same physiological data. We store and compare them to the ECG outptut.
                    ##### Starting the HR processing ############
                    # Individual data files can be analyzed if needed using the flags at the top of the file.
                    ##############
                    #Declaring the Events
                    Events = [ 'GoAroundRocks' , 'CurvedRoads' , 'Failure1' , 'HighwayExit' , 'RightTurn1' , 'RightTurn2' , 'PedestrianEvent' , 'TurnRight3' , 'BicycleEvent' , 'TurnRight4' , 'TurnRight5'\
                    , 'RoadObstructionEvent' , 'HighwayEntryEvent' , 'HighwayIslandEvent' ]
                    for event in Events:
                        if HRPROCESSING == 1:# and participant=='P002':
                            ProcessingHR(participant, event)
                        ####################################
                        if GSRPROCESSING == 1:# and participant=='P002':
                            ProcessingGSR(participant, event)
                        ####################################
                        if PPGPROCESSING == 1:# and participant=='P002':
                            ProcessingPPG(participant, event)
                        # We ignore the Drive Data for later
                ##################################################################################################
                ##################################################################################################
                if REMOVEFILE ==1:
                    # We write a short section to remove previously written files that are present in all participant folder.
                    DeleteList = ['TimeComparison.pdf' ,'Markers.pdf']
                    for item in DeleteList:
                        if os.path.exists(MAINPATH+'/Data/'+participant+'/'+item):
                            os.remove(MAINPATH+'/Data/'+participant+'/'+item)
                            if DEBUG ==1:   print "Deleted: " , item
                ##################################################################################################
                ##################################################################################################
                if BACKUPDATA == 1:
                    grouplist = ['MarkerPlot.pdf' , 'FilteredMarkers.pdf' , 'MARKERS_SHORT.csv' ]
                    for item in grouplist:
                        if os.path.exists(MAINPATH+'/Data/'+participant+'/'+item):
                            shutil.copy(MAINPATH+'/Data/'+participant+'/'+item , MAINPATH+'/AuxillaryInformation/BackupofImportantData/' + participant+item)#Adding the participant name to the item name before
                            if DEBUG ==1:   print "File moved ", item
                ##################################################################################################
                ##################################################################################################
            except Exception as e:
                print " Exception recorded for participant : ", participant, "Error is : ", e
                print 'Error on line {}'.format(sys.exc_info()[-1].tb_lineno)
                file = open(LOGFILE, 'a')
                writer = csv.writer(file)
                writer.writerow([' Exception for participant ', participant , 'Exception : ', e , 'on line {}'.format(sys.exc_info()[-1].tb_lineno)])
                file.close()
    except Exception as e:
        print "Main Exception Catcher" ,e
        print 'Error on line {}'.format(sys.exc_info()[-1].tb_lineno)
        file = open(LOGFILE,'a')
        writer = csv.writer(file)
        writer.writerow([' Main Function Exception Catcher ',e,'Error on line {}'.format(sys.exc_info()[-1].tb_lineno) ])
        file.close()
