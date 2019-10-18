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
# Listed below are the events we are looking for
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
LOGFILE = os.path.abspath('.') + '/CurrentOutputFile.csv'#Using one output file for all the scripts from now to avoid flooding the working folder
MAINPATH = os.path.abspath('.')#Always specify absolute path for all path specification and file specifications
DEBUG = 0# To print statements for debugging
TOPELIMINATION = 100
READANDEXTRACTINDIVIDUALFILES = 0# This is the flag to make sure that files are read from the start
MAKEPLOTS = 0#Make the individual plots from the files of all the data streams
REMOVEFILE = 0# We are using this marker to remove a file with the same name across all participatns in similar locations.
PROCESSMARKERS = 0#Analyze the markers and make abridged version of the markers file for event processing
REWRITEABRIDGEDMARKERFILE = 0;#This Flag is to rewrite the marker file.
HRPROCESSING = 0#This is to process the HR Data only
GSRPROCESSING = 0#This is to process the GSR Data only
BACKUPDATA = 0#This is to backup files that are important or are needed for later.
SEGMENTDATA = 1#This is to cut the data into windows for all the data in the study.
GSRSEGMENTATION = 1# This is the subsegment for GSR data segmentation in the SEGMENTDATA section
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
#Function to process the ECG data with the biosppy.
def ProcessingHR(participant):
    print " Processing the HR data for : " , participant
    try:
        # we operate in the example folder in P002, but we no longer have to do this for later iters
        # opening the file
        file = open(MAINPATH+'/Data/'+participant+'/Example/HR.csv' , 'r')
        reader = csv.reader(file)
        header = next(reader)
        data = list (reader)
        file.close()
        #We are going to set the limits to see if we can identify the peaks effectively with the algorithm for biosppy
        time = [ float(row[2]) for row in data[100:20000] ]
        ECG1 = [ float(row[4]) for row in data[101:20000] ]
        ECG2 = [ float(row[5]) for row in data[100:20000] ]
        ECG3 = [ float(row[6]) for row in data[100:20000] ]
        ECG4 = [ float(row[7]) for row in data[100:20000] ]
        # We need to calculate sampling rate as an average of the interval between all the time points. If the value is too far from the 1024 Hz, then we need to look at plots.
        out = ecg.ecg(signal=ECG1, sampling_rate=1024 , show=False)
        # Biosppy works really well in ECG processing, it takes the source of the ECG amd identifies the peaks and calculates the heart rates between the peaks.
        # the output of the ecg.ecg function ( 'out' in this case), contains all the processes output of the ecg.ecg() .
        # Output columns : ['ts', 'filtered', 'rpeaks', 'templates_ts', 'templates', 'heart_rate_ts', 'heart_rate']
        # Output is a weird tuple||dict hybrid. Using the as_dict() function makes it a little better, though obviously the best thing to do is just convert to a list.
        # Using out[-1] , out[-2] we can get the heart rate and the heart rate at which the time was recorded. Keep in mind the time is relative to the start of the interval used in the data subset
        # and is NOT absolute time.
        # Also the indices at which the r-peaks occur is given in the r-peaks interval, which is convenient for r-r interval calculation if needed
        # NOTE : there are 'n+1' r-peaks for every 'n' sample points of heart rate calculated [ out[-1] , out[-2] have n columns, while out[2] has n+1 columns ].
        print " The output of the biosppy processing: " , " Output as dict : " , out.as_dict() , "\n\n\n Output keys : " , out.keys()
        #, '\n\n\n' , len(out[-1]) , '\n\n\n' , len(out[-2]) , 'r-peaks' , len(out[2])
    except Exception as e:
        print "HR Processing Exception Catcher for participant: " , participant , 'Exception recorded: ' , e
        print 'Error on line {}'.format(sys.exc_info()[-1].tb_lineno)
        file = open(LOGFILE,'a')
        writer = csv.writer(file)
        writer.writerow([' HR Processing Function Exception Caught for participant: ', participant , 'Exception recorded: ', e , 'Error on line {}'.format(sys.exc_info()[-1].tb_lineno) ])
        file.close()
#Function to analyze the GSR data with biosppy eda function
def ProcessingGSR(participant):
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
#the time converter from iMotions data. Use the column Timestamp (col 10) from the iMotions file.
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
                                if DEBUG == 0:  print " Calculating the time using the find_nearest function for : ", event , " is " , time[index_Marker]
                                # Need to create a file for writing the data
                                file = open (folderpath+event+'/GSR.csv' ,'wb')
                                writer = csv.writer(file)
                                writer.writerow(['Time' , 'Marker' , 'GSR'])
                                # Now that we have the indices, we can clip the relevant information and write to the file
                                for i in range(index_windowstart , index_windowend):
                                    writer.writerow([ time[i] , marker[i] , gsr[i] ])
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
                if HRPROCESSING == 1 and participant=='P002':   ProcessingHR(participant)# ADD SECTION
                ####################################
                if GSRPROCESSING == 1 and participant=='P002':   ProcessingGSR(participant)# ADD SECTION
                ####################################
                ###### Now moving on to the PPG and IBI data. There's no processing this using , we need to do this by hand.
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
