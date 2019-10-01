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
import os, sys, csv , re
import matplotlib as plt
import numpy as np
import csv
import matplotlib.pyplot as plt
import math
from scipy import interpolate
import numpy as np
import pandas as pd
from statistics import mean
LOGFILE = os.path.abspath('.') + '/CurrentOutputFile.csv'#Using one output file for all the scripts from now to avoid flooding the working folder
MAINPATH = os.path.abspath('.')#Always specify absolute path for all path specification and file specifications
DEBUG = 0# To print statements for debugging
TOPELIMINATION = 50
#The 3 categories are defined here. Check here before adding the
#We use this function as a key to sorting the list of folders (participants).
CAT1 = [ 'P002', 'P004' , 'P005' , 'P007' , 'P008' , 'P009' , 'P010' , 'P011' , 'P013' , 'P016' , 'P021' , 'P023' , 'P024' , 'P025' , 'P028' , 'P032' ]
CAT2 = [ 'P022' , 'P037' , 'P040' , 'P042' , 'P045' , 'P046' , 'P047' , 'P055' , 'P057' , 'P058' , 'P060' , 'P064' , 'P066' , 'P067' , 'P069' , 'P071' , \
'P076' , 'P077' , 'P085' , 'P086' , 'P087' , 'P088' , 'P089' , 'P090' , 'P090' , 'P093' , 'P094' , 'P095' , 'P096' , 'P097' ,'P098']
CAT3 = ['P012']
def SortFunc(e):
    return int( re.sub('P','',e) )
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
        if DEBUG == 1:
            print "\nThe list of folder:", listoffolders
        for participant in listoffolders:
            #path to the contents of the folder.
            folderpath = MAINPATH+'/Data/'+participant+'/'
            #opening the iMotions File in each folder
            try:
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
                if DEBUG==0 and participant in CAT1:    print "Participant in CAT1 , " , participant
                if DEBUG==0 and participant in CAT2:    print "Participant in CAT2 , " , participant
                if DEBUG==0 and participant in CAT3:    print "Participant in CAT3 , " , participant
                #Now to make sure that the rows are properly divided
                #re_data = []
                # We are abandoning the process of compiling the data into a single array for each participant. It seems too large and almost always ends in the code crashing. Instead, we
                # are going to write the data as it is being read from the file into respective, GSR, HR, Drive and PPG data files. This would be better since we don't have to store the variables in a uber-large
                # array.
                #First we make the files
                #File Names
                GSR_FILE_NAME = MAINPATH+'/Data/'+participant+'/GSR.csv'
                HR_FILE_NAME = MAINPATH+'/Data/'+participant+'/HR.csv'
                DRIVE_FILE_NAME = MAINPATH+'/Data/'+participant+'/DRIVE.csv'
                PPG_FILE_NAME = MAINPATH+'/Data/'+participant+'/PPG.csv'
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
                GSR_WRITER.writerow(['TImestamp1' , 'TImestamp2' , 'Markers' , 'GSR( micro siemens )'])
                HR_WRITER.writerow(['TImestamp1' , 'TImestamp2' , 'Markers' , 'ECG1' , 'ECG2' , 'ECG3' , 'ECG4'])
                DRIVE_WRITER.writerow(['TImestamp1' , 'TImestamp2' , 'Markers' , 'Brake' , 'Speed' , 'Steer' , 'Throttle'])
                PPG_WRITER.writerow(['TImestamp1' , 'TImestamp2' , 'Markers' , 'PPG Heart Rate'])
                for row in data:
                    #Order of data :
                    #GSR FILE : < Timestamp 1 , TImestamp 2 , Markers, GSR >
                    #DRIVE FILE :  < Timestamp 1 , TImestamp 2 , Markers , Brake , Speed , Steer , Throttle >
                    #PPG FILE : <Timestamp 1 , Timestamp 2 , Markers , PPG , IBI(msecs)>
                    ########
                    #ECG File written after the if elif block
                    #ECG FILE < Timestamp 1 , TImestamp 2 , Markers , ECG(1-4) >
                    ########
                    #Writing the GSR File
                    GSR_WRITER.writerow([ float(row[0].split('\t')[9].split('_')[1]) , float(row[0].split('\t')[19]) , float(row[0].split('\t')[14]) , float(row[0].split('\t')[29]) ])
                    ########
                    #Writing the DRIVE File
                    DRIVE_WRITER.writerow([ float(row[0].split('\t')[9].split('_')[1]) , float(row[0].split('\t')[19]) , float(row[0].split('\t')[14]) ,  float(row[0].split('\t')[13]) , float(row[0].split('\t')[15]) , float(row[0].split('\t')[16]) , float(row[0].split('\t')[17]) ])
                    ########
                    #Writing the PPG File
                    GSR_WRITER.writerow([ float(row[0].split('\t')[9].split('_')[1]) , float(row[0].split('\t')[19]) , float(row[0].split('\t')[14]) , float(row[0].split('\t')[27]) , float(row[0].split('\t')[28]) ])
                    ########
                    if participant in CAT1:
                        HR_WRITER.writerow([ float(row[0].split('\t')[9].split('_')[1]) , float(row[0].split('\t')[19]) , float(row[0].split('\t')[14]) , float(row[0].split('\t')[44]) , float(row[0].split('\t')[45]) , float(row[0].split('\t')[46]) , float(row[0].split('\t')[47]) ])
                        #re_data.append( [ row[0].split('\t')[9] , row[0].split('\t')[19] , row[0].split('\t')[14] , row[0].split('\t')[44] , row[0].split('\t')[45] , row[0].split('\t')[46] , row[0].split('\t')[47] , row[0].split('\t')[13] \
                        #, row[0].split('\t')[15] , row[0].split('\t')[16] , row[0].split('\t')[17] , row[0].split('\t')[27] , row[0].split('\t')[28] , row[0].split('\t')[29] ] )
                    elif participant in CAT2:
                        HR_WRITER.writerow([ float(row[0].split('\t')[9].split('_')[1]) , float(row[0].split('\t')[19]) , float(row[0].split('\t')[14]) , float(row[0].split('\t')[36]) , float(row[0].split('\t')[37]) , float(row[0].split('\t')[38]) , float(row[0].split('\t')[39]) ])
                        #re_data.append( [ row[0].split('\t')[9] , row[0].split('\t')[19] , row[0].split('\t')[14] , row[0].split('\t')[36] , row[0].split('\t')[37] , row[0].split('\t')[38] , row[0].split('\t')[39] , row[0].split('\t')[13] \
                        #, row[0].split('\t')[15] , row[0].split('\t')[16] , row[0].split('\t')[17] , row[0].split('\t')[27] , row[0].split('\t')[28] , row[0].split('\t')[29] ] )
                    elif participant in CAT3:
                        HR_WRITER.writerow([ float(row[0].split('\t')[9].split('_')[1]) , float(row[0].split('\t')[19]) , float(row[0].split('\t')[14]) , float(row[0].split('\t')[31]) , float(row[0].split('\t')[32]) , float(row[0].split('\t')[33]) , float(row[0].split('\t')[34]) ])
                        #re_data.append( [ row[0].split('\t')[9] , row[0].split('\t')[19] , row[0].split('\t')[14] , row[0].split('\t')[31] , row[0].split('\t')[32] , row[0].split('\t')[33] , row[0].split('\t')[34] , row[0].split('\t')[13] \
                        #, row[0].split('\t')[15] , row[0].split('\t')[16] , row[0].split('\t')[17] , row[0].split('\t')[27] , row[0].split('\t')[28] , row[0].split('\t')[29] ] )
                ########
                #Closing the files
                GSR_FILE.close()
                HR_FILE.close()
                DRIVE_FILE.close()
                PPG_FILE.close()
                # Files are all written and we plot from them instead of loading and keeping all of the data
            except Exception as e:
                print " Exception recorded for participant : ", participant
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