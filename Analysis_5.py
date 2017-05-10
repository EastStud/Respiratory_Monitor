#import the necessary modules
import freenect
import cv2
import numpy as np
from numpy import convolve
import time as t
import matplotlib.pyplot as plt
from PIL import Image
import scipy.ndimage.filters as filt
import scipy.signal as sig
import scipy.stats as stat
from scipy.fftpack import fft
import pickle
import math
import sys
 
#function to get RGB image from kinect
def get_video():
    array,_ = freenect.sync_get_video()
    array = cv2.cvtColor(array,cv2.COLOR_RGB2BGR)
    return array
 
#function to get depth image from kinect
def get_depth():
    array,_ = freenect.sync_get_depth()
    array = array.astype(np.uint8)
    return array

#function to plot two lists
def plot_lists(list1, list2, list3):
    plt.subplot(231), plt.imshow(list1[0],'gray'),plt.title('First Frame')
    plt.subplot(234), plt.imshow(list1[len(list1)-1],'gray'),plt.title('Last Frame')
    plt.subplot(233), plt.imshow(list2[0],'gray'),plt.title('First Frame Sub')
    plt.subplot(236), plt.imshow(list2[len(list2)-1],'gray'),plt.title('Last Frame Sub')
    plt.subplot(232), plt.imshow(list3[0],'gray'),plt.title('First Frame blur')
    plt.subplot(235), plt.imshow(list3[len(list3)-1],'gray'),plt.title('Last Frame blur')
    plt.imshow
    plt.show()

#function to get the time axis steps for a given period of time and samples
def get_timeAxis(timeEnd,timeStart, Samples):
    intervals = Samples / (timeEnd - timeStart)
    timeAxis = np.ones(Samples)
    prev = 0
    for num in range(len(timeAxis)):
        timeAxis[num] = timeAxis[num] * intervals + prev
        prev = timeAxis[num]
    return timeAxis

# Function for getting the x values scaled according to the orig data
def get_maxmin_xs(timeEnd, timeStart, Samples, Xs, offset):
    intervals = Samples / (timeEnd - timeStart)
    timeAxis = np.ones(Samples)
    someYs = []
    tempX = Xs
    prev = offset * intervals
    for num in range(len(timeAxis)):
        timeAxis[num] = timeAxis[num] * intervals + prev
        if num in tempX:
	    index = tempX.index(num)
            someYs.append(timeAxis[num])
	    #someYs.append(num * intervals)
            print("num is " + str(num) + " and tempX is " + str(tempX[index]))
            tempX.pop(index)
        prev = timeAxis[num]
    return someYs

# Function for getting FFT x values scaled accordingly
def fft_xs(timeEnd, timeStart, Samples, Window=6):
    intervals = (Samples / Window) / (timeEnd - timeStart)
    timeAxis = np.ones((Samples / Window))
    someYs = []
    prev = intervals
    for num in range(len(timeAxis)):
        timeAxis[num] = timeAxis[num] * intervals + prev
        prev = timeAxis[num]
    return timeAxis

def peak_rate_xs(timeEnd, timeStart, Samples):
    timeAxis = 0
    return timeAxis

#function to plot the filtered array over the time steps
def plot_filt_array(timeAxis, filtArray, title='none', xlabel='none',ylabel='none'):
    plt.plot(timeAxis[len(timeAxis)-len(filtArray):], filtArray)
    plt.title(str(title))
    plt.xlabel(xlabel)
    plt.xlabel
    plt.ylabel(ylabel)
    plt.show()

#function to normalize summed data
def normalize_Sums(someList):
    newList = [x**2 for x in someList]
    divisor = math.sqrt(np.sum(newList))
    normList = [x/divisor for x in someList]
    return normList

#function to z-norm data
def znorm_Sums(someList):
    #First need standard deviation of set
    #First get mean
    #zMean = np.mean(someList)
    #stdDev = 
    return stat.zscore(someList)

#Function for getting a moving average
def my_moving_avg(someList, windowLength):
    myWeights = np.repeat(1.0, windowLength) / windowLength
    sma = np.convolve(someList, myWeights, 'valid')
    return sma

if __name__ == "__main__":
    # Help Options
    if((len(sys.argv) >= 1) and (str(sys.argv[1]) == '-h')):
        print(str(sys.argv[0]) + " <'pickle.pkl'> <['child', 'adult', 'infant']> <'xsmooth'>")
        sys.exit(2)

    # Breathing Rate Constants
    # Not currently used
    newbornRate = 44	# Breaths per minute
    infantRate = 40	# Breaths per minute
    childRate = 30	# Breaths per minute
    adultRate = 20	# Breaths per minute
    
    #samples = 1800
    myArrayOfImages = []	# Original Data
    MOG = []			# Data result of applying MOG to original
    summations = []
    summationsG = []
    sumMOG = []
    foreground = [] #after subtracting background
    subtraction = []
    subtractionG = []
    subMOG = []
    movingAvg = []
    curSamp = 1	# Sample count for keeping track of when next FFT is calculated
    fftWindow = []
    fftRate = []
    filtArray = []
    timeStart = t.time()

    # Load saved lists of images over 512 samples (17 secs)
    #inputFile = open('steve_1800.pkl','rb')
    inputFile = open(str(sys.argv[1]), 'rb')
    myArrayOfImages = pickle.load(inputFile)
    inputFile.close()

    # Samples
    samples = len(myArrayOfImages)

    #Create a background subtractor
    fgbg = cv2.BackgroundSubtractorMOG()

    fftCount = 0

    # Main loop for going through each saved frame and applying the
    # appropriate filter.
    for imageNum in range(len(myArrayOfImages)):
        MOG.append(fgbg.apply(myArrayOfImages[imageNum]))
	filtArray.append(filt.gaussian_filter(myArrayOfImages[imageNum],sigma=5))
	subtractionG.append(filtArray[imageNum]-myArrayOfImages[imageNum])
        summationsG.append(np.absolute(np.sum(subtractionG[len(subtractionG)-1])))
	subMOG.append(MOG[imageNum] - myArrayOfImages[imageNum])
	sumMOG.append(np.absolute(np.sum(subMOG[len(subMOG)-1])))
        summations.append(np.absolute(np.sum(myArrayOfImages[imageNum])))
	
######################### Calculation real-time breathing rate ##############
	
    # Calc breathing rate while data comes in
    # TO DO



######################### End real-time breathing rate ###########################

        # quit program when 'esc' key is pressed
        k = cv2.waitKey(5) & 0xFF
        if k == 27:
            break
    timeEnd = t.time()
    print("Length of fftRate is = " + str(len(fftRate)))

###################################### TEMP TIMES ###############################
    timeEnd = (samples / 30)    # Used for current simulation of real-time
    timeStart = 0
##################################################################################
    
    # Create norms
    normMOG = normalize_Sums(sumMOG)
    # Create zscored lists
    mogZ = znorm_Sums(sumMOG)
    print("Size of subtractions = " + str(len(subtraction)))
    timeAxis = get_timeAxis(timeEnd, timeStart, len(myArrayOfImages))
    print("Length of time axis = " + str(len(timeAxis)))
    print("Total time taken is " + str(timeEnd - timeStart))
    
    #Plot MOG results
    plt.subplot(231), plt.imshow(MOG[0],'gray'),plt.title('MOG first')
    plt.subplot(232), plt.imshow(MOG[len(MOG)-1],'gray'),plt.title('Last MOG')
    plt.imshow
    plt.show()
    
    #Plot subMOG results
    plt.subplot(231), plt.imshow(subMOG[0],'gray'),plt.title('subMOG first')
    plt.subplot(232), plt.imshow(subMOG[len(subMOG)-1],'gray'),plt.title('Last subMOG')
    plt.imshow
    plt.show()

    #plot_filt_array(timeAxis, movingAvg, "Moving Average")

    #plot MOG
    plot_filt_array(timeAxis, sumMOG, "MOG", 'Time over ' + str(timeEnd) + ' seconds','Magnitude')

    #plot MOG znorm
    plot_filt_array(timeAxis, mogZ, "MOG", 'Time over ' +str(timeEnd) + ' seconds','Magnitude')

################ Determine the window size of moving average ###################
    MA = 30
    if(len(sys.argv) >= 4):
        if(str(sys.argv[3]) == 'xsmooth'):
	    MA = 60
    mogSMA = my_moving_avg(sumMOG, MA)	#30 point moving avg
    modMogSMA = mogSMA % 2048	# Modulo 11 bit value since pixel intensity is 11 bit
    #plot MOG znorm
    plot_filt_array(timeAxis, mogSMA, "MOG with SMA", 'Time over ' +str(timeEnd) + ' seconds','Magnitude')

################################ LOCAL MIN AND MAX STUFF ########################
    window = 60
    if(len(sys.argv) >= 3):
        if(str(sys.argv[2]) == 'child'): window = 15
        elif(str(sys.argv[2]) == 'adult'): window = 60	# 60 is ok if there isn't fast movement
        elif(str(sys.argv[2]) == 'infant'): window = 10 # Good for super short breaths
    localMaxes = sig.argrelextrema(mogSMA, np.greater, order=window)[0]
    localMins = sig.argrelextrema(mogSMA, np.less, order=window)[0]
    localMaxes = localMaxes.tolist()
    localMins = localMins.tolist()
    tempMax = localMaxes
    tempMin = localMins
    print("Local maxes is ", localMaxes)
    print("local mins is ", localMins)

    print(len(sumMOG))
    print(len(localMaxes) + len(localMins))
    
    # Need to reverse since using pop method
    localMaxes.reverse()
    localMins.reverse()

    # Combine and sort. . .
    maxesCombined = []
    maxesCombined = localMaxes + localMins
    maxesCombined.sort()


    #ys = [ sumMOG(x) for x in both]
    ys = []
    xs = []
    # Get y axis and x axis for max and mins
    for x in maxesCombined:
        #print(x)
        ys.append(mogSMA[x])
        #xs.append(
    xs = get_maxmin_xs(timeEnd, timeStart, samples, maxesCombined, (len(timeAxis)-len(mogSMA)))
    intervals = samples / (timeEnd - timeStart)
    #xs = [x*intervals for x in maxesCombined]

############################ END LOCAL MIN AND MAX STUFF ########################

####### Calc breathing rate based on peaks of local maxes ###########################
    minCount = len(localMins)
    maxCount = len(localMaxes)
    #Maxes = localMaxes
    Maxes = localMaxes + localMins
    Maxes.sort()
    #Maxes.reverse()

    breathRate = [0]	# Start at zero since won't have two
    prev = 0
    for x in Maxes:	# peaks to calc difference yet
        if prev == 0:
            prev = x
	else:
	    print("x is " +str(x) +" prev is " +str(prev) + " equals " +str(x - prev))
	    result = (60 / (float((x - prev)) / float(15)))	# 15 for half second results
	    prev = x
	    breathRate.append(result)
    breathRateXs = range(0, (len(breathRate)*2), 2)  # 2 if using half second intervales  
    print("leng of breath rate = " +str(len(breathRate)))
    print("leng of breath xs = " +str(len(breathRateXs)))
    print("breath rate ys = " , breathRateXs)
    print("Breath rate values = ", breathRate)
############################### End Breathing rate and peaks ##########################

############################### Calculate baseline ###################################

    # Calculate baseline based on height of last 3 peaks and troughs
    # In this case, we will use first 3 for prototyping
    baselineValues = []
    baseLineValue = 0
    try:
        for x in range(3):
	    baselineValues.append(((mogSMA[tempMax[x]] - mogSMA[tempMin[x]]) / 2 ) + mogSMA[tempMin[x]])
        baseLineValue = (sum(baselineValues) / 3)
    except:
        print("Indexes out of range!!!!")

############################## End Calc baseline #####################################

    # The remaining section is just for plotting results.
    # This can obviously be removed or altered.
    plot_lists(myArrayOfImages, MOG, subMOG)
    print("Time axis length is " + str(len(timeAxis)))
    plt.plot(timeAxis[len(timeAxis)-len(mogSMA):], mogSMA)
    #plt.plot(timeAxis[len(timeAxis)-len(sumMOG):], sumMOG)
    plt.plot(xs, ys, 'ro')
    plt.plot([len(timeAxis)-len(mogSMA), timeAxis[len(timeAxis)-1]], [baseLineValue, baseLineValue], 'r--')
    plt.title("SMA of breathing cycles")
    plt.xlabel("Time in milliseconds")
    plt.ylabel("Amplitude based on 11 bit sum")
    plt.legend(["Breathing Signal", "Local Max / Min", "Baseline"])
    plt.show()
    print("length of xs = " + str(len(xs)))
    print("length of mogSMA = " + str(len(mogSMA)))
    print("length of sumMOG = " + str(len(sumMOG)))
    plt.plot(breathRateXs, breathRate, '--')
    plt.title("Breathing rate of patient")
    plt.xlabel("Time in seconds")
    plt.ylabel("Breaths per minute")
    plt.legend(["Breathing Rate"])
    plt.show()
    
    plt.plot(timeAxis, sumMOG)
    plt.title("Data After Subtraction")
    plt.xlabel("Time in milliseconds")
    plt.ylabel("Amplitude based on 11 bit sum")
    plt.legend(["Respiratory Signal"])
    plt.show()

    # Plot orig data
    plt.plot(timeAxis, summations)
    plt.title("Original Data")
    plt.xlabel("Time in milliseconds")
    plt.ylabel("Amplitude based on 11 bit sum")
    plt.legend(["Respiratory Signal"])
    plt.show()



