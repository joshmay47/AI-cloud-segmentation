import pickle
import glob
from os.path import exists
from time import sleep
import h5py
import scipy
from netCDF4 import Dataset
from numba import vectorize, njit,prange
from matplotlib import animation
import matplotlib.pyplot as plt
import numpy as np
import ai
import datetime

def getChunk(tcLat,tcLon,file,min30=0, size = 1024):
    """
    ---------------------------------------------------------------------------
    tcLat = latitude of the centre of the TC at a particular time
    tcLon = longitude of the centre of the TC at a particular time
    file = .nc4 file corresponding to the particular time
    min30 = 0 or 1, if it is an even or half hour
    size = determines the size of the chunk, 1024 is default

    Returns a size x size chunk of the .nc4 satellite image centered at
    coordinates: tcLat,tcLon. Can be given for an hourly or half hourly
    snapshot.
    ---------------------------------------------------------------------------
    """
    data = file['Tb']
    lons = file['lon'][:]
    lats = file['lat'][:]
    tcLati =  np.searchsorted(lats,tcLat)
    tcLoni =  np.searchsorted(lons,tcLon)
    tcLonMax = len(lons)
    tcLatMax = len(lats)
    overreachDown = False
    overreachUp = False
    half = size//2

    # Too far up
    if tcLati < half:
        overreachUp = True
        fill = np.zeros((half-tcLati,size))-9999.0

    # Too far down
    elif tcLati+half > tcLatMax:
        overreachDown = True
        fill = np.zeros((tcLati+half-tcLatMax,size))-9999.0

    # Too far left
    if tcLoni < half:
        if overreachUp: # ^\ :)
            return np.vstack((fill,np.hstack((data[min30,:tcLati+half,tcLonMax-half+tcLoni:],data[min30,:tcLati+half,:tcLoni+half]))))
        if overreachDown:# v/
            return np.vstack((np.hstack((data[min30,tcLati-half:,tcLonMax-half+tcLoni:],data[min30,tcLati-half:,:tcLoni+half])),fill))
        # <-
        return np.hstack((data[min30,tcLati-half:tcLati+half,tcLonMax-half+tcLoni:],data[min30,tcLati-half:tcLati+half,:tcLoni+half]))

    # Too far right
    if tcLoni+half > tcLonMax:
        if overreachUp: # /^ :)
            return np.vstack((fill,np.hstack((data[min30,:tcLati+half,tcLoni-half:],data[min30,:tcLati+half,:tcLoni+half-tcLonMax]))))
        if overreachDown: # \v
            return np.vstack((np.hstack((data[min30,tcLati-half:,tcLoni-half:],data[min30,tcLati-half:,:tcLoni+half-tcLonMax])),fill))
        # ->
        return np.hstack((data[min30,tcLati-half:tcLati+half,tcLoni-half:],data[min30,tcLati-half:tcLati+half,:tcLoni+half-tcLonMax]))

    if overreachUp: # ^ :)
        return np.vstack((fill,data[min30,:tcLati+half,tcLoni-half:tcLoni+half]))
    if overreachDown: # v
        return np.vstack((data[min30,tcLati-half:,tcLoni-half:tcLoni+half],fill))
    #
    return data[min30,tcLati-half:tcLati+half,tcLoni-half:tcLoni+half]

def findIndex(sid,ibtracs):
    """
    ---------------------------------------------------------------------------
    sid     = storm identification (IBTrACS Serial ID), eg: 2008278N13261
    ibtracs = IBTrACS dataset

    Returns index used in the ibtracs dataset that's corresponding to that sid
    ---------------------------------------------------------------------------
    """
    sids = ibtracs['sid'][:].data
    index = 0
    maxStorm = len(sids)
    found = False
    while (index < maxStorm) and (found is False):
        storedSid = sids[index]
        found = True
        for i, chara in enumerate(storedSid):
            try:
                if int(sid[i]) != int(chara):
                    found = False
                    break
            except ValueError: # when it comes up to N or S, ignore it
                pass
        index += 1

    if found:
        return index-1
    raise Exception("There isn't a TC with that SID in the IBTrACS dataset")

def getData(sid,dataset):
    """
    ---------------------------------------------------------------------------
    sid     = storm identification (IBTrACS Serial ID), eg: 2008278N13261
    dataset = IBTrACS file

    Returns a dictionary containing the name, category, latitude and longitude
    through time, and the time stamps that belong to those location recordings
    ---------------------------------------------------------------------------
    """

    ibtracs = Dataset(dataset,"r")
    # find the index corresponding to the sid
    index = findIndex(sid,ibtracs)

    # Get name, complicated because of the way it is stored
    rawName = ibtracs['name'][index].data
    name = ''
    for charRaw in rawName:
        name += charRaw.decode("utf-8")


    #Get duration of TC (until there is no data left)
    lats = ibtracs['lat'][index].data
    maxIndex = 0
    while lats[maxIndex] != -9999:
        maxIndex += 1

    # Get time stamps, has the structure of YYYY-MM-DD HH:MM:SS,
    # access HH:MM with <dict>['times'][<index>][11:16]
    timeRaws = ibtracs['iso_time'][index].data[0:maxIndex]
    times = []
    for timeRaw in timeRaws:
        time = ''
        for charRaw in timeRaw:
            time += charRaw.decode("utf-8")
        times.append(time)
    
    #Get other values
    lats = lats[0:maxIndex]
    lons = ibtracs['lon'][index].data[0:maxIndex]
    sshs = ibtracs['usa_sshs'][index].data[0:maxIndex]
    wmo_wind = ibtracs['wmo_wind'][index].data[0:maxIndex]
    usa_wind = ibtracs['usa_wind'][index].data[0:maxIndex]
    basinRaw = ibtracs['basin'][index].data[0]
    basin = basinRaw[0].decode("utf-8")+basinRaw[1].decode("utf-8")

    return {
        "name": name,
        "sshs": removeInvalidTimes(sshs,times),
        "wmo_wind": removeInvalidTimes(wmo_wind,times),
        "usa_wind": removeInvalidTimes(usa_wind,times),
        "times": removeInvalidTimes(times,times),
        "lats": removeInvalidTimes(lats,times),
        "lons": removeInvalidTimes(lons,times),
        "basin": basin
        }

def getTimeIndexes(tc):
    """
    ---------------------------------------------------------------------------
    tc     = tropical cyclone, in the format from getData

    Returns a list, first element is the starting time (0), each successive
    element is how many 30 minute intervals have been between that timestamp
    and the start. Is used for interpolating between lat and long locations
    ---------------------------------------------------------------------------
    """
    # Converts hours and minutes in the times entry in tc to values
    rawTimes = []
    for time in tc['times']:
        rawTimes.append(2*int(time[11:13])+int(time[14:16])//30)
    # The values go down when it enters a new day, a new day is detected when
    # the next value is less than the previous value. A day has 48 30min slots
    # this then gets added to the rawTimes array to keep the days considered
    # as 24 hour gaps
    offsets = [None]*len(rawTimes)
    offsets[0] = 0
    offset = 0
    for rawTimeIndex in range(1,len(rawTimes)):
        if rawTimes[rawTimeIndex] < rawTimes[rawTimeIndex-1]:
            offset += 48
        offsets[rawTimeIndex] = offset
    return [sum(x)-rawTimes[0] for x in zip(rawTimes, offsets)]

def interpolator(stamps, vals):
    """
    ---------------------------------------------------------------------------
    stamps  = Indexes the values are meant to be at (needs to be an ascending, always positive list)
    vals    = Values located at the indexes, identical length as stamps

    Returns a list with interpolation between values set by vals, the location
    of these values are defined by stamps. The values in between are linearly
    interpolated
    ---------------------------------------------------------------------------
    """
    interVals = np.zeros((stamps[-1]+1))
    for i in range(len(stamps)-1):
        run = stamps[i+1]-stamps[i]
        rise = vals[i+1]-vals[i]
        gradient = rise/run
        for j in range(run+1):
            interVals[stamps[i]+j]=vals[i]+j*gradient
    return interVals

def getSid(name,year,dataset,multiple=False):
    """
    ---------------------------------------------------------------------------
    name = Name of the TC (str)
    year = Year of the TC (str)
    dataset = IBTrACS dataset

    Returns sid that matches the name and year of the TC
    ---------------------------------------------------------------------------
    """
    ibtracs = Dataset(dataset,"r")
    names = ibtracs['name'][:].data
    index = 0
    maxStorm = len(names)
    found = False
    foundSids = []
    while (index < maxStorm) and (found is False):
        rawName = names[index]
        storedName = "".join(map(lambda name: name.decode("utf-8"),rawName))
        if len(name) != len(storedName):
            found = False
            index += 1
            continue
        found = True
        for i, chara in enumerate(name):
            if chara != storedName[i]:
                found = False
                break
        if found:
            rawSid = ibtracs['sid'][index].data
            sid = "".join(map(lambda sid: sid.decode("utf-8"),rawSid))
            if sid[0:4] == str(year):
                if not multiple:
                    return sid
                foundSids.append(sid)
            else:
                found = False
        index += 1
    if not multiple:
        raise Exception("There isn't a TC with that name and year in the IBTrACS dataset")
    return foundSids

def giffify(array, name, vMin = None, vMax = None):
    """
    ---------------------------------------------------------------------------
    array = array of images to be turned into a gif
    name = save name for the gif
    vMin and vMax = values that the gifs must be between

    Creates a gif of an array that is passed in.
    ---------------------------------------------------------------------------
    """
    fig = plt.figure(figsize=(10,10))
    altCmap = plt.get_cmap("gray_r").copy()
    altCmap.set_bad(color="r",alpha=1.)
    print("Creating Frames...")
    frames = [[plt.imshow(img, animated=True, vmin = vMin, vmax = vMax, cmap=altCmap)] for img in array]
    print("Creating Animation...")
    ani = animation.ArtistAnimation(fig, frames, interval=50)
    print("Saving Animation...")
    ani.save(name+'.gif')
    print("Complete.")

def giffify2(array1,array2,title1,title2,name):
    """
    ---------------------------------------------------------------------------
    array1 = first array of images to be used
    array2 = second array of images to be used
    title1 = first set title
    title2 = second set title
    name = save name for the gif
    vMin and vMax = values that the gifs must be between

    Creates a gif of an arrays that are passed in.
    ---------------------------------------------------------------------------
    """
    fig = plt.figure(figsize=(10,5))
    ax1 = fig.add_subplot(121)
    ax1.set_title(title1, fontsize = 20.0)
    ax1.axis("off")
    ax2 = fig.add_subplot(122)
    ax2.set_title(title2, fontsize = 20.0)
    ax2.axis("off")
    frames = []
    print("Creating Frames...")
    for (image1, image2) in zip(array1,array2):
        im1 = ax1.imshow(image1, cmap = "gray_r", vmin = 90, vmax = 210)
        im2 = ax2.imshow(image2, cmap = "gray", vmin = 0, vmax = 2)
        frames.append([im1, im2])
    print("Creating Animation...")
    ani = animation.ArtistAnimation(fig, frames, interval=50)
    print("Saving Animation...")
    ani.save(name+'.gif')
    print("Complete.")

def giffify3(array1,array2,array3,title1,title2,title3,name,cmaps=("viridis","viridis","viridis"), vMin=None,vMax=None):
    """
    ---------------------------------------------------------------------------
    array1 = first array of images to be used
    array2 = second array of images to be used
    array3 = third array of images to be used
    title1 = first set title
    title2 = second set title
    title3 = third set title
    name   = save name for the gif
    cmaps  = cmap for the three gifs as a tuple
    vMin and vMax = values that the gifs must be between

    Creates a gif of an arrays that are passed in.
    ---------------------------------------------------------------------------
    """
    fig = plt.figure(figsize=(15,5))
    ax1 = fig.add_subplot(131)
    ax1.set_title(title1, fontsize = 20.0)
    ax1.axis("off")
    ax2 = fig.add_subplot(132)
    ax2.set_title(title2, fontsize = 20.0)
    ax2.axis("off")
    ax3 = fig.add_subplot(133)
    ax3.set_title(title3, fontsize = 20.0)
    ax3.axis("off")
    frames = []
    print("Creating Frames...")
    for (image1, image2, image3) in zip(array1,array2,array3):
        im1 = ax1.imshow(image1, vmin = vMin, vmax = vMax,cmap=cmaps[0])
        im2 = ax2.imshow(image2, vmin = vMin, vmax = vMax,cmap=cmaps[1])
        im3 = ax3.imshow(image3, vmin = vMin, vmax = vMax,cmap=cmaps[2])
        frames.append([im1, im2, im3])
    print("Creating Animation...")
    ani = animation.ArtistAnimation(fig, frames, interval=50)
    print("Saving Animation...")
    ani.save(name+'.gif')
    print("Complete.")

def process(fname, satPath, h5Path, saveName = "processed", size = 1024, gif = False, startFrame=0, endFrame="end", downSample = 2, filling=True, threshold = False, showSpeed=True):
    """
    ---------------------------------------------------------------------------
    fname       = filename of the .h5 label file (dont include .h5 in the string)
    size        = dimensions of the output image
    path        = location of the files
    gif         = when True, creates a gif showing the images with labels
                    overlayed. False by default
    saveName    = the gif is saved as <saveName>.gif
    careful     = if process fails the accellerated method, set to True to try
                    again (default: False)
    startFrame  = First frame that is desired to be saved (default: 0)
    endFrame    = Last frame that is desired to be saved (default: end)
    downSample  = How the image should be downscaled, e.g downsample of 2
                    reduces the 1024x1024 output to 512x512 (default: 2)
    filling     = If the program fills missing data from the previous entry
                    (default: True)
    threshold   = If a number, only allows frames where the sustained 1-minute
                    windspeed is over that threshold knots. Overrides
                    startFrame and endFrame (default: False)
    showSpeed   = If True, the gif shows the speed of the TC through time, this
                    also splits the images and labels

    Creates
    An array of 1 byte arrays, which elements contain (labels):
        a 0 when it is not part of a TC
        a 1 where it is part of a TC.
    An array of satellite images that align with the labels array

    If input gif = True, also produces a gif showing satellite images, labels
    and intensity of the storm.
    ---------------------------------------------------------------------------
    """
    tc = getData(fname[0:13])
    if threshold:
        try:
            startFrame = np.where(tc["usa_wind"] > threshold)[0][0]*6
            endFrame = (np.where(tc["usa_wind"] > threshold)[0][-1]-1)*6
        except IndexError:
            startFrame = 0
            endFrame = 0
    print("Found Tropical Cyclone:",tc['name'])
    rawTimes = getTimeIndexes(tc)
    files = list(sorted(glob.glob(f"{satPath}*")))
    # Find the first file
    print("Detecting files...")
    found = False
    for attempt in range(len(rawTimes)):
        time = tc['times'][attempt]
        timeString = 'merg_'
        timeString += time[0:4]+time[5:7]+time[8:10]+time[11:13]
        timeString += '_4km-pixel.nc4'
        if exists(satPath+timeString):
            found = True
            break
    if not found:
        raise FileNotFoundError("There isn't a file that has that is in the timespan of the TC")
    if attempt != 0:
        print("WARNING:\tCouldn't find file that would reflect the first time slot in TC...")
        print("\t\t\tUsing "+timeString+" which has lost",rawTimes[attempt],"location entries")
    startFileInd = files.index(satPath+timeString)
    start = attempt

    # find the last file
    for attempt in reversed(range(len(rawTimes))):
        time = tc['times'][attempt]
        timeString = 'merg_'
        timeString += time[0:4]+time[5:7]+time[8:10]+time[11:13]
        timeString += '_4km-pixel.nc4'
        if exists(satPath+timeString):
            break
    if attempt != len(rawTimes)-1:
        print("WARNING:\tCouldn't find file that would reflect the last time slot in TC...")
        print("\t\t\tUsing "+timeString+" which has lost",rawTimes[-1]-rawTimes[attempt],"location entries")
    endFileInd = files.index(satPath+timeString)
    end = attempt

    # Find file names, and interpolate the latitude and longitude values for the tc
    tcFileNames = files[startFileInd:endFileInd+1] # Last entry, only use the first time step
    interpLats = interpolator(rawTimes[start:end+1],tc['lats'][start:end+1])[rawTimes[start]:]
    interpLons = interpolator(rawTimes[start:end+1],tc['lons'][start:end+1])[rawTimes[start]:]

    with h5py.File(h5Path+fname+".h5", "r") as f:
        data = np.array(f[list(f.keys())[0]], dtype=np.uint8)

    print("Creating processed labels...")
    full = np.ones((2*len(tcFileNames)-1,size//downSample,size//downSample), dtype = np.uint8)
    satellite = np.zeros((2*len(tcFileNames)-1,size//downSample,size//downSample), dtype = np.uint8)
    intensities = np.interp(np.arange(len(tc["sshs"])*6),np.arange(len(tc["sshs"])*6, step=6),tc["usa_wind"])

    lastUpdate = 0
    firstTime = rawTimes[start]
    offset = (1100-size)//2
    if endFrame == "end":
        endFrame = 2*len(tcFileNames)-1
    ###
    print(endFrame,len(interpLats),len(interpLons))
    ###
    for i in range(2*len(tcFileNames)-1):### normally with -1
        if 50*(i/len(tcFileNames))-lastUpdate > 10:
            print(int(50*(i/len(tcFileNames))),"% complete")
            lastUpdate += 10
        if i < startFrame or i > endFrame:
            continue
        frame = data[i+firstTime][offset:1100-offset,offset:1100-offset]
        if 4 in frame:
            processed = transferFounds(frame,4)
        elif 2 in frame:
            processed = transferFounds(frame,2)
        else:
            processed = np.zeros((size,size))

        # Retrieve file
        if i%2 == 0:
            try:
                satellitenc4 = h5py.File(tcFileNames[i//2],"r")
            except OSError as e:
                raise RuntimeError(str(tcFileNames[i//2][-29:]+" has invalid data (OSError).")) from e

        # Set no data pixels to 0
        try:
            chunk = getChunk(interpLats[i], interpLons[i], satellitenc4, i%2, size)
            mask = chunk==-9999
            chunk[mask]=100
            chunk = np.array(chunk-100)
            chunk = meanDownsample(chunk)
            processed = processed[::downSample,::downSample]
            satellite[i] = chunk
        except RuntimeError as e:
            raise RuntimeError(str(tcFileNames[i//2][-29:]+" has invalid data (runtime).")) from e
        except OSError as e:
            raise RuntimeError(str(tcFileNames[i//2][-29:]+" has invalid data (OSError).")) from e
        del chunk
        full[i] = processed
    if filling:
        full = retainDataForMissingData(full)
        satellite = retainDataForMissingData(satellite)
    if gif:
        print("Creating Gif...")
        if showSpeed:
            # Setup frames
            fig = plt.figure(figsize=(12,12))
            plt.suptitle(f"{getFullName(tc)} ({fname[0:4]})", fontsize = 25.0)
            ax1 = fig.add_subplot(212)
            ax1.set_title("Intensity", fontsize=20.0)
            createIntensityGraph(tc, ax1)
            ax2 = fig.add_subplot(221)
            ax2.set_title("Labels", fontsize = 20.0)
            ax2.axis("off")
            ax3 = fig.add_subplot(222)
            ax3.set_title("Images", fontsize = 20.0)
            ax3.axis("off")
            frames = []

            # Add data into frames
            for i in range(rawTimes[end]-rawTimes[start]):
                if i < startFrame or i > endFrame:
                    continue

                im1 = ax1.axvline(x=i, color='r')
                im2 = ax2.imshow(full[i], cmap = "gray", vmin=0, vmax=2)
                im3 = ax3.imshow(satellite[i], cmap = "gray_r", vmin = 90, vmax = 210)
                frames.append([im1, im2, im3])
        else:
            # Setup frames
            fig = plt.figure(figsize=(7.12,7.49))
            ax = fig.add_axes([0,0,1,0.95])
            ax.set_title(f"{getFullName(tc)} ({fname[0:4]})", fontsize=20.0)
            ax.axis("off")

            altCmap = plt.get_cmap("gray_r").copy()
            altCmap.set_bad(color="r",alpha=1.)
            frames = []

            # Add data into frames
            for i in range(rawTimes[end]-rawTimes[start]):
                if i < startFrame or i > endFrame:
                    continue
                im1 = ax.imshow(showPredictionsAux(satellite[i],full[i]))
                frames.append([im1])

        ani = animation.ArtistAnimation(fig, frames, interval=50)
        del frames
        print("Saving gif...")
        ani.save(f"processedAnalysis\\{fname[0:4]}\\{saveName}.gif",writer='pillow')
        del ani

    #del data
    #gc.collect()
    print("Complete.")
    return (full[startFrame:endFrame], satellite[startFrame:endFrame], intensities[startFrame:endFrame])

def loadProcessed(fileNameLabels, fileNameImages):
    """
    ---------------------------------------------------------------------------
    Reads <fileNameLabels>.pkl and <fileNameImages>.pkl
    ---------------------------------------------------------------------------
    """
    with open(f"{fileNameLabels}.pkl","rb") as pkl_file:
        processed = pickle.load(pkl_file)
    with open(f"{fileNameImages}.pkl","rb") as pkl_file:
        satellite = pickle.load(pkl_file)
    return (np.array(processed, dtype=np.uint8), np.array(satellite, dtype=np.uint8))

def saveProcessed(labels,fileNameLabels,images,fileNameImages):
    """
    ---------------------------------------------------------------------------
    Saves labels and images as <fileNameLabels>.pkl and <fileNameImages>.pkl
    respectively
    ---------------------------------------------------------------------------
    """
    with open(f"{fileNameLabels}.pkl", "wb") as pkl_file:
        pickle.dump(labels, pkl_file)
    with open(f"{fileNameImages}.pkl", "wb") as pkl_file:
        pickle.dump(images, pkl_file)

def iou(img1: np.ndarray, img2: np.ndarray,checking=True):
    """
    img1 = (binary) array of predicted labels
    im2 = (binary) array of true labels

    Returns a float indicating the IoU or Jaccard Distance between the two images.
    """
    # Do checks, always done at the start, but can be made false if already done
    if checking:
        img1 = np.squeeze(img1)
        img2 = np.squeeze(img2)
        img1size = np.shape(img1)
        img2size = np.shape(img2)
        if img1size != img2size:
            raise Exception(f"Images must be the same shape. There is {img1size} and {img2size}.")
        if not ((img1==0) | (img1==1)).all():
            raise ValueError("img1 must be binary")
        if not ((img2==0) | (img2==1)).all():
            raise ValueError("img2 must be binary")
    # Recursive for multiple images
    if img1.ndim == 3:
        return np.array([iou(image,label,False) for (image,label) in zip(img1,img2)])
    # Perform calculation
    overlap = np.add(img1,img2)
    ones = np.count_nonzero(overlap == 1)
    twos = np.count_nonzero(overlap == 2)
    try:
        return twos/(ones+twos)
    except ZeroDivisionError:
        print("Empty frame found")
        return 1.0

def checkFiles(year, path):
    """
    ---------------------------------------------------------------------------
    year = year for files that needs to be checked
    path = location of the files

    Checks the integrity of .nc4 files for the year specified. Prints all files
    that are missing or corrupted. Also creates a file that has the required
    download paths for missing or corrupted files.
    ---------------------------------------------------------------------------
    """
    #numberOfFilesInDay = lambda month,day: len(fnmatch.filter(os.listdir(path), "merg_"+str(year)+str(month)+str(day)+"*"))
    hours  = [f'{hour:0>2}'    for hour in range(24)]
    days   = [f'{day+1:0>2}'   for day in range(31)]
    months = [f'{month+1:0>2}' for month in range(12)]
    numberOfDaysInMonth = [31,28,31,30,31,30,31,31,30,31,30,31]
    mergIRAccess = f"https://data.gesdisc.earthdata.nasa.gov/data/MERGED_IR/GPM_MERGIR.1/{year}"
    if year%4 == 0:
        numberOfDaysInMonth[1] = 29
    failedFiles = []
    for month in months:
        print(f"Checking Month: {month}")
        for day in days[:numberOfDaysInMonth[int(month)-1]]:
            for hour in hours:
                fileName = f"merg_{year}{month}{day}{hour}_4km-pixel.nc4"
                url = f"{mergIRAccess}/{sum(numberOfDaysInMonth[:int(month)-1])+int(day):03}/{fileName}"
                try:
                    satellitenc4 = Dataset(f"{path}\\{fileName}","r")
                    _ = satellitenc4['Tb'][0,:,:]
                    _ = satellitenc4['Tb'][1,:,:]
                    _ = satellitenc4['lon'][:]
                    _ = satellitenc4['lat'][:]
                except FileNotFoundError:
                    print(f"merg_{year}{month}{day}{hour}_4km-pixel.nc4 is missing")
                    failedFiles.append(url)
                except RuntimeError:
                    print(f"merg_{year}{month}{day}{hour}_4km-pixel.nc4 has corrupted data")
                    failedFiles.append(url)
                except OSError:
                    print(f"merg_{year}{month}{day}{hour}_4km-pixel.nc4 has corrupted data")
                    failedFiles.append(url)
    if failedFiles != []:
        with open(f"Failed files {year}.txt","w") as f:
            for failedFile in failedFiles:
                f.write(failedFile+"\n")
    print("File verification done.")

@vectorize
def transferFounds(num,goal):
    if num == goal:
        return 1
    return 0

@vectorize("uint8(uint8)")
def convertLabels(label):
    if label == 2:
        return 1
    return 0

@vectorize("float32(uint8)")
def convertImages(pixel):
    return (pixel-128)/128

@vectorize
def convertSpeeds(speed):
    if speed < 20:
        return 20
    return speed

def retainDataForMissingData(setOfData):
    """
    ---------------------------------------------------------------------------
    setOfData = set of data, could be images or labels

    Returns the set of data, where the parts that have missed data are retained
    from the previous frames data
    ---------------------------------------------------------------------------
    """

    padding = np.zeros((1,setOfData.shape[2],setOfData.shape[2]),dtype=np.uint8)
    setShifted = np.concatenate((padding,setOfData), axis=0)
    setPadded = np.concatenate((setOfData,padding), axis=0)
    setPadded[setPadded == 0] = setShifted[setPadded == 0]
    return setPadded[:-1]

def createIntensityGraph(tcData, figure):
    """
    ---------------------------------------------------------------------------
    tcData = TC dict created by getData()

    Plots an intensity graph for the tropical cyclone using the SSHS
    ---------------------------------------------------------------------------
    """
    length = len(tcData["sshs"])*6
    figure.plot(np.interp(np.arange(length),np.arange(length, step=6),tcData["usa_wind"]), label="USA_WIND")
    figure.plot(np.interp(np.arange(length),np.arange(length, step=6),tcData["wmo_wind"]), label="WMO_WIND")
    figure.set_ylim((0,200))
    figure.legend(loc="upper right")
    figure.set_ylabel("Maximum sustained wind speed (knots)")
    figure.set_xlabel("Frame")
    figure.grid(axis='y')
    return figure

def getFullName(tcData):
    """
    ---------------------------------------------------------------------------
    tcData = TC dict created by getData()

    Returns a string that contains the full name of the TC, dependant on
        intensity
    ---------------------------------------------------------------------------
    """
    maximum = max(tcData["sshs"])
    types = ["Tropical Depression ","Tropical Storm ","Category 1 ",
             "Category 2 ","Category 3 ","Category 4 ", "Category 5 "]
    addOnn = ""
    if maximum > 0:
        basin = tcData["basin"]
        if basin == "NA":
            addOnn = "Hurricane "
        elif basin == "EP":
            addOnn = "Hurricane "
        elif basin == "WP":
            addOnn = "Typhoon "
        elif basin == "NI":
            addOnn = "Cyclone "
        elif basin == "SI":
            addOnn = "Cyclone "
        elif basin == "SP":
            addOnn = "Cyclone "
        elif basin == "SA":
            addOnn = "Hurricane "
    if maximum < -1:
        return "Storm "+tcData["name"].title()
    return types[maximum+1]+addOnn+tcData["name"].title()

def runthrough(tcs, h5Path, task=None):
    """
    ---------------------------------------------------------------------------
    tcs = which TCs to be analysed, if int, run through the first int tcs.
            If an iterable, then run through those in that iterable
    h5path = where the labels are located
    task = task that needs to be completed. Can be one of the following:
        iouspeed = Creates an accuracy (IoU) and a speed array. Where the
                    IoU corresponds to the speed at the same time
        demonstrate = From the TCs selected. Extracts the images and predicts
                        according to the AI model selected in the module ai.
                        It is shown like in showPerformanceAlt().
        generate = Generates the images, labels and speeds from the TCs selected

        select = Selective version of generate with built-in saving, prompts the
                user if they want to save that TC.

    Adjustable function, that runs through the TCs for the year downloaded with
        a task defined by the user.
    ---------------------------------------------------------------------------
    """
    if task is None:
        return "You need to specify a task, being 'iouspeed', 'demonstrate' or 'generate'"
    if task not in ["iouspeed","demonstrate","generate","select"]:
        return "Invalid task, needs to be 'iouspeed', 'demonstrate', 'generate' or 'select'"
    if isinstance(type,int):
        tcs = range(tcs)
    files = [file[len(h5Path):-3] for file in sorted(glob.glob(f"{h5Path}*.h5"))]
    # Retrieve the filenames for the TC labels
    if task == "iouspeed":
        ious = []
        speedsMaster = []

    elif task == "generate":
        labelsMaster = [None]*len(tcs)
        imagesMaster = [None]*len(tcs)
        speedsMaster = [None]*len(tcs)
        index = 0

    # Load ai
    for i, filename in enumerate(files):
        if i not in tcs:
            continue
        if task == "iouspeed":
            labels, images, speeds = process(filename)
            print("Predicting...")
            predictions = ai.predict(images)
            speeds = convertSpeeds(speeds)
            for (label, prediction, speed) in zip(labels,predictions,speeds):
                speedsMaster.append(speed)
                ious.append(iou(label,prediction))

        elif task == "demonstrate":
            labels, images, speeds = process(filename)
            images3d,sel = selectionsTo3(images, np.ones((len(images))),temporalDistance=3)
            print("Predicting...")
            predictions = ai.predict(images3d)
            for image,label,prediction in zip(images3d,labels[sel],predictions):
                showPredictions(image, label,prediction)
                plt.show()

        elif task == "generate":
            labels, images, speeds = process(filename, threshold=35, filling=False)
            labelsMaster[index] = labels
            imagesMaster[index] = images
            speedsMaster[index] = speeds
            index += 1

        elif task == "select":
            splitUp = filename.split('_')
            if splitUp[1] == "NOT":
                tcName = splitUp[0][-5:]
            else:
                tcName = splitUp[1].title()
            year = splitUp[0][0:4]

            labels, images, speeds = process(filename, threshold=35, filling=True,gif=True,saveName=tcName,showSpeed=False)
            
            try:
                labels[0] = labels[1]
            except IndexError:
                print('There was not a point that was over 35 knots')
                print('-----------------------------------------------------------\n')
                continue

            sleep(1)
            keeping = int(input(f"Keep {tcName}? (3 for yes (all), 2 for yes (crop), 1 for yes (select), 0 for no)" ))
            if keeping == 0:
                print(f"Not keeping any of {tcName}")
                print('-----------------------------------------------------------\n')
                continue
            if keeping == 1:
                selections = manualSelect(images,labels)
                print(f"Keeping {100*np.count_nonzero(selections==1)/len(selections):.2f}% of {tcName}.")
            if keeping == 2:
                start = int(input("What should be the first index? "))
                end = int(input("What should be the last index? "))
                print("Predicting selections...")
                selections = (iou(ai.predict(images),labels) > 0.2).astype(bool)
                selections[:start] = False
                selections[end:] = False
                print(f"Keeping {100*np.count_nonzero(selections==1)/len(selections):.2f}% of {tcName}.")
            if keeping == 3:
                print(f"Keeping all of {tcName}")
                #selections = np.ones(len(labels),dtype=bool)
                print("Predicting selections...")
                selections = (iou(ai.predict(images),labels) > 0.2).astype(bool)
            with open(f"{year}/{getData(splitUp[0])['basin']}/{tcName}.pkl", "wb") as pkl_file:
                pickle.dump((images,labels,selections==1), pkl_file)
            print('-----------------------------------------------------------\n')
    if task == "iouspeed":
        return (speedsMaster,ious)
    if task == "demonstrate":
        return
    if task == "generate":
        return (labelsMaster[0], imagesMaster[0], speedsMaster[0])
    if task == "select":
        return

def display3(previousImage,currentImage,currentLabel,nextImage):
    """
    ---------------------------------------------------------------------------
    previousImage = satellite image of TC one step before focus
    currentImage = satellite image of TC at time of focus
    currentLabel = TC cloud label corresponding to currentImage
    nextImage = satellite image of TC one step after focus

    Displays Image with label as an outline, as well as the image before and
    after. Used to identify good labels for models using 3 images.
    ---------------------------------------------------------------------------
    """
    altCmap = plt.get_cmap("gray_r").copy()
    altCmap.set_bad(color="r",alpha=1.)

    fig, (ax1,ax2,ax3) = plt.subplots(3,1,sharex=True,figsize=(20,20))
    fig.tight_layout()

    ax1.set_ylabel("PREVIOUS", fontsize = 20.0)
    ax1.tick_params(which="both",bottom=False,left=False,labelleft=False)
    ax2.set_ylabel("CURRENT", fontsize = 20.0)
    ax2.tick_params(which="both",bottom=False,left=False,labelleft=False)
    ax3.set_ylabel("NEXT", fontsize = 20.0)
    ax3.tick_params(which="both",bottom=False,left=False,labelbottom=False,labelleft=False)

    current = showPredictionsAux(currentImage,currentLabel)

    ax1.imshow(previousImage, cmap = altCmap)
    ax2.imshow(current)
    ax3.imshow(nextImage, cmap = altCmap)
    plt.show()

def manualSelect(images,labels):
    """
    ---------------------------------------------------------------------------
    images = satellite images of TC
    labels = TC cloud labels corresponding to images

    Allows selection of which data should be included by showing plots made by
    display3(). Entering a 1 keeps the image. Entering a 2 removes it.
    Entering a 3 is undo. Returns a list of 1's and 2's.
    ---------------------------------------------------------------------------
    """
    size = len(images)
    selections = np.zeros(size)
    loc=0

    while loc<size:
        display3(images[(loc-1)%size],images[loc],labels[loc],images[(loc+1)%size])
        try:
            selection = int(input())
        except ValueError:
            print("Try again, there was an invalid input")
            continue
        if selection == 1:
            print(f"Retaining {loc}")
            selections[loc] = 1
            loc += 1
        elif selection == 2:
            print(f"Removing {loc}")
            selections[loc] = 2
            loc += 1
        elif selection == 3:
            print(f"Undo to {loc-1}")
            loc -= 1
        else:
            print("Try again, there was an invalid input")
    return selections


def selectionsTo3(images,selections,temporalDistance = 1):
    """
    ---------------------------------------------------------------------------
    images = satellite images of TC
    labels = TC cloud labels corresponding to images
    selections = generated by manualSelect()

    Creates a set of 3D images corresponding to images before, during and after
    selections made by manualSelect()
    ---------------------------------------------------------------------------
    """
    selections[:temporalDistance] = False
    selections[-temporalDistance:] = False
    unclears = [find_missing_centre(image) for image in images]
    imagesSelected = np.zeros((np.count_nonzero(selections),512,512,3), dtype=np.uint8)
    updatedSelections = []
    upto = 0
    for i,selection in enumerate(selections):
        if selection == 1:
            if not (unclears[i-temporalDistance] or unclears[i] or unclears[i+temporalDistance]):
                imagesSelected[upto]=np.dstack((images[i-temporalDistance],images[i],images[i+temporalDistance]))
                upto += 1
                updatedSelections.append(True)
            else:
                updatedSelections.append(False)
        else:
            updatedSelections.append(False)
        
    return imagesSelected[:upto], updatedSelections

def showPredictionsAux(image,label1,label2=None,label3=None):
    """
    ---------------------------------------------------------------------------
    image = satellite image of TC
    label1 = TC cloud label corresponding to image (version 1)
    label2 = TC cloud label corresponding to image (version 2, optional)
    label3 = TC cloud label corresponding to image (version 3, optional)

    Creates an array that is the same through all three channels excpet when
    it is the border of a label, each label gets its own colours:
        label1 = Red
        label2 = Blue
        label3 = Green
    This is used instead of showPredictions() when the display method wants to
    be altered and different from showPredictions().
    ---------------------------------------------------------------------------
    """
    if image.ndim==3:
        image=image[:,:,1]
    label2Used = not isinstance(label2,type(None))
    label3Used = not isinstance(label3,type(None))
    if not ((label1==0) | (label1==1)).all():
        raise ValueError("Label 1 must be binary.")
    if label2Used:
        if not ((label2==0) | (label2==1)).all():
            raise ValueError("Label 2 must be binary.")
        label2 = label2.astype(float)
    if label3Used:
        if not ((label3==0) | (label3==1)).all():
            raise ValueError("Label 3 must be binary.")
        label3 = label3.astype(float)

    image = image[::-1,:].astype(float)
    label1 = label1[::-1,:].astype(float)
    maxVal = np.max(image)
    minVal = np.min(image)
    image = (maxVal-image)/(maxVal-minVal) # reverse the image values with max at 0 and min at 1
    imageResult = np.dstack((image,image,image))

    edges1 = (label1-scipy.ndimage.binary_erosion(label1, iterations=1))==1
    imageResult[edges1] = np.array([1,0,0]) # label1 is red
    if label2Used:
        label2 = label2[::-1,:].astype(float)
        edges2 = (label2-scipy.ndimage.binary_erosion(label2, iterations=1))==1
        imageResult[edges2] = np.array([0,0,1]) # label2 is blue
    if label3Used:
        label2 = label3[::-1,:].astype(float)
        edges3 = (label3-scipy.ndimage.binary_erosion(label3, iterations=1))==1
        imageResult[edges3] = np.array([0,1,0]) # label3 is green
    return (255*imageResult).astype(np.uint8)

def showPredictions(image,label1,label2=None,label3=None):
    """
    ---------------------------------------------------------------------------
    image = satellite image of TC
    label1 = TC cloud label corresponding to image (version 1)
    label2 = TC cloud label corresponding to image (version 2, optional)
    label3 = TC cloud label corresponding to image (version 3, optional)

    Displays the satellite image that is grayscale excpet when it is the border
    of a label, each label gets its own colours:
        label1 = Red
        label2 = Blue
        label3 = Green
    ---------------------------------------------------------------------------
    """
    fig = plt.figure(figsize=(7.12,7.12))
    ax = fig.add_axes([0,0,1,1])
    ax.axis("off")
    ax.imshow(showPredictionsAux(image,label1,label2,label3))
    plt.show()

@njit(parallel=True)
def meanDownsample(image):
    out = np.zeros((512,512))
    for i in prange(512):
        row = i*2
        for j in prange(512):
            col = j*2
            divisor = 0
            tl = image[row,col]
            if tl != 0:
                divisor += 1
            tr = image[row,col+1]
            if tr != 0:
                divisor += 1
            bl = image[row+1,col]
            if bl != 0:
                divisor += 1
            br = image[row+1,col+1]
            if br != 0:
                divisor += 1
            if divisor == 0:
                divisor = 1
            out[i,j] = (tl+tr+bl+br)/divisor
    return out

@njit(parallel=True)
def fillSpeckledMissingData(array):
    """
    array = any array with pixels that have missing data and need to be filled in
    
    Atlernative to meanDownsample that doesn't reduce size, this works by picking
    the mean of the surrounding pixels,ignoring other missing pixels. This is not
    preffered to meanDownsample if possible.
    """
    out = np.zeros_like(array)
    for framei in prange(array.shape[0]):
        for rowi in prange(512):
            for coli in prange(512):
                if array[framei,rowi,coli] == 0:
                    lowRow = max(0,rowi-1)
                    lowCol = max(0,coli-1)
                    highRow = min(511,rowi+2)
                    highCol = min(511,coli+2)
                    chunk = array[framei,lowRow:highRow,lowCol:highCol]
                    total = 0
                    divisor = 0
                    for i in chunk.flatten():
                        if i != 0:
                            total += i
                            divisor += 1
                    divisor = max(1,divisor)
                    out[framei,rowi,coli] = total/divisor
                else:
                    out[framei,rowi,coli] = array[framei,rowi,coli]
    return out

def retrieveTCData(name,year,satPath):
    """
    ---------------------------------------------------------------------------
    name = The TC's name
    year = The year in which the TC began
    satPath = location of satellite images

    Finds the TC and retrieves relevant statistical data about its evolution
    including latitudes, longitudes, times, tcName, sid. This is used for
    analysis of the storm, not for processing.
    ---------------------------------------------------------------------------
    """
    sid = getSid(name.upper(),year,multiple=True)
    if len(sid) > 1:
        raise Exception("There are more than one TC of that type")
    sid = sid[0]
    tc = getData(sid)
    startFrame = np.where(tc["usa_wind"] > 35)[0][0]*6
    endFrame = (np.where(tc["usa_wind"] > 35)[0][-1]-1)*6
    rawTimes = getTimeIndexes(tc)
    # Find the first file
    for attempt in range(len(rawTimes)):
        time = tc['times'][attempt]
        timeString = 'merg_'
        timeString += time[0:4]+time[5:7]+time[8:10]+time[11:13]
        timeString += '_4km-pixel.nc4'
        if exists(satPath+timeString):
            break
    #start = attempt
    # find the last file
    for attempt in reversed(range(len(rawTimes))):
        time = tc['times'][attempt]
        timeString = 'merg_'
        timeString += time[0:4]+time[5:7]+time[8:10]+time[11:13]
        timeString += '_4km-pixel.nc4'
        if exists(satPath+timeString):
            break
    #end = attempt
    # Find file names, and interpolate the latitude and longitude values for the tc
    #interpLats = interpolator(rawTimes[start:end+1],tc['lats'][start:end+1])[rawTimes[start]:]
    #interpLons = interpolator(rawTimes[start:end+1],tc['lons'][start:end+1])[rawTimes[start]:]
    #intensities = np.interp(np.arange(len(tc["sshs"])*6),np.arange(len(tc["sshs"])*6, step=6),tc["usa_wind"])[:len(interpLats)]
    interpLats = interpolator(rawTimes,tc['lats'])
    interpLons = interpolator(rawTimes,tc['lons'])
    intensities = np.interp(np.arange(len(tc["sshs"])*6),np.arange(len(tc["sshs"])*6, step=6),tc["usa_wind"])[:len(interpLats)]
    fullTimes = []
    for time in tc['times']:
        day = time[:10]
        hour = int(time[11:13])
        for interpHour in range(hour,hour+3):
            fullTimes.append(f"{day} {interpHour:02}:00:00")
            fullTimes.append(f"{day} {interpHour:02}:30:00")
    fullTimes = np.array(fullTimes)

    return {'lats':interpLats[startFrame:endFrame],
            'lons':interpLons[startFrame:endFrame],
            'times':fullTimes[startFrame:endFrame],
            'speed':intensities[startFrame:endFrame],
            'name':name,
            'sid':sid}

def removeInvalidTimes(original,timeStrings):
    return np.array([originali for originali,timeStr in zip(original,timeStrings)\
                     if not int(timeStr[11:13])%3 != 0])
        
@njit
def find_missing_centre(image,threshold=2000):
    missing_found = 0
    for row in range(206,306):
        for col in range(206,306):
            if image[row,col] == 0:
                missing_found += 1
            if missing_found > threshold:
                return True
    return False

def getWindChunk(tcLat,tcLon,file,time):
    """
    ---------------------------------------------------------------------------
    tcLat = latitude of the centre of the TC at a particular time
    tcLon = longitude of the centre of the TC at a particular time
    file = .nc filename of the ERA5 data
    time = any integer indicating how many half hours after the first time step
            in the .nc file.

    Returns a 40 x 40 degree chunk of the .nc wind field centered at
    coordinates: tcLat,tcLon. Interpolates between two values usually.
    ---------------------------------------------------------------------------
    """
    def get_time_interpolated_wind(windnc4,direction,time,borders):
        north,east,south,west = borders
        if time%2 == 1: #interpolate
            windPrev = windnc4[direction][int(time/2),north:south,west:east][::-1,:]
            windNext = windnc4[direction][int(time/2)+1,north:south,west:east][::-1,:]
            return 0.5*(windPrev+windNext)
        return windnc4[direction][time//2,north:south,west:east][::-1,:]
    
    windnc4 = Dataset(file,"r")
    lons = windnc4['longitude'][:]
    lats = windnc4['latitude'][::-1]
    
    south = np.searchsorted(lats,-1*(tcLat-20))
    north = np.searchsorted(lats,-1*(tcLat+20))
    if abs(tcLon)<160: # Not over extending east or west
        west = np.searchsorted(lons,tcLon-20)
        east = np.searchsorted(lons,tcLon+20)
        horiz = get_time_interpolated_wind(windnc4,'u',time,(north,east,south,west))
        vert = get_time_interpolated_wind(windnc4,'v',time,(north,east,south,west))
    else:
        if tcLon <= -160: # Over extending west
            west = np.searchsorted(lons,tcLon+340)
            east = np.searchsorted(lons,tcLon+20)
        else: # Over extending east
            west = np.searchsorted(lons,tcLon-20)
            east = np.searchsorted(lons,tcLon-340)
        leftHoz = get_time_interpolated_wind(windnc4,'u',time,(north,1440 ,south,west))
        rightHoz = get_time_interpolated_wind(windnc4,'u',time,(north,east,south,0))
        horiz = np.hstack((leftHoz,rightHoz))
        leftVer = get_time_interpolated_wind(windnc4,'v',time,(north,1440,south,west))
        rightVer = get_time_interpolated_wind(windnc4,'v',time,(north,east,south,0))
        vert = np.hstack((leftVer,rightVer))
    return horiz, vert

def hours_since_1900(time):
    """
    time = string of the format 'YYYY-MM-DD HH:MM:SS'
    
    returns how many hours have passed since 1 Jan 1900 00:00:00 UTC with half
    hourly resolution
    """
    [dateStr,timeStr] = time.split()
    hours = int(timeStr[0:2])+int(timeStr[3:5])/60
    days_since_1990 = datetime.date.fromisoformat(dateStr)-datetime.date(1900,1,1)
    return days_since_1990.days*24+hours

def wind_time_indexes(tc,dataset):
    """
    tc = tc data retrieved by getData()
    dataset = wind data retrieved by Dataset(winddata.nc)
    
    returns what the first and last timestep of the dataset taken up by the tc
    """
    startDatasetTime = dataset['time'][0]
    startTcTime = hours_since_1900(tc['times'][0])
    endTcTime = hours_since_1900(tc['times'][-1])
    endDatasetTime = dataset['time'][-1]
    if startTcTime < startDatasetTime:
        start = False
    else:
        start = int(2*(startTcTime-startDatasetTime))
    if endTcTime > endDatasetTime:
        end = False
    else:
        end = int(2*(endTcTime-startDatasetTime))
    return start,end

def wind_files(tc,wind_files_dest):
    """
    tc = tc data retrieved by getData()
    
    Returns the files relevant to the wind of the TC, in the format:
    [filename1,filename2,filename3,...],[(start,end),(start,end),(start,end),...]
    start and end indexes can be false if the tc goes outside of what that file
    covers.
    """
    wind_filenames = list(sorted(glob.glob(f"{wind_files_dest}/*.nc")))
    first_time = tc['times'][0]
    first_time_str = f"{first_time[0:4]}{first_time[5:7]}"
    first_time_ind = [idx for idx, s in enumerate(wind_filenames) if first_time_str in s][0]
    filenames = [wind_filenames[first_time_ind]]
    indicies = wind_time_indexes(tc, Dataset(wind_filenames[first_time_ind]))
    indicies_relevant = [indicies]
    offset = 1
    while not indicies_relevant[-1][1]:
        filenames.append(wind_filenames[first_time_ind+offset])
        indicies = wind_time_indexes(tc, Dataset(wind_filenames[first_time_ind+offset]))
        indicies_relevant.append(indicies)
        offset += 1
    return filenames, indicies_relevant

def process_wind(tcName,tcYear,windPath):
    tc = retrieveTCData(tcName, tcYear)
    tcTimeIndex = 0
    east = []
    north = []
    for filename,indicies in zip(*wind_files(tc,windPath)):
        start = 0 if indicies[0] == False else indicies[0]
        end = 2000 if indicies[1] == False else indicies[1]
        for t in range(start,end):
            tcLat = tc['lats'][tcTimeIndex]
            tcLon = tc['lons'][tcTimeIndex]
            try:
                easties,northies = getWindChunk(tcLat,tcLon,filename,t)
                east.append(easties)
                north.append(northies)
            except IndexError:
                break
            tcTimeIndex += 1
    return np.array(east),np.array(north)