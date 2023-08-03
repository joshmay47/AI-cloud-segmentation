# -*- coding: utf-8 -*-
"""
Created on Mon Jul  3 10:27:28 2023

@author: Josh
"""
import glob
from os.path import exists

import h5py
import matplotlib.pyplot as plt
import numpy as np
from netCDF4 import Dataset
from numba import njit,prange
from matplotlib import animation


def remove_invalid_times(original,time_strings):
    """
    original = original array that may have invalid times
    time_strings = list of times that correspond to the original array's values

    Removes times that are not associated with a timestep of 3 hours. The rest
    of this program requires timesteps in 3 hour increments until they are
    interpolated.
    """

    return np.array([originali for originali,timeStr in zip(original,time_strings)\
                     if not int(timeStr[11:13])%3 != 0])


def get_data(index,dataset):
    """
    ---------------------------------------------------------------------------
    index   = index of the TC in the IBTrACS dataset
    dataset = IBTrACS file

    Returns a dictionary containing the name, category, latitude and longitude
    through time, and the time stamps that belong to those location recordings.
    Retrieves information from IBTrACS only.
    ---------------------------------------------------------------------------
    """
    ibtracs = Dataset(dataset,"r")
    # find the index corresponding to the sid

    # Get name, complicated because of the way it is stored
    raw_name = ibtracs['name'][index].data
    name = ''
    for char_raw in raw_name:
        name += char_raw.decode("utf-8")

    #Get duration of TC (until there is no data left)
    lats = ibtracs['lat'][index].data
    max_index = 0
    while lats[max_index] != -9999:
        max_index += 1

    # Get time stamps, has the structure of YYYY-MM-DD HH:MM:SS,
    # access HH:MM with <dict>['times'][<index>][11:16]
    time_raws = ibtracs['iso_time'][index].data[0:max_index]
    times = []
    for time_raw in time_raws:
        time = ''
        for char_raw in time_raw:
            time += char_raw.decode("utf-8")
        times.append(time)

    #Get other values
    lats = lats[0:max_index]
    lons = ibtracs['lon'][index].data[0:max_index]
    sshs = ibtracs['usa_sshs'][index].data[0:max_index]
    usa_wind = ibtracs['usa_wind'][index].data[0:max_index]
    basin_raw = ibtracs['basin'][index].data[0]
    basin = basin_raw[0].decode("utf-8")+basin_raw[1].decode("utf-8")

    return {
        "name": name,
        "sshs": remove_invalid_times(sshs,times),
        "usa_wind": remove_invalid_times(usa_wind,times),
        "times": remove_invalid_times(times,times),
        "lats": remove_invalid_times(lats,times),
        "lons": remove_invalid_times(lons,times),
        "basin": basin
        }


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
    inter_vals = np.zeros((stamps[-1]+1))
    for i in range(len(stamps)-1):
        run = stamps[i+1]-stamps[i]
        rise = vals[i+1]-vals[i]
        gradient = rise/run
        for j in range(run+1):
            inter_vals[stamps[i]+j]=vals[i]+j*gradient
    return inter_vals


def get_index(name,year,ibtracs_dest,multiple=False):
    """
    ---------------------------------------------------------------------------
    name        = Name of the TC (str)
    year        = Year of the TC (str)
    ibtracs_dest = filename of the IBTrACS dataset
    multiple    = boolean whether the we check if the dataset contains more 
                    than one entry of that name and year.

    Returns sid that matches the name and year of the TC
    ---------------------------------------------------------------------------
    """
    ibtracs = Dataset(ibtracs_dest,"r")
    names = ibtracs['name'][:].data
    index = 0
    max_storm = len(names)
    found = False
    found_indicies = []
    while (index < max_storm) and (found is False):
        raw_name = names[index]
        stored_name = "".join(map(lambda name: name.decode("utf-8"),raw_name))
        if len(name) != len(stored_name):
            found = False
            index += 1
            continue
        found = True
        for i, chara in enumerate(name):
            if chara != stored_name[i]:
                found = False
                break
        if found:
            raw_sid = ibtracs['sid'][index].data
            sid = "".join(map(lambda sid: sid.decode("utf-8"),raw_sid))
            if sid[0:4] == str(year):
                if not multiple:
                    return index
                found_indicies.append(index)
            else:
                found = False
        index += 1
    if not multiple:
        raise Exception("There isn't a TC with that name and year in the IBTrACS dataset")
    return found_indicies


def get_time_indicies(tc):
    """
    ---------------------------------------------------------------------------
    tc     = tropical cyclone, in the format from get_data

    Returns a list, first element is the starting time (0), each successive
    element is how many 30 minute inter_vals have been between that timestamp
    and the start. Is used for interpolating between lat and long locations
    ---------------------------------------------------------------------------
    """
    # Converts hours and minutes in the times entry in tc to values
    raw_times = []
    for time in tc['times']:
        raw_times.append(2*int(time[11:13])+int(time[14:16])//30)
    # The values go down when it enters a new day, a new day is detected when
    # the next value is less than the previous value. A day has 48 30min slots
    # this then gets added to the raw_times array to keep the days considered
    # as 24 hour gaps
    offsets = [None]*len(raw_times)
    offsets[0] = 0
    offset = 0
    for raw_time_index in range(1,len(raw_times)):
        if raw_times[raw_time_index] < raw_times[raw_time_index-1]:
            offset += 48
        offsets[raw_time_index] = offset
    return [sum(x)-raw_times[0] for x in zip(raw_times, offsets)]

def time_to_filenames(time):
    start = f'merg_{time[0:4]}{time[5:7]}{time[8:10]}'
    end = '_4km-pixel.nc4'
    return (f'{start}{time[11:13]}{end}',
        f'{start}{int(time[11:13])+1:02}{end}',
        f'{start}{int(time[11:13])+2:02}{end}')

def get_full_tc_info(name,year,ibtracs_dest,sat_path):
    """
    ---------------------------------------------------------------------------
    name         = The TC's name
    year         = The year in which the TC began
    ibtracs_dest = Filename of the IBTrACS dataset
    sat_path     = Where the satellite files are located

    Finds the TC and retrieves relevant statistical data about its evolution
    including latitudes, longitudes, times, tc_name, sid. This is used for
    analysis of the storm, not for processing.
    ---------------------------------------------------------------------------
    """
    index = get_index(name.upper(),year,ibtracs_dest,multiple=True)
    if len(index) > 1:
        raise Exception("There are more than one TC of that type")
    index = index[0]
    tc = get_data(index)
    start_frame = np.where(tc["usa_wind"] > 35)[0][0]*6
    end_frame = (np.where(tc["usa_wind"] > 35)[0][-1]-1)*6
    raw_times = get_time_indicies(tc)
    files = list(sorted(glob.glob(f"{sat_path}*")))
    # Find the first file
    for three_hourly in range(len(raw_times)):
        time = tc['times'][three_hourly]
        h1,h2,h3 = time_to_filenames(time)
        if not exists(sat_path+h1):
            raise FileNotFoundError(f"{sat_path}{h1} was requested but not found.")
        if not exists(sat_path+h2):
            raise FileNotFoundError(f"{sat_path}{h2} was requested but not found.")
        if not exists(sat_path+h3):
            raise FileNotFoundError(f"{sat_path}{h3} was requested but not found.")
    first_file = time_to_filenames(tc['times'][0])[0]
    last_file = time_to_filenames(tc['times'][-1])[0]
    start_file_ind = files.index(sat_path+first_file)
    end_file_ind = files.index(sat_path+last_file)

    # Find file names, and interpolate the latitude and longitude values for the tc
    interp_lats = interpolator(raw_times,tc['lats'])
    interp_lons = interpolator(raw_times,tc['lons'])
    full_times = []
    for time in tc['times']:
        day = time[:10]
        hour = int(time[11:13])
        for interp_hour in range(hour,hour+3):
            full_times.append(f"{day} {interp_hour:02}:00:00")
            full_times.append(f"{day} {interp_hour:02}:30:00")
    full_times = np.array(full_times)
    return {'lats':interp_lats[start_frame:end_frame],
            'lons':interp_lons[start_frame:end_frame],
            'times':full_times[start_frame:end_frame],
            'files':files[start_file_ind+start_frame//2:min(end_file_ind+1,start_file_ind+end_frame//2)],
            'name':name,
            'index':index,
            'sshs':tc['sshs'],
            'basin':tc['basin']}


def get_chunk(tc_lat,tc_lon,file,min30=0):
    """
    ---------------------------------------------------------------------------
    tc_lat = latitude of the centre of the TC at a particular time
    tc_lon = longitude of the centre of the TC at a particular time
    file = .nc4 file corresponding to the particular time
    min30 = 0 or 1, if it is an even or half hour

    Returns a size x size chunk of the .nc4 satellite image centered at
    coordinates: tc_lat,tc_lon. Can be given for an hourly or half hourly
    snapshot.
    ---------------------------------------------------------------------------
    """
    data = file['Tb']
    lons = file['lon'][:]
    lats = file['lat'][:]
    tc_lati =  np.searchsorted(lats,tc_lat)
    tc_loni =  np.searchsorted(lons,tc_lon)
    tc_lon_max = len(lons)
    tc_lat_max = len(lats)
    overreach_down = False
    overreach_up = False
    half = 512

    # Too far up
    if tc_lati < half:
        overreach_up = True
        fill = np.zeros((half-tc_lati,1024))-9999.0

    # Too far down
    elif tc_lati+half > tc_lat_max:
        overreach_down = True
        fill = np.zeros((tc_lati+half-tc_lat_max,1024))-9999.0

    # Too far left
    if tc_loni < half:
        if overreach_up: # ^\ :)
            left = data[min30,:tc_lati+half,tc_lon_max-half+tc_loni:]
            right = data[min30,:tc_lati+half,:tc_loni+half]
            return np.vstack((fill,np.hstack((left,right))))
        if overreach_down:# v/
            left = data[min30,tc_lati-half:,tc_lon_max-half+tc_loni:]
            right = data[min30,tc_lati-half:,:tc_loni+half]
            return np.vstack((np.hstack((left,right)),fill))
        # <-
        left = data[min30,tc_lati-half:tc_lati+half,tc_lon_max-half+tc_loni:]
        right = data[min30,tc_lati-half:tc_lati+half,:tc_loni+half]
        return np.hstack((left,right))

    # Too far right
    if tc_loni+half > tc_lon_max:
        if overreach_up: # /^ :)
            left = data[min30,:tc_lati+half,tc_loni-half:]
            right = data[min30,:tc_lati+half,:tc_loni+half-tc_lon_max]
            return np.vstack((fill,np.hstack((left,right))))
        if overreach_down: # \v
            left = data[min30,tc_lati-half:,tc_loni-half:]
            right = data[min30,tc_lati-half:,:tc_loni+half-tc_lon_max]
            return np.vstack((np.hstack((left,right)),fill))
        # ->
        left = data[min30,tc_lati-half:tc_lati+half,tc_loni-half:]
        right = data[min30,tc_lati-half:tc_lati+half,:tc_loni+half-tc_lon_max]
        return np.hstack((left,right))

    if overreach_up: # ^ :)
        return np.vstack((fill,data[min30,:tc_lati+half,tc_loni-half:tc_loni+half]))
    if overreach_down: # v
        return np.vstack((data[min30,tc_lati-half:,tc_loni-half:tc_loni+half],fill))
    #
    return data[min30,tc_lati-half:tc_lati+half,tc_loni-half:tc_loni+half]


@njit(parallel=True)
def average_pool(image):
    """
    image = image to be downscaled, must be of dimensions 1024x1024.
    
    Returns a 512x512 downscaled image with each pixel based on the mean 
    (ignoring missing values) of 2x2 pixels of the original.

    """
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


def retain_data_for_missing_data(dataset):
    """
    ---------------------------------------------------------------------------
    dataset = set of data, could be images or labels

    Returns the set of data, where the parts that have missed data are retained
    from the previous frames data
    ---------------------------------------------------------------------------
    """

    padding = np.zeros((1,dataset.shape[2],dataset.shape[2]),dtype=np.uint8)
    set_shifted = np.concatenate((padding,dataset), axis=0)
    set_padded = np.concatenate((dataset,padding), axis=0)
    set_padded[set_padded == 0] = set_shifted[set_padded == 0]
    return set_padded[:-1]


def get_full_name(tc_data):
    """
    ---------------------------------------------------------------------------
    tc_data = TC dict created by get_data()

    Returns a string that contains the full name of the TC, dependant on
        intensity
    ---------------------------------------------------------------------------
    """
    maximum = max(tc_data["sshs"])
    types = ["Tropical Depression ","Tropical Storm ","Category 1 ",
             "Category 2 ","Category 3 ","Category 4 ", "Category 5 "]
    addon = ""
    if maximum > 0:
        basin = tc_data["basin"]
        if basin == "NA":
            addon = "Hurricane "
        elif basin == "EP":
            addon = "Hurricane "
        elif basin == "WP":
            addon = "Typhoon "
        elif basin == "NI":
            addon = "Cyclone "
        elif basin == "SI":
            addon = "Cyclone "
        elif basin == "SP":
            addon = "Cyclone "
        elif basin == "SA":
            addon = "Hurricane "
    if maximum < -1:
        return "Storm "+tc_data["name"].title()
    return types[maximum+1]+addon+tc_data["name"].title()


def create_gif(images,save_dest,tc_name):
    """
    images = images that are made into a gif (ideally 512x512)
    save_dest = the destination and filename that the gif will be saved as
    tc_name = the label that is attached to the gif at the top of the gif.
    
    Creates and saves a gif that contains an image that is 512x512 with a
    banner on the top that says the TCs name (according to tc_name).
    """
    # Setup frames
    fig = plt.figure(figsize=(7.12,7.49))
    ax = fig.add_axes([0,0,1,0.95])
    ax.set_title(tc_name, fontsize=20.0)
    ax.axis("off")
    frames = []

    # Add data into frames
    for image in images:
        im1 = ax.imshow(image,cmap='gray_r',interpolation='none')
        frames.append([im1])

    ani = animation.ArtistAnimation(fig, frames, interval=50)
    ani.save(save_dest,writer='pillow')


def process(name,year, sat_path, ibtracs_dest,
            gif = False, save_name = "processed", verbose = False):
    """
    ---------------------------------------------------------------------------
    name        = name of the TC
    year        = year that the TC occurred
    sat_path     = Where the satellite files are located
    ibtracs_dest = filename of the ibtracs dataset
    gif         = when True, creates a gif showing the images. False by default
    save_name    = the gif is saved as <save_name>.gif

    Creates an array of satellite images where missing pixels are set to 0, and
    every other pixel is set to the infrared temperature in kelvin-100. Times
    where the 1-minute sustained winds are above 35 knots are included only.

    If input gif = True, also produces a gif showing satellite images of the
    storm.
    ---------------------------------------------------------------------------
    """
    if verbose: print("Retrieving TC information...")
    tc = get_full_tc_info(name, year, ibtracs_dest, sat_path=sat_path)
    tc_filenames = tc['files']
    interp_lats = tc['lats']
    interp_lons = tc['lons']
    
    if verbose: print("Creating satellite Images...")
    satellite = np.zeros((2*len(tc_filenames)-1,512,512), dtype = np.uint8)

    last_update = 0
    end_frame = 2*len(tc_filenames)-1
    for i in range(end_frame):
        if verbose:
            if 50*(i/len(tc_filenames))-last_update > 10:
                print(int(50*(i/len(tc_filenames))),"% complete")
                last_update += 10

        # Retrieve file
        if i%2 == 0:
            try:
                satellitenc4 = h5py.File(tc_filenames[i//2],"r")
            except OSError as e:
                raise RuntimeError(str(tc_filenames[i//2][-29:]+" has invalid data (OSError).")) from e

        # Set no data pixels to 0
        try:
            chunk = get_chunk(interp_lats[i], interp_lons[i], satellitenc4, i%2)
            mask = chunk==-9999
            chunk[mask]=100
            chunk = np.array(chunk-100)
            chunk = average_pool(chunk)
            satellite[i] = chunk
        except RuntimeError as e:
            raise RuntimeError(str(tc_filenames[i//2][-29:]+" has invalid data (runtime).")) from e
        except OSError as e:
            raise RuntimeError(str(tc_filenames[i//2][-29:]+" has invalid data (OSError).")) from e
        del chunk
    satellite = retain_data_for_missing_data(satellite)
    if gif:
        if verbose: print("Creating Gif...")
        create_gif(satellite,f"processedAnalysis\\{year}\\{save_name}.gif",f"{get_full_name(tc)} ({year})")
    if verbose: print("Complete.")
    return satellite
