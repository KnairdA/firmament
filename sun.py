import numpy as np
from datetime import datetime

## Sun direction depending on time and place
# As described in Appendix D of "ME 4131 Thermal Environmental Engineering Laboratory Manual"

def sun_declination(time):
    day_of_year = time.timetuple().tm_yday
    return 23.45 * np.sin(np.radians((360/365)*(284+day_of_year)))

def equation_of_time(time):
    day_of_year = time.timetuple().tm_yday
    b = np.radians(360*(day_of_year-81)/364)
    return 0.165*np.sin(2*b) - 0.126*np.cos(b) - 0.025*np.sin(b)

def sun_direction(lat, lon, time, time_diff, summertime_shift = 0):
    lon_std = time_diff * 15
    clock_time       = time.hour + time.minute/60
    local_solar_time = clock_time + (1/15)*(lon - lon_std) + equation_of_time(time) - summertime_shift
    hour_angle = 15*(local_solar_time - 12)

    l = np.radians(lat)
    h = np.radians(hour_angle)
    d = np.radians(sun_declination(time))

    altitude = np.arcsin(np.cos(l) * np.cos(h) * np.cos(d) + np.sin(l) * np.sin(d))
    azimuth  = np.arccos((np.cos(d) * np.sin(l) * np.cos(h) - np.sin(d) * np.cos(l)) / np.cos(altitude))

    if h < 0:
        return (altitude,  azimuth)
    else:
        return (altitude, -azimuth)
