#################################################
# Download GOES-16/17 data, you will need to run 
# this for each date and time range you need  
# Default channels: 01, 02, 03, 05, 13 are needed
# to make: TrueColor and Day Cloud Phase RGB
#################################################

################################################
# Define libraries

import GOES as GOES
import os

################################################
# User edit section
# Note: it will attach the date to the end of the directory chain (e.g., \\XX\\YY\\20150232)

date = '20210518' #YYYYMMDD
init_time ='160000' #HHMMSS
end_time = '190000' #HHMMSS
ch=['01', '02', '03', '05', '13'] #Channel numbers

####################################################################
# Edit only if you know what you are doing!!!! 

sat='goes16' # select satellite type 
type='ABI-L2-CMIPC' # file type

cwd = os.getcwd()

root = cwd+'\\REPO\\' #root directory where you want the data to be saved. 

################################################
# Crunch-a-tize me Cap'n

output_dir=root+date+'\\'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

init = date + '-' + init_time
end = date + '-' + end_time

for i in range(0, len(ch)):
    flist = GOES.download(sat, type, DateTimeIni = init, DateTimeFin = end, channel = [ch[i]], path_out=output_dir)

