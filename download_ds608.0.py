#!/usr/bin/env python
#
#
#
#                 RUN THIS FROM TERMINAL ONLY
#
#
#
#
#################################################################
# Python Script to retrieve 71 online Data files of 'ds608.0',
# total 57.77G. This script uses 'requests' to download data.
#
# Highlight this script by Select All, Copy and Paste it into a file;
# make the file executable and run it on command line.
#
# You need pass in your password as a parameter to execute
# this script; or you can set an environment variable RDAPSWD
# if your Operating System supports it.
#
# Contact chifan@ucar.edu (Chi-Fan Shih) for further assistance.
#################################################################


import sys, os
import requests

def check_file_status(filepath, filesize):
    sys.stdout.write('\r')
    sys.stdout.flush()
    size = int(os.stat(filepath).st_size)
    percent_complete = (size/filesize)*100
    sys.stdout.write('%.3f %s' % (percent_complete, '% Completed'))
    sys.stdout.flush()

# Try to get password
if len(sys.argv) < 2 and not 'RDAPSWD' in os.environ:
    try:
        import getpass
        input = getpass.getpass
    except:
        try:
            input = raw_input
        except:
            pass
    pswd = input('Password: ')
else:
    try:
        pswd = sys.argv[1]
    except:
        pswd = os.environ['RDAPSWD']

url = 'https://rda.ucar.edu/cgi-bin/login'
values = {'email' : 'daniel.b.potter@gmail.com', 'passwd' : pswd, 'action' : 'login'}
# Authenticate
ret = requests.post(url,data=values)
if ret.status_code != 200:
    print('Bad Authentication')
    print(ret.text)
    exit(1)
dspath = 'http://rda.ucar.edu/data/ds608.0/'
# filelist = [
# '3HRLY/1979/NARR3D_197904_0103.tar',
# '3HRLY/1979/NARR3D_197904_0406.tar',
# '3HRLY/1979/NARR3D_197904_0709.tar',
# '3HRLY/1979/NARR3D_197904_1012.tar',
# '3HRLY/1979/NARR3D_197904_1315.tar',
# '3HRLY/1979/NARR3D_197904_1618.tar',
# '3HRLY/1979/NARR3D_197904_1921.tar',
# '3HRLY/1979/NARR3D_197904_2224.tar',
# '3HRLY/1979/NARR3D_197904_2527.tar',
# '3HRLY/1979/NARR3D_197904_2830.tar',
# '3HRLY/1979/NARR3D_197905_0103.tar',
# '3HRLY/1979/NARR3D_197905_0406.tar',
# '3HRLY/1979/NARR3D_197905_0709.tar',
# '3HRLY/1979/NARR3D_197905_1012.tar',
# '3HRLY/1979/NARR3D_197905_1315.tar',
# '3HRLY/1979/NARR3D_197905_1618.tar',
# '3HRLY/1979/NARR3D_197905_1921.tar',
# '3HRLY/1979/NARR3D_197905_2224.tar',
# '3HRLY/1979/NARR3D_197905_2527.tar',
# '3HRLY/1979/NARR3D_197905_2831.tar',
# '3HRLY/1979/NARR3D_197906_0103.tar',
# '3HRLY/1979/NARR3D_197906_0406.tar',
# '3HRLY/1979/NARR3D_197906_0709.tar',
# '3HRLY/1979/NARR3D_197906_1012.tar',
# '3HRLY/1979/NARR3D_197906_1315.tar',
# '3HRLY/1979/NARR3D_197906_1618.tar',
# '3HRLY/1979/NARR3D_197906_1921.tar',
# '3HRLY/1979/NARR3D_197906_2224.tar',
# '3HRLY/1979/NARR3D_197906_2527.tar',
# '3HRLY/1979/NARR3D_197906_2830.tar',
# '3HRLY/1979/NARR3D_197907_0103.tar',
# '3HRLY/1979/NARR3D_197907_0406.tar',
# '3HRLY/1979/NARR3D_197907_0709.tar',
# '3HRLY/1979/NARR3D_197907_1012.tar',
# '3HRLY/1979/NARR3D_197907_1315.tar',
# '3HRLY/1979/NARR3D_197907_1618.tar',
# '3HRLY/1979/NARR3D_197907_1921.tar',
# '3HRLY/1979/NARR3D_197907_2224.tar',
# '3HRLY/1979/NARR3D_197907_2527.tar',
# '3HRLY/1979/NARR3D_197907_2831.tar',
# '3HRLY/1979/NARR3D_197908_0103.tar',
# '3HRLY/1979/NARR3D_197908_0406.tar',
# '3HRLY/1979/NARR3D_197908_0709.tar',
# '3HRLY/1979/NARR3D_197908_1012.tar',
# '3HRLY/1979/NARR3D_197908_1315.tar',
# '3HRLY/1979/NARR3D_197908_1618.tar',
# '3HRLY/1979/NARR3D_197908_1921.tar',
# '3HRLY/1979/NARR3D_197908_2224.tar',
# '3HRLY/1979/NARR3D_197908_2527.tar',
# '3HRLY/1979/NARR3D_197908_2831.tar',
# '3HRLY/1979/NARR3D_197909_0103.tar',
# '3HRLY/1979/NARR3D_197909_0406.tar',
# '3HRLY/1979/NARR3D_197909_0709.tar',
# '3HRLY/1979/NARR3D_197909_1012.tar',
# '3HRLY/1979/NARR3D_197909_1315.tar',
# '3HRLY/1979/NARR3D_197909_1618.tar',
# '3HRLY/1979/NARR3D_197909_1921.tar',
# '3HRLY/1979/NARR3D_197909_2224.tar',
# '3HRLY/1979/NARR3D_197909_2527.tar',
# '3HRLY/1979/NARR3D_197909_2830.tar',
# '3HRLY/1979/NARR3D_197910_0103.tar',
# '3HRLY/1979/NARR3D_197910_0406.tar',
# '3HRLY/1979/NARR3D_197910_0709.tar',
# '3HRLY/1979/NARR3D_197910_1012.tar',
# '3HRLY/1979/NARR3D_197910_1315.tar',
# '3HRLY/1979/NARR3D_197910_1618.tar',
# '3HRLY/1979/NARR3D_197910_1921.tar',
# '3HRLY/1979/NARR3D_197910_2224.tar',
# '3HRLY/1979/NARR3D_197910_2527.tar',
# '3HRLY/1979/NARR3D_197910_2831.tar',
# '3HRLY/1979/NARR3D_197911_0103.tar']
for file in filelist:
    filename=dspath+file
    file_base = os.path.basename(file)
    print('Downloading',file_base)
    req = requests.get(filename, cookies = ret.cookies, allow_redirects=True, stream=True)
    filesize = int(req.headers['Content-length'])
    with open(file_base, 'wb') as outfile:
        chunk_size=1048576
        for chunk in req.iter_content(chunk_size=chunk_size):
            outfile.write(chunk)
            if chunk_size < filesize:
                check_file_status(file_base, filesize)
    check_file_status(file_base, filesize)
    print()
