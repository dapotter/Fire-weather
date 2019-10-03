#!/usr/bin/env python
#################################################################
# Python Script to retrieve 240 online Data files of 'ds608.0',
# total 163.38G. This script uses 'requests' to download data.
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
import time
from urllib3.exceptions import NewConnectionError

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
values = {'email' : 'daniel.b.potter@gmail.com', 'passwd' : '8*forereffect', 'action' : 'login'}
# Authenticate
ret = requests.post(url,data=values)
if ret.status_code != 200:
    print('Bad Authentication')
    print(ret.text)
    exit(1)
dspath = 'http://rda.ucar.edu/data/ds608.0/'
filelist = [
'3HRLY/1983/NARR3D_198301_0103.tar']

connection_timeout = 1200 # seconds

for file in filelist:
    filename=dspath+file
    file_base = os.path.basename(file)
    print('Downloading',file_base)

    start_time = time.time()
    while True:
        try:
            print('Waiting 10 seconds')
            time.sleep(10)
            print('Trying connection...')
            req = requests.get(filename, cookies = ret.cookies, allow_redirects=True, stream=True, timeout=5)
            break
        except requests.exceptions.RequestException as e:
            print('************* TimeoutError detected *************')
            print('Error:\n', e)

            print('Sleeping 80 seconds...')
            time.sleep(80) # attempting once every 20 seconds
            
            try:
                print('Posting...')
                ret = requests.post(url,data=values, timeout=30)
                if ret.status_code != 200:
                    print('Bad Authentication')
                    print(ret.text)
                    exit(1)
                elif ret.status_code == 200:
                    print('Reconnection status code is good')
            except:
                print('Posting...')
                ret = requests.post(url,data=values, timeout=30)

            if time.time() > start_time + connection_timeout:
                raise Exception('Unable to get updates after {} seconds of ConnectionErrors'.format(connection_timeout))
            else:
                print('Sleeping 20 seconds...')
                time.sleep(20) # attempting once every 20 seconds
        # except NewConnectionError as nce:
        #   print('************* NewConnectionError detected *************')
        #   print('Error:\n', nce)
        #   if time.time() > start_time + connection_timeout:
        #       raise Exception('Unable to get updates after {} seconds of NewConnectionError'.format(connection_timeout))
        #   else:
        #       print('Sleeping 10 seconds...')
        #       time.sleep(10) # attempting once every 20 seconds


    filesize = int(req.headers['Content-length'])
    with open(file_base, 'wb') as outfile:
        chunk_size=1048576
        for chunk in req.iter_content(chunk_size=chunk_size):
            outfile.write(chunk)
            if chunk_size < filesize:
                check_file_status(file_base, filesize)
    check_file_status(file_base, filesize)
    print()