#! /usr/bin/env python
#
# python script to download selected files from rda.ucar.edu
# after you save the file, don't forget to make it executable
#   i.e. - "chmod 755 <name_of_script>"
#
import sys
import os
#import urllib2
#from urllib.request import urlopen
import urllib
#import cookielib
from cookiejar import CookieJar
#
if (len(sys.argv) != 2):
  print("usage: "+sys.argv[0]+" [-q] password_on_RDA_webserver")
  print("-q suppresses the progress message for each file that is downloaded")
  sys.exit(1)
#
passwd_idx=1
verbose=True
if (len(sys.argv) == 3 and sys.argv[1] == "-q"):
  passwd_idx=2
  verbose=False
#
#cj=cookielib.MozillaCookieJar()
cj=CookieJar.MozillaCookieJar()
#opener=urllib2.build_opener(urllib2.HTTPCookieProcessor(cj))
opener=urllib.request.build_opener(urllib.request.HTTPCookieProcessor(cj))

#
# check for existing cookies file and authenticate if necessary
do_authentication=False
if (os.path.isfile("auth.rda.ucar.edu")):
  cj.load("auth.rda.ucar.edu",False,True)
  for cookie in cj:
    if (cookie.name == "sess" and cookie.is_expired()):
      do_authentication=True
else:
  do_authentication=True
if (do_authentication):
  #login=opener.open("https://rda.ucar.edu/cgi-bin/login","email=daniel.b.potter@gmail.com&password="+sys.argv[1]+"&action=login")
  login=urllib.request.urlopen("https://rda.ucar.edu/cgi-bin/login","email=daniel.b.potter@gmail.com&password="+sys.argv[1]+"&action=login")

#
# save the authentication cookies for future downloads
# NOTE! - cookies are saved for future sessions because overly-frequent authentication to our server can cause your data access to be blocked
  cj.clear_session_cookies()
  cj.save("auth.rda.ucar.edu",True,True)
#
# download the data file(s)
listoffiles=[
'3HRLY/1980/NARRflx_198001_0108.tar',
'3HRLY/1980/NARRflx_198001_0916.tar',
'3HRLY/1980/NARRflx_198001_1724.tar',
'3HRLY/1980/NARRflx_198001_2531.tar',
'3HRLY/1980/NARRflx_198002_0108.tar',
'3HRLY/1980/NARRflx_198002_0916.tar',
'3HRLY/1980/NARRflx_198002_1724.tar',
'3HRLY/1980/NARRflx_198002_2529.tar',
'3HRLY/1980/NARRflx_198003_0108.tar',
'3HRLY/1980/NARRflx_198003_0916.tar',
'3HRLY/1980/NARRflx_198003_1724.tar',
'3HRLY/1980/NARRflx_198003_2531.tar',
'3HRLY/1980/NARRflx_198004_0108.tar',
'3HRLY/1980/NARRflx_198004_0916.tar',
'3HRLY/1980/NARRflx_198004_1724.tar',
'3HRLY/1980/NARRflx_198004_2530.tar',
'3HRLY/1980/NARRflx_198005_0108.tar',
'3HRLY/1980/NARRflx_198005_0916.tar',
'3HRLY/1980/NARRflx_198005_1724.tar',
'3HRLY/1980/NARRflx_198005_2531.tar',
'3HRLY/1980/NARRflx_198006_0108.tar',
'3HRLY/1980/NARRflx_198006_0916.tar',
'3HRLY/1980/NARRflx_198006_1724.tar',
'3HRLY/1980/NARRflx_198006_2530.tar',
'3HRLY/1980/NARRflx_198007_0108.tar',
'3HRLY/1980/NARRflx_198007_0916.tar',
'3HRLY/1980/NARRflx_198007_1724.tar',
'3HRLY/1980/NARRflx_198007_2531.tar',
'3HRLY/1980/NARRflx_198008_0108.tar',
'3HRLY/1980/NARRflx_198008_0916.tar',
'3HRLY/1980/NARRflx_198008_1724.tar',
'3HRLY/1980/NARRflx_198008_2531.tar',
'3HRLY/1980/NARRflx_198009_0108.tar',
'3HRLY/1980/NARRflx_198009_0916.tar',
'3HRLY/1980/NARRflx_198009_1724.tar',
'3HRLY/1980/NARRflx_198009_2530.tar',
'3HRLY/1980/NARRflx_198010_0108.tar',
'3HRLY/1980/NARRflx_198010_0916.tar',
'3HRLY/1980/NARRflx_198010_1724.tar',
'3HRLY/1980/NARRflx_198010_2531.tar',
'3HRLY/1980/NARRflx_198011_0108.tar',
'3HRLY/1980/NARRflx_198011_0916.tar',
'3HRLY/1980/NARRflx_198011_1724.tar',
'3HRLY/1980/NARRflx_198011_2530.tar',
'3HRLY/1980/NARRflx_198012_0108.tar',
'3HRLY/1980/NARRflx_198012_0916.tar',
'3HRLY/1980/NARRflx_198012_1724.tar',
'3HRLY/1980/NARRflx_198012_2531.tar',
'3HRLY/1980/NARRpbl_198001_0109.tar',
'3HRLY/1980/NARRpbl_198001_1019.tar',
'3HRLY/1980/NARRpbl_198001_2031.tar',
'3HRLY/1980/NARRpbl_198002_0109.tar',
'3HRLY/1980/NARRpbl_198002_1019.tar',
'3HRLY/1980/NARRpbl_198002_2029.tar',
'3HRLY/1980/NARRpbl_198003_0109.tar',
'3HRLY/1980/NARRpbl_198003_1019.tar',
'3HRLY/1980/NARRpbl_198003_2031.tar',
'3HRLY/1980/NARRpbl_198004_0109.tar',
'3HRLY/1980/NARRpbl_198004_1019.tar',
'3HRLY/1980/NARRpbl_198004_2030.tar',
'3HRLY/1980/NARRpbl_198005_0109.tar',
'3HRLY/1980/NARRpbl_198005_1019.tar',
'3HRLY/1980/NARRpbl_198005_2031.tar',
'3HRLY/1980/NARRpbl_198006_0109.tar',
'3HRLY/1980/NARRpbl_198006_1019.tar',
'3HRLY/1980/NARRpbl_198006_2030.tar',
'3HRLY/1980/NARRpbl_198007_0109.tar',
'3HRLY/1980/NARRpbl_198007_1019.tar',
'3HRLY/1980/NARRpbl_198007_2031.tar',
'3HRLY/1980/NARRpbl_198008_0109.tar',
'3HRLY/1980/NARRpbl_198008_1019.tar',
'3HRLY/1980/NARRpbl_198008_2031.tar',
'3HRLY/1980/NARRpbl_198009_0109.tar',
'3HRLY/1980/NARRpbl_198009_1019.tar',
'3HRLY/1980/NARRpbl_198009_2030.tar',
'3HRLY/1980/NARRpbl_198010_0109.tar',
'3HRLY/1980/NARRpbl_198010_1019.tar',
'3HRLY/1980/NARRpbl_198010_2031.tar',
'3HRLY/1980/NARRpbl_198011_0109.tar',
'3HRLY/1980/NARRpbl_198011_1019.tar',
'3HRLY/1980/NARRpbl_198011_2030.tar',
'3HRLY/1980/NARRpbl_198012_0109.tar',
'3HRLY/1980/NARRpbl_198012_1019.tar',
'3HRLY/1980/NARRpbl_198012_2031.tar',
'3HRLY/1980/NARRsfc_198001_0109.tar',
'3HRLY/1980/NARRsfc_198001_1019.tar',
'3HRLY/1980/NARRsfc_198001_2031.tar',
'3HRLY/1980/NARRsfc_198002_0109.tar',
'3HRLY/1980/NARRsfc_198002_1019.tar',
'3HRLY/1980/NARRsfc_198002_2029.tar',
'3HRLY/1980/NARRsfc_198003_0109.tar',
'3HRLY/1980/NARRsfc_198003_1019.tar',
'3HRLY/1980/NARRsfc_198003_2031.tar',
'3HRLY/1980/NARRsfc_198004_0109.tar',
'3HRLY/1980/NARRsfc_198004_1019.tar',
'3HRLY/1980/NARRsfc_198004_2030.tar',
'3HRLY/1980/NARRsfc_198005_0109.tar',
'3HRLY/1980/NARRsfc_198005_1019.tar',
'3HRLY/1980/NARRsfc_198005_2031.tar',
'3HRLY/1980/NARRsfc_198006_0109.tar',
'3HRLY/1980/NARRsfc_198006_1019.tar',
'3HRLY/1980/NARRsfc_198006_2030.tar',
'3HRLY/1980/NARRsfc_198007_0109.tar',
'3HRLY/1980/NARRsfc_198007_1019.tar',
'3HRLY/1980/NARRsfc_198007_2031.tar',
'3HRLY/1980/NARRsfc_198008_0109.tar',
'3HRLY/1980/NARRsfc_198008_1019.tar',
'3HRLY/1980/NARRsfc_198008_2031.tar',
'3HRLY/1980/NARRsfc_198009_0109.tar',
'3HRLY/1980/NARRsfc_198009_1019.tar',
'3HRLY/1980/NARRsfc_198009_2030.tar',
'3HRLY/1980/NARRsfc_198010_0109.tar',
'3HRLY/1980/NARRsfc_198010_1019.tar',
'3HRLY/1980/NARRsfc_198010_2031.tar',
'3HRLY/1980/NARRsfc_198011_0109.tar',
'3HRLY/1980/NARRsfc_198011_1019.tar',
'3HRLY/1980/NARRsfc_198011_2030.tar',
'3HRLY/1980/NARRsfc_198012_0109.tar',
'3HRLY/1980/NARRsfc_198012_1019.tar',
'3HRLY/1980/NARRsfc_198012_2031.tar',
'3HRLY/1980/NARR3D_198001_0103.tar',
'3HRLY/1980/NARR3D_198001_0406.tar',
'3HRLY/1980/NARR3D_198001_0709.tar',
'3HRLY/1980/NARR3D_198001_1012.tar',
'3HRLY/1980/NARR3D_198001_1315.tar',
'3HRLY/1980/NARR3D_198001_1618.tar',
'3HRLY/1980/NARR3D_198001_1921.tar',
'3HRLY/1980/NARR3D_198001_2224.tar',
'3HRLY/1980/NARR3D_198001_2527.tar',
'3HRLY/1980/NARR3D_198001_2831.tar',
'3HRLY/1980/NARR3D_198002_0103.tar',
'3HRLY/1980/NARR3D_198002_0406.tar',
'3HRLY/1980/NARR3D_198002_0709.tar',
'3HRLY/1980/NARR3D_198002_1012.tar',
'3HRLY/1980/NARR3D_198002_1315.tar',
'3HRLY/1980/NARR3D_198002_1618.tar',
'3HRLY/1980/NARR3D_198002_1921.tar',
'3HRLY/1980/NARR3D_198002_2224.tar',
'3HRLY/1980/NARR3D_198002_2527.tar',
'3HRLY/1980/NARR3D_198002_2829.tar',
'3HRLY/1980/NARR3D_198003_0103.tar',
'3HRLY/1980/NARR3D_198003_0406.tar',
'3HRLY/1980/NARR3D_198003_0709.tar',
'3HRLY/1980/NARR3D_198003_1012.tar',
'3HRLY/1980/NARR3D_198003_1315.tar',
'3HRLY/1980/NARR3D_198003_1618.tar',
'3HRLY/1980/NARR3D_198003_1921.tar',
'3HRLY/1980/NARR3D_198003_2224.tar',
'3HRLY/1980/NARR3D_198003_2527.tar',
'3HRLY/1980/NARR3D_198003_2831.tar',
'3HRLY/1980/NARR3D_198004_0103.tar',
'3HRLY/1980/NARR3D_198004_0406.tar',
'3HRLY/1980/NARR3D_198004_0709.tar',
'3HRLY/1980/NARR3D_198004_1012.tar',
'3HRLY/1980/NARR3D_198004_1315.tar',
'3HRLY/1980/NARR3D_198004_1618.tar',
'3HRLY/1980/NARR3D_198004_1921.tar',
'3HRLY/1980/NARR3D_198004_2224.tar',
'3HRLY/1980/NARR3D_198004_2527.tar',
'3HRLY/1980/NARR3D_198004_2830.tar',
'3HRLY/1980/NARR3D_198005_0103.tar',
'3HRLY/1980/NARR3D_198005_0406.tar',
'3HRLY/1980/NARR3D_198005_0709.tar',
'3HRLY/1980/NARR3D_198005_1012.tar',
'3HRLY/1980/NARR3D_198005_1315.tar',
'3HRLY/1980/NARR3D_198005_1618.tar',
'3HRLY/1980/NARR3D_198005_1921.tar',
'3HRLY/1980/NARR3D_198005_2224.tar',
'3HRLY/1980/NARR3D_198005_2527.tar',
'3HRLY/1980/NARR3D_198005_2831.tar',
'3HRLY/1980/NARR3D_198006_0103.tar',
'3HRLY/1980/NARR3D_198006_0406.tar',
'3HRLY/1980/NARR3D_198006_0709.tar',
'3HRLY/1980/NARR3D_198006_1012.tar',
'3HRLY/1980/NARR3D_198006_1315.tar',
'3HRLY/1980/NARR3D_198006_1618.tar',
'3HRLY/1980/NARR3D_198006_1921.tar',
'3HRLY/1980/NARR3D_198006_2224.tar',
'3HRLY/1980/NARR3D_198006_2527.tar',
'3HRLY/1980/NARR3D_198006_2830.tar',
'3HRLY/1980/NARR3D_198007_0103.tar',
'3HRLY/1980/NARR3D_198007_0406.tar',
'3HRLY/1980/NARR3D_198007_0709.tar',
'3HRLY/1980/NARR3D_198007_1012.tar',
'3HRLY/1980/NARR3D_198007_1315.tar',
'3HRLY/1980/NARR3D_198007_1618.tar',
'3HRLY/1980/NARR3D_198007_1921.tar',
'3HRLY/1980/NARR3D_198007_2224.tar',
'3HRLY/1980/NARR3D_198007_2527.tar']

for file in listoffiles:
  idx=file.rfind("/")
  if (idx > 0):
    ofile=file[idx+1:]
  else:
    ofile=file
  if (verbose):
    sys.stdout.write("downloading "+ofile+"...")
    sys.stdout.flush()
  infile=opener.open("http://rda.ucar.edu/data/ds608.0/"+file)
  outfile=open(ofile,"wb")
  outfile.write(infile.read())
  outfile.close()
  if (verbose):
    sys.stdout.write("done.\n")