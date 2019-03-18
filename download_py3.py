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
listoffiles=["3HRLY/1979/NARR3D_197901_0103.tar","3HRLY/1979/NARR3D_197901_0406.tar","3HRLY/1979/NARR3D_197901_0709.tar","3HRLY/1979/NARR3D_197901_1012.tar","3HRLY/1979/NARR3D_197901_1315.tar","3HRLY/1979/NARR3D_197901_1618.tar","3HRLY/1979/NARR3D_197901_1921.tar","3HRLY/1979/NARR3D_197901_2224.tar","3HRLY/1979/NARR3D_197901_2527.tar","3HRLY/1979/NARR3D_197901_2831.tar","3HRLY/1979/NARR3D_197902_0103.tar","3HRLY/1979/NARR3D_197902_0406.tar","3HRLY/1979/NARR3D_197902_0709.tar","3HRLY/1979/NARR3D_197902_1012.tar","3HRLY/1979/NARR3D_197902_1315.tar","3HRLY/1979/NARR3D_197902_1618.tar","3HRLY/1979/NARR3D_197902_1921.tar","3HRLY/1979/NARR3D_197902_2224.tar","3HRLY/1979/NARR3D_197902_2527.tar","3HRLY/1979/NARR3D_197902_2828.tar","3HRLY/1979/NARR3D_197903_0103.tar","3HRLY/1979/NARR3D_197903_0406.tar","3HRLY/1979/NARR3D_197903_0709.tar","3HRLY/1979/NARR3D_197903_1012.tar","3HRLY/1979/NARR3D_197903_1315.tar","3HRLY/1979/NARR3D_197903_1618.tar","3HRLY/1979/NARR3D_197903_1921.tar","3HRLY/1979/NARR3D_197903_2224.tar","3HRLY/1979/NARR3D_197903_2527.tar","3HRLY/1979/NARR3D_197903_2831.tar","3HRLY/1979/NARR3D_197911_0406.tar","3HRLY/1979/NARR3D_197911_0709.tar","3HRLY/1979/NARR3D_197911_1012.tar","3HRLY/1979/NARR3D_197911_1315.tar","3HRLY/1979/NARR3D_197911_1618.tar","3HRLY/1979/NARR3D_197911_1921.tar","3HRLY/1979/NARR3D_197911_2224.tar","3HRLY/1979/NARR3D_197911_2527.tar","3HRLY/1979/NARR3D_197911_2830.tar","3HRLY/1979/NARR3D_197912_0103.tar","3HRLY/1979/NARR3D_197912_0406.tar","3HRLY/1979/NARR3D_197912_0709.tar","3HRLY/1979/NARR3D_197912_1012.tar","3HRLY/1979/NARR3D_197912_1315.tar","3HRLY/1979/NARR3D_197912_1618.tar","3HRLY/1979/NARR3D_197912_1921.tar","3HRLY/1979/NARR3D_197912_2224.tar","3HRLY/1979/NARR3D_197912_2527.tar","3HRLY/1979/NARR3D_197912_2831.tar","3HRLY/1980/NARR3D_198001_0103.tar","3HRLY/1979/NARRflx_197901_0108.tar","3HRLY/1979/NARRflx_197901_0916.tar","3HRLY/1979/NARRflx_197901_1724.tar","3HRLY/1979/NARRflx_197901_2531.tar","3HRLY/1979/NARRflx_197902_0108.tar","3HRLY/1979/NARRflx_197902_0916.tar","3HRLY/1979/NARRflx_197902_1724.tar","3HRLY/1979/NARRflx_197902_2528.tar","3HRLY/1979/NARRflx_197903_0108.tar","3HRLY/1979/NARRflx_197903_0916.tar","3HRLY/1979/NARRflx_197903_1724.tar","3HRLY/1979/NARRflx_197903_2531.tar","3HRLY/1979/NARRflx_197904_0108.tar","3HRLY/1979/NARRflx_197904_0916.tar","3HRLY/1979/NARRflx_197904_1724.tar","3HRLY/1979/NARRflx_197904_2530.tar","3HRLY/1979/NARRflx_197905_0108.tar","3HRLY/1979/NARRflx_197905_0916.tar","3HRLY/1979/NARRflx_197905_1724.tar","3HRLY/1979/NARRflx_197905_2531.tar","3HRLY/1979/NARRflx_197906_0108.tar","3HRLY/1979/NARRflx_197906_0916.tar","3HRLY/1979/NARRflx_197906_1724.tar","3HRLY/1979/NARRflx_197906_2530.tar","3HRLY/1979/NARRflx_197907_0108.tar","3HRLY/1979/NARRflx_197907_0916.tar","3HRLY/1979/NARRflx_197907_1724.tar","3HRLY/1979/NARRflx_197907_2531.tar","3HRLY/1979/NARRflx_197908_0108.tar","3HRLY/1979/NARRflx_197908_0916.tar","3HRLY/1979/NARRflx_197908_1724.tar","3HRLY/1979/NARRflx_197908_2531.tar","3HRLY/1979/NARRflx_197909_0108.tar","3HRLY/1979/NARRflx_197909_0916.tar","3HRLY/1979/NARRflx_197909_1724.tar","3HRLY/1979/NARRflx_197909_2530.tar","3HRLY/1979/NARRflx_197910_0108.tar","3HRLY/1979/NARRflx_197910_0916.tar","3HRLY/1979/NARRflx_197910_1724.tar","3HRLY/1979/NARRflx_197910_2531.tar","3HRLY/1979/NARRflx_197911_0108.tar","3HRLY/1979/NARRflx_197911_0916.tar","3HRLY/1979/NARRflx_197911_1724.tar","3HRLY/1979/NARRflx_197911_2530.tar","3HRLY/1979/NARRflx_197912_0108.tar","3HRLY/1979/NARRflx_197912_0916.tar","3HRLY/1979/NARRflx_197912_1724.tar","3HRLY/1979/NARRflx_197912_2531.tar","3HRLY/1980/NARRflx_198001_0108.tar","3HRLY/1979/NARRpbl_197901_0109.tar","3HRLY/1979/NARRpbl_197901_1019.tar","3HRLY/1979/NARRpbl_197901_2031.tar","3HRLY/1979/NARRpbl_197902_0109.tar","3HRLY/1979/NARRpbl_197902_1019.tar","3HRLY/1979/NARRpbl_197902_2028.tar","3HRLY/1979/NARRpbl_197903_0109.tar","3HRLY/1979/NARRpbl_197903_1019.tar","3HRLY/1979/NARRpbl_197903_2031.tar","3HRLY/1979/NARRpbl_197904_0109.tar","3HRLY/1979/NARRpbl_197904_1019.tar","3HRLY/1979/NARRpbl_197904_2030.tar","3HRLY/1979/NARRpbl_197905_0109.tar","3HRLY/1979/NARRpbl_197905_1019.tar","3HRLY/1979/NARRpbl_197905_2031.tar","3HRLY/1979/NARRpbl_197906_0109.tar","3HRLY/1979/NARRpbl_197906_1019.tar","3HRLY/1979/NARRpbl_197906_2030.tar","3HRLY/1979/NARRpbl_197907_0109.tar","3HRLY/1979/NARRpbl_197907_1019.tar","3HRLY/1979/NARRpbl_197907_2031.tar","3HRLY/1979/NARRpbl_197908_0109.tar","3HRLY/1979/NARRpbl_197908_1019.tar","3HRLY/1979/NARRpbl_197908_2031.tar","3HRLY/1979/NARRpbl_197909_0109.tar","3HRLY/1979/NARRpbl_197909_1019.tar","3HRLY/1979/NARRpbl_197909_2030.tar","3HRLY/1979/NARRpbl_197910_0109.tar","3HRLY/1979/NARRpbl_197910_1019.tar","3HRLY/1979/NARRpbl_197910_2031.tar","3HRLY/1979/NARRpbl_197911_0109.tar","3HRLY/1979/NARRpbl_197911_1019.tar","3HRLY/1979/NARRpbl_197911_2030.tar","3HRLY/1979/NARRpbl_197912_0109.tar","3HRLY/1979/NARRpbl_197912_1019.tar","3HRLY/1979/NARRpbl_197912_2031.tar","3HRLY/1980/NARRpbl_198001_0109.tar","3HRLY/1979/NARRsfc_197901_0109.tar","3HRLY/1979/NARRsfc_197901_1019.tar","3HRLY/1979/NARRsfc_197901_2031.tar","3HRLY/1979/NARRsfc_197902_0109.tar","3HRLY/1979/NARRsfc_197902_1019.tar","3HRLY/1979/NARRsfc_197902_2028.tar","3HRLY/1979/NARRsfc_197903_0109.tar","3HRLY/1979/NARRsfc_197903_1019.tar","3HRLY/1979/NARRsfc_197903_2031.tar","3HRLY/1979/NARRsfc_197904_0109.tar","3HRLY/1979/NARRsfc_197904_1019.tar","3HRLY/1979/NARRsfc_197904_2030.tar","3HRLY/1979/NARRsfc_197905_0109.tar","3HRLY/1979/NARRsfc_197905_1019.tar","3HRLY/1979/NARRsfc_197905_2031.tar","3HRLY/1979/NARRsfc_197906_0109.tar","3HRLY/1979/NARRsfc_197906_1019.tar","3HRLY/1979/NARRsfc_197906_2030.tar","3HRLY/1979/NARRsfc_197907_0109.tar","3HRLY/1979/NARRsfc_197907_1019.tar","3HRLY/1979/NARRsfc_197907_2031.tar","3HRLY/1979/NARRsfc_197908_0109.tar","3HRLY/1979/NARRsfc_197908_1019.tar","3HRLY/1979/NARRsfc_197908_2031.tar","3HRLY/1979/NARRsfc_197909_0109.tar","3HRLY/1979/NARRsfc_197909_1019.tar","3HRLY/1979/NARRsfc_197909_2030.tar","3HRLY/1979/NARRsfc_197910_0109.tar","3HRLY/1979/NARRsfc_197910_1019.tar","3HRLY/1979/NARRsfc_197910_2031.tar","3HRLY/1979/NARRsfc_197911_0109.tar","3HRLY/1979/NARRsfc_197911_1019.tar","3HRLY/1979/NARRsfc_197911_2030.tar","3HRLY/1979/NARRsfc_197912_0109.tar","3HRLY/1979/NARRsfc_197912_1019.tar","3HRLY/1979/NARRsfc_197912_2031.tar","3HRLY/1980/NARRsfc_198001_0109.tar"]
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