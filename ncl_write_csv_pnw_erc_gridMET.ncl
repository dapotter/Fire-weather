;----------------------------------------------------------------------
; write_csv_2.ncl
;
; Concepts illustrated:
;   - Writing a CSV file with a header using write_table
;----------------------------------------------------------------------
; This example reads three 4D arrays off a NetCDF file and writes
; the contents to a CSV file with a header that contains the
; long_name and units of each field.
;----------------------------------------------------------------------

begin
;---NetCDF file to read in
  cdir     = "/home/dp/Documents/FWP/gridMET/"

;---Select subset
  lat_min =   39;39
  lat_max =   40;52
  lon_min =   -114;-114
  lon_max =   -115;-127
  
  list_cmd = "cd "+cdir+" ; ls *.nc"
  print(list_cmd)
  file_list = systemfunc(list_cmd)
  print("file_list(0)="+file_list(0))
  nfiles = dimsizes(file_list)
  print("nfiles = " + nfiles)

  ;---Loads all files into f
  f = addfiles(cdir + file_list, "r")
  
  csv_filename = "ERC_OR_WA.csv"
  system("rm -rf " + csv_filename) ; Removes current csv file if it exists

  ;---Manually set header lines
  field_names = (/ "time", "lon", "lat", "erc" /)

  ;---Merge column names into comma separated string
  header = [/str_join(field_names,",")/]

  write_table(csv_filename, "w", header, "%s")

  endfile = nfiles-1
  do i = 0, nfiles-1
    print("********** Currently processing "+file_list(i)+" **********")

    ;---Pick one 3D array to write to CSV file
    erc    = f[i]->$"energy_release_component-g"$
    printVarSummary(erc)

    erc_or_wa := erc(:,{lat_min:lat_max},{lon_min:lon_max})
    print("erc_or_wa var summary:")
    printVarSummary(erc_or_wa)

    ; Read the 1D coordinate arrays (day, lon, lat), 
    ; conform them to 3D,
    ; then convert to 1D so we can write them to a CSV 
    ; file along with data.
          
    dims = dimsizes(erc_or_wa)
    print("dimsizes(erc_or_wa):" + dimsizes(erc_or_wa))
    day_conf = conform_dims(dims, erc_or_wa&day, 0)
    print("~~~~~~~~~~~~~~~~~~~~ day_conf var summary:")
    printVarSummary(day_conf)
    lat_conf = conform_dims(dims, erc_or_wa&lat, 1)
    print("~~~~~~~~~~~~~~~~~~~~ lat_conf var summary:")
    printVarSummary(lat_conf)
    lon_conf = conform_dims(dims, erc_or_wa&lon, 2)
    print("~~~~~~~~~~~~~~~~~~~~ lon_conf var summary:")
    printVarSummary(lon_conf)

    day1d  = ndtooned(day_conf)
    lat1d  = ndtooned(lat_conf)
    lon1d  = ndtooned(lon_conf)

    print("******************** day1d var summary:")
    printVarSummary(day1d)
    print("******************** lat1d var summary:")
    printVarSummary(lat1d)
    print("******************** lon1d var summary:")
    printVarSummary(lon1d)

    ;---Construct column names for csv
    field_names = (/"time", "lon", "lat", "erc"/)
    header = [/str_join(field_names,",")/]
    print("header:")
    printVarSummary(header)

    ;---Write header to CSV file
    csv_filename = "erc_1979.csv"
    system("rm -rf " + csv_filename)
    write_table(csv_filename, "w", header, "%s")

    ;---Convert 3D array to 1D for writing to CSV file
    erc_or_wa_1d  = ndtooned(erc_or_wa)

    ;---Write data to file
    alist  = [/day1d,lat1d,lon1d,erc_or_wa_1d/]
    format = "%g,%g,%g,%g"
    write_table(csv_filename, "a", alist, format)
  end do
end
