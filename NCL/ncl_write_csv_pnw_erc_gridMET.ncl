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
  cdir_nc  = cdir + "nc/"  ; /FWP/gridMET/nc/
  cdir_csv = cdir + "csv/" ; /FWP/gridMET/csv/

;---Select subset
  lat_min =   39
  lat_max =   52
  lon_min =   -114
  lon_max =   -127
  
  list_cmd = "cd "+cdir_nc+" ; ls *.nc"
  print(list_cmd)
  file_list = systemfunc(list_cmd)
  print("file_list(0)="+file_list(0))
  nfiles = dimsizes(file_list)
  print("nfiles = " + nfiles)

  ;---Loads all files into f
  f = addfiles(cdir_nc + file_list, "r")

  do i = 0, nfiles-1
    print("********** Currently processing "+file_list(i)+" **********")

    ;---Pick one 3D array to write to CSV file
    erc    = f[i]->$"energy_release_component-g"$
    printVarSummary(erc)
    d    = f[i]->day

    dims_erc = dimsizes(erc) ; Should be 365 days
    ndays = dims_erc(0)

    do j = 0, ndays-1
      j_string = "_" + tostring(j)
      erc_or_wa := erc(j,{lat_min:lat_max},{lon_min:lon_max})
      print("erc_or_wa var summary:")
      printVarSummary(erc_or_wa)

      ; Read the 1D coordinate arrays (day, lon, lat), 
      ; conform them to 3D,
      ; then convert to 1D so we can write them to a CSV 
      ; file along with data.
      
      dims_erc_or_wa = dimsizes(erc_or_wa)
      print("dimsizes erc_or_wa")
      print(dims_erc_or_wa)

      ; print("dimsizes(erc_or_wa):" + dimsizes(erc_or_wa))
      ; day_conf = conform_dims(dims_erc_or_wa, erc_or_wa&day, 0)
      ; print("~~~~~~~~~~~~~~~~~~~~ day_conf var summary:")
      ; printVarSummary(day_conf)
      lat_conf = conform_dims(dims_erc_or_wa, erc_or_wa&lat, 0)
      print("~~~~~~~~~~~~~~~~~~~~ lat_conf var summary:")
      printVarSummary(lat_conf)
      lon_conf = conform_dims(dims_erc_or_wa, erc_or_wa&lon, 1)
      print("~~~~~~~~~~~~~~~~~~~~ lon_conf var summary:")
      printVarSummary(lon_conf)

      ; Convert 3D array to 1D for writing to CSV
      erc_or_wa_1d  = ndtooned(erc_or_wa)
      ; Convert conformed arrays to 1D for writing to CSV
      ;day1d  = ndtooned(day_conf) ; Not needed, making one CSV per day
      lat1d  = ndtooned(lat_conf)
      lon1d  = ndtooned(lon_conf)
      dims1d = dimsizes(lat1d)
      print(dims1d)
      ; Not sure why day1d isn't being populated with day&day(j),
      ; to deal with this I set the fill value with the day:
      day1d  = new(dimsizes(lat1d), typeof(lat1d), d&day(j)) ; Making one CSV per day
      day1d@_FillValue = d&day(j) ; Explicitly setting the day
      printVarSummary(day1d)
      ; print("******************** day1d var summary:")
      ; printVarSummary(day1d)
      ; print("******************** lat1d var summary:")
      ; printVarSummary(lat1d)
      ; print("******************** lon1d var summary:")
      ; printVarSummary(lon1d)

      ;----------------------------------------------------------------------
      ; This function is writing one CSV per day of ERC data, too much data
      ; to fit into one CSV file.
      ;----------------------------------------------------------------------
      ; Get NC filename, use it to create CSV filename
      nc_filename = file_list(i)
      ;print("nc_filename:" + nc_filename)
      nc_suffix = get_file_suffix(nc_filename,0)
      nc_base = nc_suffix@fBase ; nc file basename, e.g. "erc_1979.nc" base is "erc_1979"
      csv_filename = cdir_csv + nc_base + "_" + tostring(j) + "_ERC.csv" ; cdir_csv tells it where to put the file
      print("******************** csv_filename:" + csv_filename)

      ; Remove current CSV file, make header line, write header to CSV
      system("rm -rf " + csv_filename)
      field_names = (/ "time", "lon", "lat", "erc" /)
      header = [/str_join(field_names,",")/]
      write_table(csv_filename, "w", header, "%s")

      ; Write data to file
      alist  = [/day1d,lon1d,lat1d,erc_or_wa_1d/]
      format = "%g,%g,%g,%g"
      write_table(csv_filename, "a", alist, format)



      /; THIS IS BINARY WRITE, NOT USING. WORKS BUT NOT NUMPY CAN'T READ IT CORRECTLY.
      ;----------------------------------------------------------------------
      ; write_bin_1.ncl
      ;
      ; Concepts illustrated:
      ;   - Using fbinrecwrite to write multiple variables to an unformatted binary file
      ;----------------------------------------------------------------------

      ; The "bin" extension is arbitrary
      file_out = "ERC_OR_WA_binary.bin"
      system ("rm -f " + file_out)  ; remove any previously exist file

      ;************************************************************
      ; note the -1 indicates to add on to the end of the file
      ; the (/.../) syntax means output the values only with no meta
      ; data
      ;************************************************************
      
      ; Putting all variables into one, specify shape
      gridMET = (/ day1d, lon1d, lat1d, erc_or_wa_1d /)
      shape = dimsizes( gridMET );erc_or_wa_1d)
      print("shape of gridMET:")
      print(shape)

      ; **************** WRITING ****************:
      ;
      ; One way of writing to a binary:
      ; fbinrecwrite (file_out,-1, (/ day1d /)) 
      ; fbinrecwrite (file_out,-1, (/ lon1d /))
      ; fbinrecwrite (file_out,-1, (/ lat1d /))
      ; fbinrecwrite (file_out,-1, (/ erc_or_wa_1d /))

      ; A second way of writing to a binary:
      ; fbindirwrite(file_out, day1d)
      ; system("ls -l " + file_out)
      ; fbindirwrite(file_out, lon1d)
      ; system("ls -l " + file_out)
      ; fbindirwrite(file_out, lat1d)
      ; system("ls -l " + file_out)
      ; fbindirwrite(file_out, erc_or_wa_1d)
      ; system("ls -l " + file_out)

      ; A third way of writing to a binary:
      ;fbinrecwrite(file_out, -1, gridMET)
      fbindirwrite(file_out, gridMET)


      ; **************** READING ****************:

      ; One way to read in a binary file. Differences should be equal to 0.
      ; day_r = fbindirread(file_out,0,(/shape(0)/),"integer")
      ; lon_r = fbindirread(file_out,1,(/shape(0)/),"integer")
      ; lat_r = fbindirread(file_out,2,(/shape(0)/),"integer")
      ; erc_or_wa_r = fbindirread(file_out,3,(/shape(0)/),"integer")

      ; A second way to read a binary when data is in one var
      ; fbindirread(file path, 
      ;             record number to start reading from (if unknown use -1
      ;             and it will read in as a singly dimensioned array),
      ;             dimension variable that describes the data,
      ;             string name of the data type of the record)
      ; IF USING fbindirwrite() ABOVE, IT WILL WRITE OUT AS "double",
      ; SPECIFY "double" FOR READ IN WITH fbindirread()
      gridMET_r = fbindirread(file_out,0,(/shape/),"double")

      ; **************** READ WRITE COMPARISON ****************:
      ; print("Difference comparison:")
      ; print(min(day_r - day1d) + "/" + max(day_r - day1d))
      ; print(min(lon_r - lon1d) + "/" + max(lon_r - lon1d))
      ; print(min(lat_r - lat1d) + "/" + max(lat_r - lat1d))
      ; print(min(erc_or_wa_r - erc_or_wa_1d) + "/" + max(erc_or_wa_r - erc_or_wa_1d))

      ; A second way of comparing the data
      print("Difference comparison:")
      print(min(gridMET_r(:,0) - gridMET(:,0)) + "/" + max(gridMET_r(:,0) - gridMET(:,0)))
      print(min(gridMET_r(:,1) - gridMET(:,1)) + "/" + max(gridMET_r(:,1) - gridMET(:,1)))
      print(min(gridMET_r(:,2) - gridMET(:,2)) + "/" + max(gridMET_r(:,2) - gridMET(:,2)))
      print(min(gridMET_r(:,3) - gridMET(:,3)) + "/" + max(gridMET_r(:,3) - gridMET(:,3)))

      printVarSummary(gridMET)
      printVarSummary(gridMET_r)
      ;/
    end do
  end do
end
