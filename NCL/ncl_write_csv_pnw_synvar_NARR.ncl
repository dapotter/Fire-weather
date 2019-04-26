;----------------------------------------------------------------------
; write_csv_2.ncl
;
; Concepts illustrated:
;   - Writing a CSV file with a header using write_table
;----------------------------------------------------------------------
; This example reads arrays off a NetCDF file and writes
; the contents to a CSV file with a header that contains the
; long_name and units of each field.
;----------------------------------------------------------------------

begin

; Read groups of NetCDF or Grib1 files in directory:
; This script writes many Grib1 files in a directory to a single CSV
; file.

;~~~~~~~~~~~~~~ ENTER grib1 FILES DIRECTORY HERE ~~~~~~~~~~~~~~~~~
;~~~~~~~~~~~~~~  Directory needs to end with /"  ~~~~~~~~~~~~~~~~~
    
    ;-------  Laptop NARR data directories  -------

    ;cdir = "/home/dp/Documents/FWP/NARR/3D_grib1_exp/"        ; 3D directory
    ;cdir = "/home/dp/Documents/FWP/NARR/3D_grib1_exp/temp/"   ; Another 3D directory with different dates

    ;cdir = "/home/dp/Documents/FWP/NARR/flx_grib1_exp/"       ; flx is PMSL: directory Jan 1-3 files
    ;cdir = "/home/dp/Documents/FWP/NARR/flx_grib1_exp/temp/"

    ;cdir = "/home/dp/Documents/FWP/NARR/pbl_grib1_exp/"       ; pbl is CAPE: directory Jan 1-3 files
    ;cdir = "/home/dp/Documents/FWP/NARR/pbl_grib1_exp/temp/"   ; pbl is CAPE: directory June 1-3 files
    
    ;cdir = "/home/dp/Documents/FWP/NARR/sfc_grib1_exp/"       ; sfc is a number of vars: directory Jan 1-3 files


    ;-------  Seagate NARR data directories  -------
    
    ; cdir = "/mnt/seagate/NARR/3D_grib1/"                    ; Seagate  - NARR data - 3D - H500
    ; cdir = "/mnt/seagate/NARR/flx/"                         ; Seagate  - NARR data - flx - PMSL
    cdir = "/mnt/seagate/NARR/pbl/"                         ; Seagate  - NARR data - pbl - CAPE


    grib_per_csv = 100   ; Set the number of grib files to go into one csv file. Max is 131 due to row limitations.
    

    ; SYNVAR              = "HGT_221_ISBL"
    ; SYNVAR             = "PRMSL_221_MSL"
    SYNVAR             = "CAPE_221_SPDY"
    

    ; SYNABBR            = "H500"
    ; SYNABBR            = "PMSL"
    SYNABBR            = "CAPE"
 

    ; csv_type   = "multiple"     ; Each grib1 file in cdir is written to its own csv file
    ; csv_type   = "single"      ; All grib1 files in cdir are collectively written to one csv
;~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~


    
    ;---Get grib1 file list, print it
    if(SYNVAR .eq. "HGT_221_ISBL") then
        list_cmd = " ; ls *.3D.grib1"
    elseif(SYNVAR .eq. "CAPE_221_SPDY") then
        list_cmd = " ; ls *.pbl.grib1"
    elseif(SYNVAR .eq. "PRMSL_221_MSL") then
        list_cmd = " ; ls *.flx.grib1"
    else
        print("Need to include file type (e.g. 3D, pbl, flx, etc) in if statements")
    end if
    
    cmd = "cd "+cdir+list_cmd
    print(cmd)
    file_list = systemfunc(cmd)
    print("file_list(0)="+file_list(0))
    nfiles = dimsizes(file_list)
    print("nfiles="+nfiles)
    
    ;---Loads all files into f
    f       = addfiles(cdir + file_list, "r")
    
    ;---Selects csv mode. This needs to be moved inside of a
    ;---loop for it to work
    ;if csv_type == "multiple"
    ;    ;csv_filename = cdir + file_list(i) + ".csv"
    ;if csv_type == "single"
    ;    ;csv_filename = cdir + SYNABBR + ".csv"
    
    ;--------------------- Initialize csv file parameters -------------------
    ;---Set csv file name
    SYNNAME_CSV = "0_" + SYNABBR + ".csv"
    
    ;---Add directory to csv file name
    csv_filename = cdir + SYNNAME_CSV
            
    ;---Remove current csv file if it exists
    system("rm -rf " + csv_filename)
            
    ;---Manually set header lines
    field_names = (/ "time", "lon", "lat", SYNABBR /)
    
    ;---Merge column names into comma separated string
    header = [/str_join(field_names,",")/]
    
    ;---Write header into the csv file
    write_table(csv_filename, "w", header, "%s")
    ;------------------------------------------------------------------------
    
    
    endfile = nfiles-1
    c = 0
    do i = 0, nfiles-1
        
        ;---Making csv_filename suffixes.
        ;---i .eq. (number of grib files to pack into one csv file)
        if(i .gt. 0 .and. i%grib_per_csv .eq. 0) then
            ;---Update counter variable
            c = c+1
            print("number of csv files made = "+c)

            ;---Set csv file name
            SYNNAME_CSV = tostring(c) + "_" + SYNABBR + ".csv"
            
            ;---Add directory to csv file name
            csv_filename = cdir + SYNNAME_CSV
            
            ;---Remove current csv file if it exists
            system("rm -rf " + csv_filename)
            
            ;---Manually set header lines
            field_names = (/ "time", "lon", "lat", SYNABBR /)
    
            ;---Merge column names into comma separated string
            header = [/str_join(field_names,",")/]
    
            ;---Write header into the csv file
            write_table(csv_filename, "w", header, "%s")
            
        end if
    
        print("~~~~~~ Currently processing "+file_list(i)+" ~~~~~~")
        
        ;---Printing "0_H500.csv", etc. outside of if loop above so it prints
        ;---on every iteration, not just when the if statement above is true
        print("csv file name: "+SYNNAME_CSV)

        ;---Pick three 3D arrays to write to CSV file
        ;v    = f[i]->HGT_221_ISBL ; selects ith file's Geopotential height
        ;v    = f[i]->CAPE_221_SPDY
        ;v    = f[i]->PRMSL_221_MSL
        v    = f[i]->$SYNVAR$
        ;print("v:"+v(0,0,0:100))

        ;---Selecting SYNVAR over all of North America at 500 mbar level
        ;H500 = v(12,:,:)
        ;CAPE = v(:,:)
        ;PMSL = v(:,:)
        
        ;---Decides level based on synoptic variable abbreviation
        if(SYNABBR .eq. "H500") then
            SYNNAME = v(12,:,:)         ; H500 is at level 12
        else
            SYNNAME = v(:,:)            ; CAPE and PMSL have no levels
        end if

        x    = f[i]->gridlon_221
        y    = f[i]->gridlat_221
        ;t    = H500@initial_time
        ;t    = CAPE@initial_time
        ;t    = PMSL@initial_time
        t    = SYNNAME@initial_time

        ;print("typeof(x):*******************************")
        ;print(typeof(x))

        ;  print(getfileatts(f))
        ;  print("#####################################################")
        ;  print(getfilevardims(f, "HGT_221_ISBL"))
        ;  print("#####################################################")
        ;  print(getfilevaratts(f, "HGT_221_ISBL"))
        ;  print("#####################################################")
        ;  print(H500@initial_time)
        ;  print("#####################################################")
	
        ;dims = dimsizes(t)
        dims = dimsizes(x)
        ;print("x dims:*******************************")
        ;print(dims)

        x_by_y = dims(0)*dims(1)
        ;print("x_by_y product. This should be the length of the 1D arrays:******")
        ;print(x_by_y)
        
        ;---Putting the initial time value into an array for redimensioning:
        t_array = new(1, typeof(t), t)
        ;print("typeof(t):*******************************")
        ;print(typeof(t))
        ;t_array = (/t/) ; array of size 1 ; Same result as the line above.

        ;---Redimensioning to dims. dims is 277 x 349. t is 1x1.
        t_conf = conform_dims(dims, t, 0)
        ;print("t_conf:*******************************")
        ;print(t_conf)

        ;print("=======================================================")
        ;printVarSummary(H500)
        ;print("=======================================================")
        ;printVarSummary(x)
        ;print("=======================================================")
        ;printVarSummary(y)
        ;print("=======================================================")

        ;######################################################################
        ;---------- Subsetting SYNVAR for Oregon and Washington region ----------
        ; Can't call H500 = v(12, -126:-115, 40:51), need index values of the lat
        ; and lon values, not the lat lon values themselves.
        ; e.g.:
        ; H500 = v(level_index, lat_min_index:lat_max_index, lon_min_index:lon_max_index)
        ;----------------------------------------------------------------------
        lat_min =  39  ;  40
        lat_max =  52  ;  51
        lon_min = -127 ; -126
        lon_max = -114 ; -115
        ij      = getind_latlon2d(y, x, (/lat_min,lat_max/),(/lon_min,lon_max/))

        ;---Store to local variables for better code readability
        ilat1 = ij(0,0)
        ilat2 = ij(1,0)
        ilon1 = ij(0,1)
        ilon2 = ij(1,1)

        ;---Subscript variables using these index values
        ;H500_OR_WA = H500(ilat1:ilat2,ilon1:ilon2)      ; 33 x 34
        ;CAPE_OR_WA = CAPE(ilat1:ilat2,ilon1:ilon2)      ; 33 x 34
        ;PMSL_OR_WA = PMSL(ilat1:ilat2,ilon1:ilon2)      ; 33 x 34

        SYNNAME_OR_WA = SYNNAME(ilat1:ilat2,ilon1:ilon2)      ; 33 x 34
        y_OR_WA = y(ilat1:ilat2,ilon1:ilon2)         ; 33 x 34
        x_OR_WA = x(ilat1:ilat2,ilon1:ilon2)         ; 33 x 34
        t_OR_WA = t_conf(ilat1:ilat2,ilon1:ilon2)    ; 33 x 34

        ;   print("H500_OR_WA data:")
        ;   print(H500_OR_WA)
        ;   dims   = dimsizes(H500)
        ;   print(dims)
        ;######################################################################

        ;---Reduce to 1D arrays
        ;H500_OR_WA_1d  = ndtooned(H500_OR_WA)
        ;CAPE_OR_WA_1d  = ndtooned(CAPE_OR_WA)
        ;PMSL_OR_WA_1d  = ndtooned(PMSL_OR_WA)
        

        SYNNAME_OR_WA_1D   = ndtooned(SYNNAME_OR_WA)
        y_OR_WA_1d  = ndtooned(y_OR_WA)
        x_OR_WA_1d  = ndtooned(x_OR_WA)
        t_OR_WA_1d  = ndtooned(t_OR_WA)
        ;print("t_OR_WA_1d:")
        ;print(t_OR_WA_1d)

        ;---Construct header line based on variables selected above.
        ;---Commented out because I'm setting them manually below. This
        ;---code may be useful if changing the variables in the 
        ;field_names = (/ y@long_name  + " ["  + y@units  + "]", \
        ;                 x@long_name  + " ["  + x@units  + "]", \
        ;                 v@long_name  + " ["  + v@units  + "]" /)


        ;---Write data to file
        ;alist  = [/t_OR_WA_1d, x_OR_WA_1d, y_OR_WA_1d, H500_OR_WA_1d/]
        ;alist  = [/t_OR_WA_1d, x_OR_WA_1d, y_OR_WA_1d, CAPE_OR_WA_1d/]
        ;alist  = [/t_OR_WA_1d, x_OR_WA_1d, y_OR_WA_1d, PMSL_OR_WA_1d/]
        alist  = [/t_OR_WA_1d, x_OR_WA_1d, y_OR_WA_1d, SYNNAME_OR_WA_1D/]
        format = "%s,%g,%g,%g"
        print("csv directory: "+csv_filename)
        write_table(csv_filename, "a", alist, format)
        
        ;progress = (c/(n_files-1))*100
        ;print("progress: "+progress)
	    print("progress: "+i+"/"+endfile)
	
    end do
    
end

