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

    ; CAN ONLY PROCESS ABOUT 1000 netCDF FILES:

    ;cdir = "/home/dp/Documents/FWP/NARR/3D_grib1_exp/"        ; 3D directory
    ; cdir = "/home/dp/Documents/FWP/NARR/3D_grib1_exp/temp/"   ; Another 3D directory with different dates
    ; cdir = "/home/dp/Documents/FWP/NARR/3D_grib1_exp/nc/"     ; netCDF directory
    ; cdir = "/home/dp/Documents/FWP/NARR/3D_grib1_exp/temp/nc/"   ; Another 3D directory with different dates

    ;cdir = "/home/dp/Documents/FWP/NARR/flx_grib1_exp/"       ; flx is PMSL: directory Jan 1-3 files
    ;cdir = "/home/dp/Documents/FWP/NARR/flx_grib1_exp/temp/"
    ; cdir = "/home/dp/Documents/FWP/NARR/flx_grib1_exp/nc/"      ; netCDF directory
    ; cdir = "/home/dp/Documents/FWP/NARR/flx_grib1_exp/temp/nc/"   ; Another 3D directory with different dates

    ;cdir = "/home/dp/Documents/FWP/NARR/pbl_grib1_exp/"       ; pbl is CAPE: directory Jan 1-3 files
    ;cdir = "/home/dp/Documents/FWP/NARR/pbl_grib1_exp/temp/"   ; pbl is CAPE: directory June 1-3 files
    ; cdir = "/home/dp/Documents/FWP/NARR/pbl_grib1_exp/nc/"       ; netCDF directory
    ; cdir = "/home/dp/Documents/FWP/NARR/pbl_grib1_exp/temp/nc/"   ; Another 3D directory with different dates

    ;cdir = "/home/dp/Documents/FWP/NARR/sfc_grib1_exp/"       ; sfc is a number of variables: directory covers Jan 1-3


    ;-------  Seagate NARR data directories  -------
    
    ; cdir =   "/mnt/seagate/NARR/3D/temp/"                    ; Seagate  - NARR data - 3D - H500
    ; cdir = "/mnt/seagate/NARR/flx/temp/"                         ; Seagate  - NARR data - flx - PMSL
    ; cdir = "/mnt/seagate/NARR/pbl/temp/"                         ; Seagate  - NARR data - pbl - CAPE

    cdir = "/mnt/seagate/NARR/1980/3D/temp/nc/"

    grib_per_csv = 100   ; Set the number of grib/netCDF files to go into one csv file. Max is 131 due to csv row limitations.

    ;-------  Grib or netCDF?  -------
    Grib               = False ; If processing Grib files
    netCDF             = True  ; If processing netCDF files

    ;-------  Grib SYNVAR  -------
    SYNVAR             = "HGT_221_ISBL" ; for grib
    SYNVAR             = "SPF_H_221_ISBL" ; for grib
    ; SYNVAR             = "PRMSL_221_MSL" ; for grib
    ; SYNVAR             = "CAPE_221_SPDY" ; for grib

    ;-------  netCDF SYNVAR  -------
    SYNVAR             = "H500" ; for netCDF
    SYNVAR             = "SPFH" ; for netCDF
    ; SYNVAR             = "PMSL" ; for netCDF
    ; SYNVAR             = "CAPE" ; for netCDF

    SYNABBR            = "H500"
    SYNABBR            = "SPFH"
    ; SYNABBR            = "PMSL"
    ; SYNABBR            = "CAPE"

 
    ; csv_type   = "multiple"     ; Each grib1 file in cdir is written to its own csv file
    ; csv_type   = "single"      ; All grib1 files in cdir are collectively written to one csv
;~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    ;---Get grib1 file list or netCDF list, print it
    if(SYNVAR .eq. "HGT_221_ISBL") then ; from grib
        list_cmd = " ; ls *.3D.grib1"
    elseif(SYNVAR .eq. "H500") then ; from netCDF
        list_cmd = " ; ls *.3D.grib1.nc"
    elseif(SYNVAR .eq. "SPF_H_221_ISBL") then ; from grib
        list_cmd = " ; ls *.3D.grib1"
    elseif(SYNVAR .eq. "SPFH") then ; from netCDF
        list_cmd = " ; ls *.3D.grib1.nc"
    elseif(SYNVAR .eq. "PRMSL_221_MSL") then ; from grib
        list_cmd = " ; ls *.flx.grib1"
    elseif(SYNVAR .eq. "PMSL") then ; from netCDF
        list_cmd = " ; ls *.flx.grib1.nc"
    elseif(SYNVAR .eq. "CAPE_221_SPDY") then ; from grib
        list_cmd = " ; ls *.pbl.grib1"
    elseif(SYNVAR .eq. "CAPE") then ; from netCDF
        list_cmd = " ; ls *.pbl.grib1.nc"
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
    f = addfiles(cdir + file_list, "r")
    
    ;--------------------- Initialize csv file parameters -------------------
    ;---Set csv file name
    SYNNAME_CSV = "00_" + SYNABBR + ".csv"
    
    ;---Add directory to csv file name
    csv_filename = cdir + SYNNAME_CSV
            
    ;---Remove current csv file if it exists
    system("rm -rf " + csv_filename)

    ;---Manually set header lines, SYNABBR is "H500" or "PMSL" or "CAPE")
    if(netCDF) then
        SYNABBR_GRAD_X = SYNABBR + " Grad X"
        SYNABBR_GRAD_Y = SYNABBR + " Grad Y"
        field_names = (/ "time", "lon", "lat", SYNABBR, SYNABBR_GRAD_X, SYNABBR_GRAD_Y /)
    else ; If 
        field_names = (/ "time", "lon", "lat", SYNABBR /)
    end if
    
    ;---Merge column names into comma separated string
    header = [/str_join(field_names,",")/]
    
    ;---Write header into the csv file
    write_table(csv_filename, "w", header, "%s")
    ;------------------------------------------------------------------------
    
    endfile = nfiles-1
    c = 0
    do i = 0, nfiles-1

        ;****************************** CSV file writing ****************************
        ;---Jumps into this if statement if the first csv file (e.g. 0_H500.csv)
        ;---is maxed out and will make 1_H500.csv, etc.
        ;---Making csv_filename suffixes.
        ;---i .eq. (number of grib files to pack into one csv file)
        if(i .gt. 0 .and. i%grib_per_csv .eq. 0) then
            ;---Update counter variable
            c = c+1
            print("number of csv files made = "+c)

            ;---Set csv file name
            if(c .lt. 10) then
                ;--- If c is 0 to 9, add a zero prefix
                ;--- Resulting names will be 01_H500.csv, 02_H500.csv, etc.
                ;--- Makes sorting the csv files easier in import_NARR_csv()
                SYNNAME_CSV = tostring(0) + tostring(c) + "_" + SYNABBR + ".csv"
            else
                ;--- If c is >= 10, no need to add a zero prefix
                SYNNAME_CSV = tostring(c) + "_" + SYNABBR + ".csv"
            end if

            ;---Add directory to csv file name
            csv_filename = cdir + SYNNAME_CSV
            
            ;---Remove current csv file if it exists
            system("rm -rf " + csv_filename)
            
            ;---Manually set header lines
            ;---If netCDF derived H500 or PMSL (SYNABBR is "H500" or "PMSL")
            if(netCDF) then
                SYNABBR_GRAD_X = SYNABBR + " Grad X"
                SYNABBR_GRAD_Y = SYNABBR + " Grad Y"
                field_names = (/ "time", "lon", "lat", SYNABBR, SYNABBR_GRAD_X, SYNABBR_GRAD_Y /)
            else
                field_names = (/ "time", "lon", "lat", SYNABBR /)
            end if

            ;---Merge column names into comma separated string
            header = [/str_join(field_names,",")/]
    
            ;---Write header into the csv file
            write_table(csv_filename, "w", header, "%s")
            
        end if
    
        print("~~~~~~ Currently processing "+file_list(i)+" ~~~~~~")
        
        ;---Printing "0_H500.csv", etc. outside of if loop above so it prints
        ;---on every iteration, not just when the if statement above is true
        print("csv file name: "+SYNNAME_CSV)

        ;---Pick the data to write to CSV file
        if(SYNVAR .eq. "HGT_221_ISBL") then
            v    = f[i]->HGT_221_ISBL ; selects ith file's Geopotential height
        elseif(SYNVAR .eq. "SPF_H_221_ISBL") then
            v    = f[i]->SPF_H_221_ISBL ; selects ith file's Geopotential height
        elseif(SYNVAR .eq. "PRMSL_221_MSL") then
            v    = f[i]->PRMSL_221_MSL
        elseif(SYNVAR .eq. "CAPE_221_SPDY") then
            v    = f[i]->CAPE_221_SPDY
        else
            ; If SYNVAR is specified as H500, PMSL or CAPE,
            ; then it came from a netCDF with rectilinear
            ; NARR data, and it contains variables H500,
            ; H500GradX, and H500GradY. Or the PMSL equivalents.
            v               = f[i]->$SYNVAR$
            SYNVAR_GRAD_X   = SYNVAR + "GradX"
            SYNVAR_GRAD_Y   = SYNVAR + "GradY"
            dvdx            = f[i]->$SYNVAR_GRAD_X$
            dvdy            = f[i]->$SYNVAR_GRAD_Y$
            ; print("v:"+v(0,0,0:100))
        end if

        ;---Selecting SYNVAR over all of North America at 500 mbar level
        ;H500 = v(12,:,:)
        ;CAPE = v(:,:)
        ;PMSL = v(:,:)
        
        ;---Decides level based on synoptic variable abbreviation
        if(SYNVAR .eq. "HGT_221_ISBL") then
            SYNNAME = v(12,:,:)         ; H500 is at level 12
        elseif(SYNVAR .eq. "SPF_H_221_ISBL") then
            SYNNAME = v(12,:,:)         ; SPFH is at level 12
        elseif(SYNVAR .eq. "PRMSL_221_MSL" .or. SYNVAR .eq. "CAPE_221_SPDY") then
            SYNNAME = v(:,:)            ; CAPE and PMSL have no levels
        else ; If SYNVAR = H500, PMSL or CAPE
            SYNNAME = v(:,:)
            SYNNAME_GRAD_X = dvdx(:,:)
            SYNNAME_GRAD_Y = dvdy(:,:)
        end if

        x    = f[i]->lon
        y    = f[i]->lat
        t    = SYNNAME@initial_time
	
        dims = dimsizes(v)
        ;print("SYNVAR dims:*******************************")
        ;print(dims)

        ; THIS WAS NEVER DOING ANYTHING
        x_by_y = dims(0)*dims(1)
        ; print("x_by_y product. This should be the length of the 1D arrays:******")
        ; print(x_by_y)

        ;---Redimensioning to dims. dims is 277 x 349. t is 1x1.
        t_conf = conform_dims(x_by_y, t, 0); t_conf = conform_dims(dims, t, 0)
        ;print("t_conf:*******************************")
        ;print(t_conf)

        ;---Recall that the latitude (y) dimension is the first dimension in
        ;---the H500 nc file
        y_conf = conform_dims(dims, y, 0)
        x_conf = conform_dims(dims, x, 1)

        SYNNAME_OR_WA = SYNNAME
        ;---Assumes gradients are present. If not, this will error.
        if(netCDF) then
            SYNNAME_GRAD_X_OR_WA = SYNNAME_GRAD_X
            SYNNAME_GRAD_Y_OR_WA = SYNNAME_GRAD_Y
        end if

        y_OR_WA = y_conf
        x_OR_WA = x_conf
        t_OR_WA = t_conf
        
        SYNNAME_OR_WA_1D   = ndtooned(SYNNAME_OR_WA)
        if(netCDF) then
            SYNNAME_GRAD_X_OR_WA_1D   = ndtooned(SYNNAME_GRAD_X_OR_WA)
            SYNNAME_GRAD_Y_OR_WA_1D   = ndtooned(SYNNAME_GRAD_Y_OR_WA)
        end if

        y_OR_WA_1d  = ndtooned(y_OR_WA)
        x_OR_WA_1d  = ndtooned(x_OR_WA)
        t_OR_WA_1d  = ndtooned(t_OR_WA)
        ;print("t_OR_WA_1d:")
        ;print(t_OR_WA_1d)

        ;---Write data to file        
        if(netCDF) then ; for H500, PMSL or CAPE
            alist  = [/t_OR_WA_1d, x_OR_WA_1d, y_OR_WA_1d, SYNNAME_OR_WA_1D, SYNNAME_GRAD_X_OR_WA_1D, SYNNAME_GRAD_Y_OR_WA_1D/]
        else ; for CAPE
            alist  = [/t_OR_WA_1d, x_OR_WA_1d, y_OR_WA_1d, SYNNAME_OR_WA_1D/]
        end if

        format = "%s,%g,%g,%g,%g,%g"
        print("csv directory: "+csv_filename)
        write_table(csv_filename, "a", alist, format)
        
	    print("progress: "+i+"/"+endfile)
	
    end do
    
end

