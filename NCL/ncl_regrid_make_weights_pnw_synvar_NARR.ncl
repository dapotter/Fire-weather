;======================================================================
; ESMF_wgts_30a.ncl
;
; Concepts illustrated:
;   - Generate ESMF weight files:
;     (a)  NARR to Rectilinear
;     (b)  Rectilinear to NARR
;     (c)  Create a difference plot showing the difference
;          between the the grid generated in (b) and the source NARR grid.
;======================================================================
;load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/gsn_code.ncl"  ; not needed for 6.1.0 onward
;load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/gsn_csm.ncl"

load "$NCARG_ROOT/lib/ncarg/nclscripts/esmf/ESMF_regridding.ncl"

begin

;======================================================================
; Part A: Generate NARR to Rectilinear regrid weights
;         Note: The NARR has grid points with missing values
;======================================================================
;~~~~~~~~~~~~~~ ENTER grib1 FILES DIRECTORY HERE ~~~~~~~~~~~~~~~~~
;~~~~~~~~~~~~~~  Directory needs to end with /"  ~~~~~~~~~~~~~~~~~
    
    ;-------------------  NOTE  -------------------
    ; In the directory where the grib files are located
    ; make a folder and name it 'nc'. The netCDF files
    ; will exported here.


    ;-------  Laptop NARR data directories  -------

    ; cdir = "/home/dp/Documents/FWP/NARR/3D_grib1_exp/"        ; 3D directory
    ; cdir = "/home/dp/Documents/FWP/NARR/3D_grib1_exp/temp/"   ; Another 3D directory with different dates
    ; cdir = "/home/dp/Documents/FWP/NARR/3D_grib1_exp/test/"   ; Another 3D directory with different dates

    ; cdir = "/home/dp/Documents/FWP/NARR/flx_grib1_exp/"        ; flx is PMSL: directory Jan 1-3 files
    ; cdir = "/home/dp/Documents/FWP/NARR/flx_grib1_exp/temp/"

    ; cdir = "/home/dp/Documents/FWP/NARR/pbl_grib1_exp/"       ; pbl is CAPE: directory Jan 1-3 files
    ; cdir = "/home/dp/Documents/FWP/NARR/pbl_grib1_exp/temp/"   ; pbl is CAPE: directory June 1-3 files
    
    ;cdir = "/home/dp/Documents/FWP/NARR/sfc_grib1_exp/"       ; sfc is a number of vars: directory Jan 1-3 files


    ;-------  Seagate NARR data directories  -------

    ; cdir = "/mnt/seagate/NARR/3D/"                          ; Seagate  - NARR data - 3D - H500
    ; cdir = "/mnt/seagate/NARR/flx/"                         ; Seagate  - NARR data - flx - PMSL
    ; cdir = "/mnt/seagate/NARR/flx/temp/"                    ; Seagate  - NARR data - flx - PMSL
    ; cdir = "/mnt/seagate/NARR/pbl/temp/"                    ; Seagate  - NARR data - pbl - CAPE
    cdir = "/mnt/seagate/NARR/1980/3D/temp/"
    
    ; If exporting one or multiple variables to netCDF files:
    single               = False ; Processing single var...
    multi                = True  ; or multi var?

    SYNVAR               = "HGT_221_ISBL" ; This is always set
    SYNVAR_2             = "TMP_221_ISBL" ; SYNVAR_2 to SYNVAR_5 only matter if multi = True
    SYNVAR_3             = "V_VEL_221_ISBL"
    SYNVAR_4             = "SPF_H_221_ISBL"
    SYNVAR_5             = "CLWMR_221_ISBL"

    ; SYNVAR             = "PRMSL_221_MSL"
    ; SYNVAR             = "CAPE_221_SPDY"


    lvl         = 500                                  ; arbitrary pressure level for H500


    SYNABBR              = "H500"         ; Geopotential height, gpm
    SYNABBR_2            = "TEMP"         ; Temperature, K
    SYNABBR_3            = "PVEL"         ; Pressure vertical velocity, Pa/s
    SYNABBR_4            = "SPFH"         ; Specific humidity, kg/kg
    SYNABBR_5            = "CWTR"         ; Cloud water, kg/kg
    
    ; SYNABBR            = "PMSL"
    ; SYNABBR            = "CAPE"
 
    ; csv_type   = "multiple"     ; Each grib1 file in cdir is written to its own csv file
    ; csv_type   = "single"      ; All grib1 files in cdir are collectively written to one csv
;~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    ;---Setting Gradient strings. Calculating gradients for all variables.
    SYNABBR_GRAD_X = SYNABBR+"GradX"
    SYNABBR_GRAD_Y = SYNABBR+"GradY"
    
    ;---Setting Gradient attribute strings to be written to dxdlon and dxdlat
    if(SYNABBR .eq. "H500") then
        SYNVAR_GRAD_X_UNITS = "gpm/deg"
        SYNVAR_GRAD_Y_UNITS = "gpm/deg"
        SYNVAR_GRAD_X_LONG_NAME = "Geopotential height lon gradient"
        SYNVAR_GRAD_Y_LONG_NAME = "Geopotential height lat gradient"
    ; elseif(SYNABBR .eq. "SPFH") then
    ;     SYNVAR_GRAD_X_UNITS = "kg/kg"
    ;     SYNVAR_GRAD_Y_UNITS = "kg/kg"
    ;     SYNVAR_GRAD_X_LONG_NAME = "Specific humidity lon gradient"
    ;     SYNVAR_GRAD_Y_LONG_NAME = "Specific humidity lat gradient"
    elseif(SYNABBR .eq. "PMSL") then
        SYNVAR_GRAD_X_UNITS = "Pa/deg"
        SYNVAR_GRAD_Y_UNITS = "Pa/deg"
        SYNVAR_GRAD_X_LONG_NAME = "Pressure lon gradient"
        SYNVAR_GRAD_Y_LONG_NAME = "Pressure lat gradient" 
    elseif(SYNABBR .eq. "CAPE") then
        SYNVAR_GRAD_X_UNITS = "J/kg-deg"
        SYNVAR_GRAD_Y_UNITS = "J/kg-deg"
        SYNVAR_GRAD_X_LONG_NAME = "Convective energy lon gradient"
        SYNVAR_GRAD_Y_LONG_NAME = "Convective energy lat gradient" 
    end if

    cmd = "cd "+cdir+" ; ls *.grib1"

    print(cmd)
    file_list = systemfunc(cmd)
    print("*********************************************file_list(0)="+file_list(0))
    nfiles = dimsizes(file_list)
    print("nfiles="+nfiles)
    
    ;---Loads all files into f
    f = addfiles(cdir + file_list, "r")

    ;---Specify directories for netCDF export and then import for plotting
    diri_name   = cdir
    diro_name   = cdir + "nc/"

    endfile = nfiles-1
    c = 0
    do i = 0, nfiles-1
        print("~~~~~~ Currently processing "+file_list(i)+" ~~~~~~")

    ;---This loops converts all grib1 into nc                     
    ;---Input file
        fili_name  = file_list(i) ; "merged_AWIP32.1979010100.3D.grib1"
        fili  =  diri_name + fili_name
        filo  =  diro_name + fili_name + ".nc" ; output netCDF
        
    ;---Get the variable to be regridded;  only need one level to generate the weight
    ;---Also, the grid coordinates

    ;---Pick the data to write to netCDF
        if(SYNVAR .eq. "HGT_221_ISBL") then
            x    = f[i]->HGT_221_ISBL({lvl},:,:)    ;  ( gridx_221, gridy_221) ; (   277,349)
        else
            ; If CAPE or PMSL SYNVAR, no levels to specify
            x    = f[i]->$SYNVAR$
            ;print("x:"+x(0,0,0:100))
        end if

        lat2d = f[i]->gridlat_221                ;  (gridx_221, gridy_221)
        lon2d = f[i]->gridlon_221

        nmsg  = num(ismissing(x))                   ; # of msg values
        printVarSummary(x)
        print("x: min="+min(x)+"   max="+max(x)+"    nmsg="+nmsg)
        print(" ")
        print("lat2d: min="+min(lat2d)+"   max="+max(lat2d))
        print("lon2d: min="+min(lon2d)+"   max="+max(lon2d))

        lon2d   = where(lon2d.lt.0, lon2d+360, lon2d) ; graphical convenience
        print("lon2d: min="+min(lon2d)+"   max="+max(lon2d))

        x@lat2d = lat2d                            ; These attributes will be used by
        x@lon2d = lon2d                            ; ESMF_regrid for the source grid

        ; WARNING: KEEP AN EYE ON MISSING VALUES. IF THERE ARE DISAGREEMENTS BETWEEN MISSING
        ; VALUES OF x, x2, x3, x4, OR x5 THEN THIS FUNCTION WILL PRODUCE INCORRECT DATA.
        if(multi) then
            x2    = f[i]->TMP_221_ISBL({lvl},:,:)    ;  ( gridx_221, gridy_221) ; (   277,349)
            x3    = f[i]->V_VEL_221_ISBL({lvl},:,:)    ;  ( gridx_221, gridy_221) ; (   277,349)
            x4    = f[i]->SPF_H_221_ISBL({lvl},:,:)    ;  ( gridx_221, gridy_221) ; (   277,349)
            x5    = f[i]->CLWMR_221_ISBL({lvl},:,:)    ;  ( gridx_221, gridy_221) ; (   277,349)

            nmsg_2  = num(ismissing(x2))                   ; # of msg values
            nmsg_3  = num(ismissing(x3))                   ; # of msg values
            nmsg_4  = num(ismissing(x4))                   ; # of msg values
            nmsg_5  = num(ismissing(x5))                   ; # of msg values
            printVarSummary(x2)
            printVarSummary(x3)
            printVarSummary(x4)
            printVarSummary(x5)
            print("x2: min="+min(x2)+"   max="+max(x2)+"    nmsg="+nmsg_2)
            print("x3: min="+min(x3)+"   max="+max(x3)+"    nmsg="+nmsg_3)
            print("x4: min="+min(x4)+"   max="+max(x4)+"    nmsg="+nmsg_4)
            print("x5: min="+min(x5)+"   max="+max(x5)+"    nmsg="+nmsg_5)
            
            x2@lat2d = lat2d                            ; These attributes will be used by
            x2@lon2d = lon2d                            ; ESMF_regrid for the source grid

            x3@lat2d = lat2d                            ; These attributes will be used by
            x3@lon2d = lon2d                            ; ESMF_regrid for the source grid

            x4@lat2d = lat2d                            ; These attributes will be used by
            x4@lon2d = lon2d                            ; ESMF_regrid for the source grid

            x5@lat2d = lat2d                            ; These attributes will be used by
            x5@lon2d = lon2d                            ; ESMF_regrid for the source grid

        end if
        
    ;---Create the destination rectilinear lat[*]/lon[*] arrays. 
    ;---Here, roughly the same resolution as the source grid at 45N.
    ;---This is arbitrary. It can be whatever the user desires.
    ;---For example, it could be just the area of the USA.
    ;---Create rectilinear coordinates; monotonically increasing

        nlat  = 52
        nlon  = 52
        lat   = fspan(  39.0,  52.0, nlat)
        lon   = fspan( 233.0, 246.0, nlon)

    ;---For netCDF

        lat!0         = "lat"
        lat@units     = "degrees_north"
        lat&lat       = lat

        lon!0         = "lon"
        lon@units     = "degrees_east"
        lon&lon       = lon

        ;lev!0         = "lev"
        ;lev&lev       = lev

        if(i .eq. 0) then
        ;--- Only create the ESMF weights file once during the
        ;--- first file processing
        ;---Create regrid options

            InterpMethod = "bilinear"                           ; "bilinear", "conserve"
            Opt                = True
            Opt@SrcTitle       = "NARR grid"   ; optional
            Opt@WgtFileName    = "NARR_to_Rect.WgtFile_"+InterpMethod+".nc"

        ;---Generate the names of the files containing the source and destination grid descriptions
        ;---Remove after Part A is complete

            Opt@SrcFileName    = "NARR.SCRIP_grid_description.nc"         ; Name of source and
            Opt@SrcRegional    = True
            
        ;---If source data contains missing values, set the 
        ;---special SrcMask2D option to indicate the missing values 

            Opt@SrcMask2D  = where(ismissing(x),0,1)

            DstDirName         = "./"
            Opt@DstFileName    = DstDirName + "Rectilinear.SCRIP_grid_description.nc" 
            Opt@DstGridType    = "rectilinear"
            Opt@DstGridLat     = lat
            Opt@DstGridLon     = lon
            Opt@DstRegional    = True

        ;---Specify other options

            Opt@ForceOverwrite = True
            Opt@InterpMethod   = InterpMethod
            Opt@RemoveSrcFile  = False; True                  ; remove SCRIP grid destination files
            Opt@RemoveDstFile  = False; True
            Opt@RemoveWgtFile  = False; True
            Opt@NoPETLog       = False; True                  ; 6.2.1 onward
        end if

    ;---Perform the regrid: NARR ==> rectilinear (_reclin)
        print("________________________________________________Creating x rectilinear...")
        x_reclin = ESMF_regrid(x, Opt)             ; Do the regridding for x

        mmsg  = num(ismissing(x_reclin))           ; # of msg values
        print("x_reclin before adding file dimensions and attributes: ****************************")
        printVarSummary(x_reclin)
        print("x_reclin: min="+min(x_reclin)+"   max="+max(x_reclin)+"    mmsg="+mmsg)
        if(multi) then
            x2_reclin = ESMF_regrid(x2, Opt)
            x3_reclin = ESMF_regrid(x3, Opt)
            x4_reclin = ESMF_regrid(x4, Opt)
            x5_reclin = ESMF_regrid(x5, Opt)

            mmsg_2  = num(ismissing(x2_reclin))           ; # of msg values
            mmsg_3  = num(ismissing(x3_reclin))           ; # of msg values
            mmsg_4  = num(ismissing(x4_reclin))           ; # of msg values
            mmsg_5  = num(ismissing(x5_reclin))           ; # of msg values
            print("x2_reclin: min="+min(x2_reclin)+"   max="+max(x2_reclin)+"    mmsg="+mmsg_2)
            print("x3_reclin: min="+min(x3_reclin)+"   max="+max(x3_reclin)+"    mmsg="+mmsg_3)
            print("x4_reclin: min="+min(x4_reclin)+"   max="+max(x4_reclin)+"    mmsg="+mmsg_4)
            print("x5_reclin: min="+min(x5_reclin)+"   max="+max(x5_reclin)+"    mmsg="+mmsg_5)
        end if


        Opt@ForceOverwrite = False
        Opt@SkipSrcGrid    = True
        Opt@SkipDstGrid    = True
        Opt@SkipWgtGen     = True

    ;---For netCDF
        print("x_reclin after adding file dimensions and attributes: ****************************")
        printVarSummary(x_reclin)
        ;printVarSummary(lat)
        ;printVarSummary(lon)

    ;---Calculate gradients of x_reclin

    ;---Only calculate gradients for H500 and PMSL
        dxdlon = center_finite_diff_n(x_reclin, lon, False, 0, 1)
        dxdlat = center_finite_diff_n(x_reclin, lat, False, 0, 0)
        ;print(dxdlon)
        ;print(dxdlat)

    ;---Name and assign dimensions to the gradient variables:
        dxdlon!0    = "lat"      ; name the dimension
        dxdlon&lat  = lat      ; assign lat to lat dimension
        dxdlon!1    = "lon"      ; name the dimension
        dxdlon&lon  = lon      ; assign lat to lat dimension
        dxdlon@lv_ISBL0 = lvl
        dxdlon@units = SYNVAR_GRAD_X_UNITS ; "gpm/deg", "Pa/deg", "J/kg-deg"
        dxdlon@long_name = SYNVAR_GRAD_X_LONG_NAME ; e.g. "Geopotential height lon gradient"
        dxdlon@time = x@initial_time

        dxdlat!0    = "lat"      ; name the dimension
        dxdlat&lat  = lat      ; assign lat to lat dimension
        dxdlat!1    = "lon"      ; name the dimension
        dxdlat&lon  = lon      ; assign lat to lat dimension
        dxdlat@lv_ISBL0 = lvl
        dxdlat@units = SYNVAR_GRAD_Y_UNITS
        dxdlat@long_name = SYNVAR_GRAD_Y_LONG_NAME
        dxdlat@time = x@initial_time

        printVarSummary(dxdlat)
        printVarSummary(dxdlon)


        ;------------------------------------- EXPORT netCDF -------------------------------------

        ;---------------------------------------------------
        ; Create netCDF: http://www.ncl.ucar.edu/Applications/method_2.shtml
        ;---------------------------------------------------

        system ("'rm' -f "+filo)
        ncdf  = addfile(filo,"c")

        ;---------------------------------------------------
        ; create global attributes of the file
        ;---------------------------------------------------
        fAtt               = True            ; assign file attributes
        fAtt@title         = "NARR: ESMF weighted regrid to 0.25 deg rectilinear"
        fAtt@source        = "The North American Regional Reanalysis (NARR) Project"
        fAtt@analysis_time = x@initial_time
        fAtt@input_file    = fili_name
        fAtt@Conventions   = "None"
        fAtt@creation_date = systemfunc("date")

        fileattdef( ncdf, fAtt )            ; copy file attributes

        ;---------------------------------------------------
        ; dimension names/sizes/attributes
        ;---------------------------------------------------
        dimNames = (/ "lat", "lon"/)
        ; dimSizes = (/  klev , NLAT,  MLON /)
        dimSizes = (/  nlat,  nlon /)
        dimUnlim = (/ False, False /)
        filedimdef(ncdf,dimNames,dimSizes,dimUnlim)


        ; filevardef(ncdf, "lev"   ,typeof(lev)  , getvardims(lev))
        filevardef(ncdf, "lat"      ,typeof(lat)     , getvardims(lat))
        filevardef(ncdf, "lon"      ,typeof(lon)     , getvardims(lon))
        filevardef(ncdf, SYNABBR     ,typeof(x_reclin), getvardims(x_reclin))
        filevardef(ncdf, SYNABBR_GRAD_X   ,typeof(dxdlon)  , getvardims(dxdlon))
        filevardef(ncdf, SYNABBR_GRAD_Y   ,typeof(dxdlat)  , getvardims(dxdlat))

        ;---------------------------------------------------
        ; Copy attributes associated with each variable to the file
        ; All attributes associated with each variable will be copied.
        ;---------------------------------------------------
        ; filevarattdef(ncdf, "lev"  ,lev  )                   ; copy lev attributes
        filevarattdef(ncdf, "lat"   ,lat  )                   ; copy lat attributes
        filevarattdef(ncdf, "lon"   ,lon  )                   ; copy lon attributes
        filevarattdef(ncdf, SYNABBR  ,x_reclin )
        filevarattdef(ncdf, SYNABBR_GRAD_X, dxdlon )
        filevarattdef(ncdf, SYNABBR_GRAD_Y, dxdlat )

        ;---------------------------------------------------
        ; Write values to the file
        ;---------------------------------------------------
        ; ncdf->lev    = (/ lev /)
        ncdf->lat         = (/ lat /)
        ncdf->lon         = (/ lon /)
        ncdf->$SYNABBR$   = (/ x_reclin /)
        ncdf->$SYNABBR_GRAD_X$ = (/ dxdlon /)
        ncdf->$SYNABBR_GRAD_Y$ = (/ dxdlat /)

        print("progress: "+i+"/"+endfile)

        if(multi) then
            filevardef(ncdf, SYNABBR_2     ,typeof(x2_reclin), getvardims(x2_reclin))
            filevardef(ncdf, SYNABBR_3     ,typeof(x3_reclin), getvardims(x3_reclin))
            filevardef(ncdf, SYNABBR_4     ,typeof(x4_reclin), getvardims(x4_reclin))
            filevardef(ncdf, SYNABBR_5     ,typeof(x5_reclin), getvardims(x5_reclin))
            ; filevardef(ncdf, SYNABBR_GRAD_X   ,typeof(dxdlon)  , getvardims(dxdlon))
            ; filevardef(ncdf, SYNABBR_GRAD_Y   ,typeof(dxdlat)  , getvardims(dxdlat))

            ;---------------------------------------------------
            ; Copy attributes associated with each variable to the file
            ; All attributes associated with each variable will be copied.
            ;---------------------------------------------------
            ; filevarattdef(ncdf, "lat"   ,lat  )                   ; copy lat attributes
            ; filevarattdef(ncdf, "lon"   ,lon  )                   ; copy lon attributes
            filevarattdef(ncdf, SYNABBR_2  , x2_reclin )
            filevarattdef(ncdf, SYNABBR_3  , x3_reclin )
            filevarattdef(ncdf, SYNABBR_4  , x4_reclin )
            filevarattdef(ncdf, SYNABBR_5  , x5_reclin )
            ; filevarattdef(ncdf, SYNABBR_GRAD_X, dxdlon )
            ; filevarattdef(ncdf, SYNABBR_GRAD_Y, dxdlat )

            ;---------------------------------------------------
            ; Write values to the file
            ;---------------------------------------------------
            ; ncdf->lat         = (/ lat /)
            ; ncdf->lon         = (/ lon /)
            ncdf->$SYNABBR_2$   = (/ x2_reclin /)
            ncdf->$SYNABBR_3$   = (/ x3_reclin /)
            ncdf->$SYNABBR_4$   = (/ x4_reclin /)
            ncdf->$SYNABBR_5$   = (/ x5_reclin /)
            ; ncdf->$SYNABBR_GRAD_X$ = (/ dxdlon /)
            ; ncdf->$SYNABBR_GRAD_Y$ = (/ dxdlat /)

            print("progress: "+i+"/"+endfile)
        
        end if

    end do




    ;============================= Import netCDF and plot ==============================
    ; This function makes three plots: H500, H500 Grad X, H500 Grad Y
    ; They are titled "H500_OR_WA.png", "H500_Grad_X_OR_WA.png", etc
    ;----------------------------------------------------------------------
    ; latlon_subset_1.ncl
    ;----------------------------------------------------------------------
    ; Concepts illustrated:
    ;   - Using coordinate subscripting to extract a lat/lon region
    ;   - Subsetting a rectilinear grid
    ;   - Drawing a lat/lon grid using gsn_coordinates
    ;   - Attaching polymarkers to a map
    ;   - Using lonFlip to convert 0 to 360 longitudes to -180 to 180
    ;   - Zooming in on a particular area on a map
    ;----------------------------------------------------------------------
    ; The data file for this example can be downloaded from
    ; http://www.ncl.ucar.edu/Applications/Data/#cdf
    ;
    ; For an example of subsetting data represented by 2D lat/lon arrays,
    ; see latlon_subset_2.ncl and the "getind_latlon2d" function.
    ;----------------------------------------------------------------------
    ; These files are loaded by default in NCL V6.2.0 and newer
    ; load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/gsn_code.ncl"
    ; load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/gsn_csm.ncl"

    ;---Read in netCDF file:
    ;a  = addfile("/home/dp/Documents/FWP/NARR/3D_grib1_exp/merged_AWIP32.1979010100.3D.nc", "r")
    ;---This is the last file out from the loop above
    a  = addfile(filo,"r")

    fnames  = getfilevarnames(a)
    print(fnames)

    ; NOTE: USING v FOR VARIABLE AND NOT x. x ALREADY EXISTS AND WON'T ALLOW ITSELF TO BE
    ; DYNAMICAL OVERWRITTEN. IT MUST BE EXPLICITLY DELETED. IT CAN BE DELECTED AT THIS POINT.
    ; NEED TO DELETE THEM HERE BECAUSE I DON'T WANT THE REMAINING VARIABLES FROM THE LAST 
    ; REGRIDDING OPERATION
    delete(x_reclin)
    if(multi) then
        delete(x2_reclin)
        delete(x3_reclin)
        delete(x4_reclin)
        delete(x5_reclin)
    end if

    x_reclin = a->$SYNABBR$
    printVarSummary(x_reclin)
    if(multi) then
        x2_reclin = a->$SYNABBR_2$
        x3_reclin = a->$SYNABBR_3$
        x4_reclin = a->$SYNABBR_4$
        x5_reclin = a->$SYNABBR_5$
        printVarSummary(x2_reclin)
        printVarSummary(x3_reclin)
        printVarSummary(x4_reclin)
        printVarSummary(x5_reclin)
    end if

    ;---Deletes from the last H500 or PMSL file converted to netCDF above:
    delete(dxdlon)
    delete(dxdlat)
    ;---Reassigns from netCDF:
    dxdlon = a->$SYNABBR_GRAD_X$
    dxdlat = a->$SYNABBR_GRAD_Y$
    printVarSummary(dxdlon)
    printVarSummary(dxdlat)

    ;---Save resulting plot as H500_OR_WA.png, PMSL_OR_WA.png:
    wks = gsn_open_wks("png", SYNABBR+"_OR_WA")
    if(multi) then
        wks2 = gsn_open_wks("png", SYNABBR_2+"_OR_WA")
        wks3 = gsn_open_wks("png", SYNABBR_3+"_OR_WA")
        wks4 = gsn_open_wks("png", SYNABBR_4+"_OR_WA")
        wks5 = gsn_open_wks("png", SYNABBR_5+"_OR_WA")
    end if

    ;---Specify lon lat subset:
    lat_min =   39
    lat_max =   52
    lon_min =   233;-114
    lon_max =   246;-127

    ;---H500 or PMSL or CAPE:
    x_reclin_sub  := x_reclin({lat_min:lat_max},{lon_min:lon_max})
    printVarSummary(x_reclin_sub)
    if(multi) then
        x2_reclin_sub  := x2_reclin({lat_min:lat_max},{lon_min:lon_max})
        x3_reclin_sub  := x3_reclin({lat_min:lat_max},{lon_min:lon_max})
        x4_reclin_sub  := x4_reclin({lat_min:lat_max},{lon_min:lon_max})
        x5_reclin_sub  := x5_reclin({lat_min:lat_max},{lon_min:lon_max})
        printVarSummary(x2_reclin_sub)
        printVarSummary(x3_reclin_sub)
        printVarSummary(x4_reclin_sub)
        printVarSummary(x5_reclin_sub)
    end if

    ;---Setting some resources
    res                 = True
    res@gsnMaximize     = True     ; maximize plot in frame
    res@cnFillOn        = True     ; turn on contour fill
    res@cnLinesOn       = False    ; turn off contour lines
    res@cnLineLabelsOn  = False    ; turn off contour line labels
    res@tiMainString    = "Plotting lat/lon subset of data"
    res@pmTitleZone     = 4        ; Moves title down
    res@gsnAddCyclic    = False    ; The subset lon data doesn't extend 360 degress (not cyclical)
    res@cnLevelSelectionMode = "AutomaticLevels"     ; set manual contour levels

    ;---Recalculate values for new min/max lat/lon and create new plot
    res@gsnDraw       = False 
    res@gsnFrame      = False
    res@mpMinLatF     = min(x_reclin_sub&lat)-2
    res@mpMaxLatF     = max(x_reclin_sub&lat)+2
    res@mpMinLonF     = min(x_reclin_sub&lon)-2
    res@mpMaxLonF     = max(x_reclin_sub&lon)+2
    res@mpCenterLonF  = (res@mpMinLonF + res@mpMaxLonF) / 2.
    res@pmTickMarkDisplayMode = "Always"  ; nicer map tickmarks

    ;---Plot:
    plt = gsn_csm_contour_map(wks, x_reclin_sub, res)
    draw(plt)
    frame(wks)
    ; WARNING: USING res FROM x_reclin. THIS MAY CAUSE PROBLEMS.
    if(multi) then
        plt2 = gsn_csm_contour_map(wks2, x2_reclin_sub, res)
        draw(plt2)
        frame(wks2)
        plt3 = gsn_csm_contour_map(wks3, x3_reclin_sub, res)
        draw(plt3)
        frame(wks3)
        plt4 = gsn_csm_contour_map(wks4, x4_reclin_sub, res)
        draw(plt4)
        frame(wks4)
        plt5 = gsn_csm_contour_map(wks5, x5_reclin_sub, res)
        draw(plt5)
        frame(wks5)
    end if


    ;---X GRADIENT: H500, PMSL or CAPE Grad X:
    wks = gsn_open_wks("png", SYNABBR+"_Grad_X_OR_WA")

    dxdlon_sub := dxdlon({lat_min:lat_max},{lon_min:lon_max})
    printVarSummary(dxdlon_sub)
    ;---Setting some resources
    res                 = True
    res@gsnMaximize     = True     ; maximize plot in frame
    res@cnFillOn        = True     ; turn on contour fill
    res@cnLinesOn       = False    ; turn off contour lines
    res@cnLineLabelsOn  = False    ; turn off contour line labels
    res@tiMainString    = "Plotting lat/lon subset of zonal gradient"
    res@pmTitleZone     = 4        ; Moves title down
    res@gsnAddCyclic    = False    ; The subset lon data doesn't extend 360 degress (not cyclical)
    res@cnLevelSelectionMode = "AutomaticLevels"     ; set manual contour levels

    ;---Recalculate values for new min/max lon/lon and create new plot
    res@gsnDraw       = False 
    res@gsnFrame      = False
    res@mpMinLatF     = min(dxdlon_sub&lat)-2
    res@mpMaxLatF     = max(dxdlon_sub&lat)+2
    res@mpMinLonF     = min(dxdlon_sub&lon)-2
    res@mpMaxLonF     = max(dxdlon_sub&lon)+2
    res@mpCenterLonF  = (res@mpMinLonF + res@mpMaxLonF) / 2.
    res@pmTickMarkDisplayMode = "Always"  ; nicer map tickmarks

    ;---Plot:
    plt = gsn_csm_contour_map(wks, dxdlon_sub, res)
    draw(plt)
    frame(wks)


    ;---Y GRADIENT: H500 or PMSL Grad Y:
    wks = gsn_open_wks("png", SYNABBR+"_Grad_Y_OR_WA")

    dxdlat_sub  := dxdlat({lat_min:lat_max},{lon_min:lon_max})
    printVarSummary(dxdlat_sub)
    ;---Setting some resources
    res                 = True
    res@gsnMaximize     = True     ; maximize plot in frame
    res@cnFillOn        = True     ; turn on contour fill
    res@cnLinesOn       = False    ; turn off contour lines
    res@cnLineLabelsOn  = False    ; turn off contour line labels
    res@tiMainString    = "Plotting lat/lon subset of meridional gradient"
    res@pmTitleZone     = 4        ; Moves title down
    res@gsnAddCyclic    = False    ; The subset lon data doesn't extend 360 degress (not cyclical)
    res@cnLevelSelectionMode = "AutomaticLevels"     ; set manual contour levels

    ;---Recalculate values for new min/max lon/lon and create new plot
    res@gsnDraw       = False 
    res@gsnFrame      = False
    res@mpMinLatF     = min(dxdlat_sub&lat)-2
    res@mpMaxLatF     = max(dxdlat_sub&lat)+2
    res@mpMinLonF     = min(dxdlat_sub&lon)-2
    res@mpMaxLonF     = max(dxdlat_sub&lon)+2
    res@mpCenterLonF  = (res@mpMinLonF + res@mpMaxLonF) / 2.
    res@pmTickMarkDisplayMode = "Always"  ; nicer map tickmarks

    ;---Plot:
    plt = gsn_csm_contour_map(wks, dxdlat_sub, res)
    draw(plt)
    frame(wks)



; BELOW SECTION IS COMMENTED OUT TO AVOID RUNNING. IT WORKS, UNCOMMENT TO RUN:
; ;======================================================================
; ; Part B: Generate Rectilinear to NARR regrid weights 
; ;         This interpolates the above grid to the NARR Grid.
; ;======================================================================

; ;---For clarity, delete the above options and start again.
;     delete(Opt)

; ;---Options for regridding rectilinear to NARR (curvilinear) grid

;     Opt                 = True
;     Opt@ForceOverwrite  = True

;     Opt@SrcTitle        = fili_name; srcFileName                  ; source grid
;     Opt@SrcRegional     = True
;     Opt@SrcFileName     = "Rectilinear.SCRIP_grid_description.nc"  ; destination files
;     Opt@SrcMask2D       = where(ismissing(x_reclin),0,1)

;     Opt@DstTitle        = "Rectilinear_to_NARR"
;     Opt@DstGridLat      = lat2d
;     Opt@DstGridLon      = lon2d
;     Opt@DstRegional     = True
;     Opt@DstGridType     = "curvilinear"
;     Opt@DstFileName     = "NARR.SCRIP_grid_description.nc"
;     Opt@DstMask2D       = where(ismissing(x_reclin),0,1)

;     Opt@InterpMethod    = InterpMethod
;     Opt@WgtFileName     = "Rect_to_NARR.WgtFile_"+InterpMethod+".nc"

;     Opt@RemoveSrcFile  = True                  ; remove SCRIP grid destination files
;     Opt@RemoveDstFile  = True 
;     Opt@NoPETLog       = True                  ; 6.2.1 onward

;     x_regrid           = ESMF_regrid(x_reclin, Opt)

; ;---Print regridded NARR variable information
;     nmsgrl = num(ismissing(x_regrid))
;     printVarSummary(x_regrid)
;     print("x_regrid: min="+min(x_regrid)+"   max="+max(x_regrid)+"   nmsgrl="+nmsgrl)

; ;************************************************
; ; create plots
; ;************************************************  
;     ; wks = gsn_open_wks("ps","ESMF")          
;     wks = gsn_open_wks("png","ESMF")          
;     plot = new(3,graphic)                      ; create a plot array
    
;     res                      = True
;     res@gsnDraw              = False           ; don't draw
;     res@gsnFrame             = False           ; don't advance frame
;    ;res@gsnSpreadColors      = True            ; spread out color table [default v6.1.0]
;     res@gsnAddCyclic         = False           ; regional data
  
;     res@cnInfoLabelOn        = False           ; turn off cn info label
;     res@cnFillOn             = True            ; turn on color
;    ;res@cnFillMode           = "RasterFill"
;     res@cnLinesOn            = False
;     res@cnLineLabelsOn       = False
;     res@lbLabelBarOn         = False           ; turn off individual cb's
  
;     res@mpMinLatF            = min(lat2d)      ; range to zoom in on
;     res@mpMaxLatF            = max(lat2d)
;     res@mpMinLonF            = min(lon2d)
;     res@mpMaxLonF            = max(lon2d)
;     res@mpCenterLonF         = -107.0          ; from file (253-360)
;     res@mpFillOn             = False
;    ;res@mpGridAndLimbOn      = True
;    ;res@mpGridLineDashPattern= 10                 ; lat/lon lines as dashed
    
;     res@gsnLeftString = ""
;     res@gsnRightString= ""
      
;     res@gsnCenterString= "Source NARR"
;     plot(0) = gsn_csm_contour_map(wks, x_reclin, res)
;     res@gsnCenterString= "NARR => 0.25 Rectilinear Grid"
;     plot(1) = gsn_csm_contour_map(wks, x_regrid, res)
;     res@gsnCenterString= "0.25 Rectilinear Grid => NARR"
;     plot(2) = gsn_csm_contour_map(wks, x_reclin, res)

; ;************************************************
; ; create panel
; ;************************************************
;     resP                     = True                ; modify the panel plot

;     if(SYNABBR .eq. "H500") then
;         resP@txString            = x_reclin@long_name+": "+lvl+"gpm: "+InterpMethod
;     elseif(SYNABBR .eq. "PMSL") then
;         resP@txString            = x_reclin@long_name+": Pa: "+InterpMethod
;     elseif(SYNABBR .eq. "CAPE") then
;         resP@txString            = x_reclin@long_name+": J/kg: "+InterpMethod
;     end if

;     resP@gsnPanelLabelBar    = True                ; add common colorbar
;     resP@gsnMaximize         = True  
;     gsn_panel(wks,plot,(/3,1/),resP)               ; now draw as one plot

; ;************************************************
; ; Calculate difference
; ;************************************************

;     diff = x_reclin
;     diff = x_regrid-x_reclin
;     diff@long_name = "Difference (Regrid-Source): "+lvl+"hPa" 

;     res@gsnDraw              = True
;     res@gsnFrame             = True
;     res@lbLabelBarOn         = True 
;     delete(res@gsnCenterString)

;     res@cnLevelSelectionMode = "ManualLevels"     ; set manual contour levels
;     res@cnMinLevelValF       = -0.1               ; set min contour level
;     res@cnMaxLevelValF       =  0.1               ; set max contour level
;     res@cnLevelSpacingF      =  0.025             ; set contour spacing

;     res@tiMainString         = diff@long_name+": "+InterpMethod          
;     plt  = gsn_csm_contour_map(wks, diff, res)
; ;****************************************************************************************

end