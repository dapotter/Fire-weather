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


    InterpMethod = "bilinear"                           ; "bilinear", "conserve"

    ;hlvl         = 700                                  ; arbitrary pressure level
    hlvl         = 500                                  ; arbitrary pressure level
                                
;---Input file
    ;srcDirName   = "./"
    ;srcFileName  = "merged_AWIP32.1979010100.3D.NARR.grb"
    ;srcFilePath  =  srcDirName + srcFileName
    
    srcDirName   = "/home/dp/Documents/FWP/NARR/3D_grib1_exp/"
    ;srcFileName  = "merged_AWIP32.1979010609.3D.grib1"
    srcFileName  = "merged_AWIP32.1979010100.3D"
    srcFileSuffix = ".grib1"

    srcFilePath  =  srcDirName + srcFileName + srcFileSuffix

    netCDF = True 
    if (netCDF) then
        ;diro  = "./"                                      ; output directory
        fili  = srcFileName + srcFileSuffix
        diro  = srcDirName
        filo  = srcFileName + ".nc" ;merged_AWIP32.1979010100.3D.NARR.nc" ; output netCDF file
    end if
    

;---Get the variable to be regridded;  only need one level to generate the weight
;---Also, the grid coordinates

    sfile =  addfile(srcFilePath,"r")    
    ;x     = sfile->TMP_221_ISBL({hlvl},:,:)    ;  ( gridx_221, gridy_221) ; (   277,349)
    x     = sfile->HGT_221_ISBL({hlvl},:,:)    ;  ( gridx_221, gridy_221) ; (   277,349)
    lat2d = sfile->gridlat_221                 ; (gridx_221, gridy_221)
    lon2d = sfile->gridlon_221
    lev   = sfile->lv_ISBL0

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
    
;---Create the destination rectilinear lat[*]/lon[*] arrays. 
;---Here, roughly the same resolution as the source grid at 45N.
;---This is arbitrary. It can be whatever the user desires.
;---For example, it could be just the area of the USA.
;---Create rectilinear coordinates; monotonically increasing 

    ; nlat  = 337
    ; nlon  = 831
    ; lat   = fspan(   1.0, 85.0 ,nlat)
    ; lon   = fspan( 150.0,358.5 ,nlon)

    nlat  = 52
    nlon  = 52
    lat   = fspan(  39.0, 52.0 ,nlat)
    lon   = fspan( 233.0,246.0 ,nlon)

;---For netCDF
    lat!0         = "lat"
    lat@units     = "degrees_north"
    lat&lat       = lat

    lon!0         = "lon"
    lon@units     = "degrees_east"
    lon&lon       = lon

    lev!0         = "lev"

;---Create regrid options

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
    Opt@RemoveSrcFile  = True                  ; remove SCRIP grid destination files
    Opt@RemoveDstFile  = True 
    Opt@NoPETLog       = True                  ; 6.2.1 onward

;---Perform the regrid: NARR ==> rectilinear (_reclin)

    x_reclin = ESMF_regrid(x, Opt)             ; Do the regridding for x  

    mmsg  = num(ismissing(x_reclin))           ; # of msg values
    print("x_reclin before adding file dimensions and attributes: ****************************")
    printVarSummary(x_reclin)
    print("x_reclin: min="+min(x_reclin)+"   max="+max(x_reclin)+"    mmsg="+mmsg)
    
;---For netCDF
    ; x_reclin!0         = "lev"
    ; x_reclin&lev       =  lev
    ; x_reclin!1         = "lat"
    ; x_reclin&lat       =  lat
    ; x_reclin!2         = "lon"
    ; x_reclin&lon       =  lon
    ; x_reclin@units     = x@units
    ; x_reclin@long_name = x@long_name
    ; x_reclin@time      = x@initial_time
    print("x_reclin after adding file dimensions and attributes: ****************************")
    printVarSummary(x_reclin)


;======================================================================
; Part B: Generate Rectilinear to NARR regrid weights 
;         This interpolates the above grid to the NARR Grid.
;======================================================================

;---For clarity, delete the above options and start again.
    delete(Opt)

;---Options for regridding rectilinear to NARR (curvilinear) grid

    Opt                 = True
    Opt@ForceOverwrite  = True

    Opt@SrcTitle        = srcFileName                  ; source grid
    Opt@SrcRegional     = True
    Opt@SrcFileName     = "Rectilinear.SCRIP_grid_description.nc"  ; destination files
    Opt@SrcMask2D       = where(ismissing(x_reclin),0,1)

    Opt@DstTitle        = "Rectilinear_to_NARR"
    Opt@DstGridLat      = lat2d
    Opt@DstGridLon      = lon2d
    Opt@DstRegional     = True
    Opt@DstGridType     = "curvilinear"
    Opt@DstFileName     = "NARR.SCRIP_grid_description.nc"         
    Opt@DstMask2D       = where(ismissing(x),0,1)

    Opt@InterpMethod    = InterpMethod
    Opt@WgtFileName     = "Rect_to_NARR.WgtFile_"+InterpMethod+".nc"

    Opt@RemoveSrcFile  = True                  ; remove SCRIP grid destination files
    Opt@RemoveDstFile  = True 
    Opt@NoPETLog       = True                  ; 6.2.1 onward

    x_regrid           = ESMF_regrid(x_reclin,Opt)

;---Print regridded NARR variable information
    nmsgrl = num(ismissing(x_regrid))
    printVarSummary(x_regrid)
    print("x_regrid: min="+min(x_regrid)+"   max="+max(x_regrid)+"   nmsgrl="+nmsgrl)

;************************************************
; create plots
;************************************************  
    ; wks = gsn_open_wks("ps","ESMF")          
    wks = gsn_open_wks("png","ESMF")          
    plot = new(3,graphic)                      ; create a plot array
    
    res                      = True
    res@gsnDraw              = False           ; don't draw
    res@gsnFrame             = False           ; don't advance frame
   ;res@gsnSpreadColors      = True            ; spread out color table [default v6.1.0]
    res@gsnAddCyclic         = False           ; regional data
  
    res@cnInfoLabelOn        = False           ; turn off cn info label
    res@cnFillOn             = True            ; turn on color
   ;res@cnFillMode           = "RasterFill"
    res@cnLinesOn            = False
    res@cnLineLabelsOn       = False
    res@lbLabelBarOn         = False           ; turn off individual cb's
  
    res@mpMinLatF            = min(lat2d)      ; range to zoom in on
    res@mpMaxLatF            = max(lat2d)
    res@mpMinLonF            = min(lon2d)
    res@mpMaxLonF            = max(lon2d)
    res@mpCenterLonF         = -107.0          ; from file (253-360)
    res@mpFillOn             = False
   ;res@mpGridAndLimbOn      = True
   ;res@mpGridLineDashPattern= 10                 ; lat/lon lines as dashed
    
    res@gsnLeftString = ""
    res@gsnRightString= ""
      
    res@gsnCenterString= "Source NARR"
    plot(0) = gsn_csm_contour_map(wks,x        ,res) 
    res@gsnCenterString= "NARR => 0.25 Rectilinear Grid"
    plot(1) = gsn_csm_contour_map(wks,x_regrid ,res) 
    res@gsnCenterString= "0.25 Rectilinear Grid => NARR"
    plot(2) = gsn_csm_contour_map(wks,x_reclin ,res) 

;************************************************
; create panel
;************************************************
    resP                     = True                ; modify the panel plot
    resP@txString            = x@long_name+": "+hlvl+"hPa: "+InterpMethod
    resP@gsnPanelLabelBar    = True                ; add common colorbar
    resP@gsnMaximize         = True  
    gsn_panel(wks,plot,(/3,1/),resP)               ; now draw as one plot

;************************************************
; Calculate difference
;************************************************

    diff = x
    diff = x_regrid-x 
    diff@long_name = "Difference (Regrid-Source): "+hlvl+"hPa" 

    res@gsnDraw              = True
    res@gsnFrame             = True
    res@lbLabelBarOn         = True 
    delete(res@gsnCenterString)

    res@cnLevelSelectionMode = "ManualLevels"     ; set manual contour levels
    res@cnMinLevelValF       = -0.1               ; set min contour level
    res@cnMaxLevelValF       =  0.1               ; set max contour level
    res@cnLevelSpacingF      =  0.025             ; set contour spacing

    res@tiMainString         = diff@long_name+": "+InterpMethod          
    plt  = gsn_csm_contour_map(wks,diff     ,res)
;****************************************************************************************





;------------------------------------- EXPORT netCDF -------------------------------------

;---------------------------------------------------
; Create netCDF: http://www.ncl.ucar.edu/Applications/method_2.shtml
;---------------------------------------------------
  if (netCDF) then

      system ("'rm' -f "+diro+filo)
      ncdf  = addfile(diro+filo,"c")

;---------------------------------------------------
; create global attributes of the file
;---------------------------------------------------
     fAtt               = True            ; assign file attributes
     fAtt@title         = "NARR: ESMF weighted regrid to 0.25 deg rectilinear"
     fAtt@source        = "The North American Regional Reanalysis (NARR) Project"
     fAtt@analysis_time = x@initial_time
     fAtt@input_file    = fili
     fAtt@Conventions   = "None"
     fAtt@creation_date = systemfunc ("date")

     fileattdef( ncdf, fAtt )            ; copy file attributes

;---------------------------------------------------
; dimension names/sizes/attributes
;---------------------------------------------------
     dimNames = (/ "lat", "lon" /)
     ; dimSizes = (/  klev , NLAT,  MLON /)
     dimSizes = (/  nlat,  nlon /)
     dimUnlim = (/ False,False/)
     filedimdef(ncdf,dimNames,dimSizes,dimUnlim)

     ; filevardef(ncdf, "lev"   ,typeof(lev)  , getvardims(lev))
     filevardef(ncdf, "lat"   ,typeof(lat)  , getvardims(lat))
     filevardef(ncdf, "lon"   ,typeof(lon)  , getvardims(lon))
     filevardef(ncdf, "H500"  ,typeof(x_reclin)  , getvardims(x_reclin))

;---------------------------------------------------
; Copy attributes associated with each variable to the file
; All attributes associated with each variable will be copied.
;---------------------------------------------------
     ; filevarattdef(ncdf,"lev"  ,lev  )                   ; copy lev attributes
     filevarattdef(ncdf,"lat"  ,lat  )                   ; copy lat attributes
     filevarattdef(ncdf,"lon"  ,lon  )                   ; copy lon attributes
     filevarattdef(ncdf,"H500" ,x_reclin )

;---------------------------------------------------
; Write values to the file
;---------------------------------------------------
     ; ncdf->lev    = (/ lev   /)
     ncdf->lat    = (/ lat   /)
     ncdf->lon    = (/ lon   /)
     ncdf->H500      = (/ x_reclin   /)
  end if

end

; ;************************************************
; ; create plots
; ;************************************************  
;     wks = gsn_open_wks("png","ncl_regrid_make_weights_pnw_synvar_NARR")      ; send graphics to PNG file
;     plot = new(3,graphic)                      ; create a plot array
    
;     res                      = True
;     res@gsnDraw              = False           ; don't draw
;     res@gsnFrame             = False           ; don't advance frame
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
;     plot(0) = gsn_csm_contour_map(wks,x        ,res) 
;     res@gsnCenterString= "NARR => 0.25 Rectilinear Grid"
;     plot(1) = gsn_csm_contour_map(wks,x_regrid ,res) 
;     res@gsnCenterString= "0.25 Rectilinear Grid => NARR"
;     plot(2) = gsn_csm_contour_map(wks,x_reclin ,res) 

; ;************************************************
; ; create panel
; ;************************************************
;     resP                     = True                ; modify the panel plot
;     resP@gsnPanelMainString  = x@long_name+": "+hlvl+"hPa: "+InterpMethod
;     resP@gsnPanelLabelBar    = True                ; add common colorbar
;     resP@gsnMaximize         = True  
;     gsn_panel(wks,plot,(/3,1/),resP)               ; now draw as one plot

; ;************************************************
; ; Calculate difference
; ;************************************************

;     diff = x
;     diff = x_regrid-x 
;     diff@long_name = "Difference (Regrid-Source): "+hlvl+"hPa" 

;     res@gsnDraw              = True
;     res@gsnFrame             = True
;     res@lbLabelBarOn         = True 
;     delete(res@gsnCenterString)

;     res@cnLevelSelectionMode = "ManualLevels"     ; set manual contour levels
;     res@cnMinLevelValF       = -0.1               ; set min contour level
;     res@cnMaxLevelValF       =  0.1               ; set max contour level
;     res@cnLevelSpacingF      =  0.025             ; set contour spacing

;     res@tiMainString         = diff@long_name+": "+InterpMethod          
;     plt  = gsn_csm_contour_map(wks,diff     ,res)