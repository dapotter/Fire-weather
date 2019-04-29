;======================================================================
; narr_5.ncl
;
; Concepts illustrated:
;   - Using a weight file generated via ESMF to:
;     (a)  Regrid the curvilinear NARR grig to a rectilinear grid "ESMF_regrid_with_weights"
;          The ESMF weight file is generated via ESMF_30.ncl
;     (b)  Generate a pressure x latitude cross-section
;     (c)  Generate a pressure x longitude cross-section
;     (d)  Generate a pressure x user_specified_points cross-section
;          Here, the user specified points are a great circle path between locations 
;======================================================================
;load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/gsn_code.ncl"  ; not needed for 6.1.0 onward
;load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/gsn_csm.ncl"

load "$NCARG_ROOT/lib/ncarg/nclscripts/esmf/ESMF_regridding.ncl"

;======================================================================
; User specifications
;======================================================================

    InterpMethod = "bilinear"                           ; "bilinear", "conserve"
    ;varName      = (/ "TMP_221_ISBL" /)                 ; could be multiple variables
    varName      = (/ "HGT_221_ISBL" /)                 ; could be multiple variables

    ;wgtDirName   = "./"                                 ; directory where weight file is located
    ;wgtFileName  = "NARR_to_Rect.WgtFile_"+InterpMethod+".nc"
    
    wgtDirName   = "/home/dp/Documents/FWP/NCL/"  ; directory where weight file is located
    wgtFileName  = "NARR_to_Rect.WgtFile_"+InterpMethod+".nc"

    ;srcDirName   = "./"
    ;srcFileName  = "merged_AWIP32.1979010100.3D.NARR.grb"
    
    srcDirName   = "/home/dp/Documents/FWP/NARR/3D_grib1_exp/"
    srcFileName  = "merged_AWIP32.1979010609.3D.grib1"
    
    latXsec      = 45.0               ; (b)
    latLonStrt   = 233.0 ;260.0              
    latLonLast   = 246.0 ;305.0

    lonXsec      = 240.0; 290.0              ; (c)
    lonLatStrt   =  39.0; 25.0
    lonLatLast   =  52.0; 75.0

    NPTS         = 100                ; (d) user specified number of points
    ;ptsXsec      = gc_latlon( 25.5, 250., 70., 285.75, NPTS, 2)
    ptsXsec      = gc_latlon( 39.5, 220., 45.0, 240.0, NPTS, 2)

    nLabels      = 8 

    wksType      = "png"              ; ps, png, pdf, x11, ....
    wksDirName   = "./"   
    wksFileName  = "narr" 

;======================================================================
; Miscellaneous; One time settings
;======================================================================
    nvar         = dimsizes(varName)
    wgtFilePath  = wgtDirName + wgtFileName
    srcFilePath  = srcDirName + srcFileName
    wksFilePath  = wksDirName + wksFileName

    sfile        = addfile(srcFilePath,"r")    

    latXsecUser  = ptsXsec@gclat      ; convenience
    lonXsecUser  = ptsXsec@gclon
    N1           = NPTS-1

    XBValues    = toint( fspan(0,N1,nLabels) )
    XBLabels    = new(nLabels,"string")
    do i=0,nLabels-1
       x = lonXsecUser(XBValues(i))
       y = latXsecUser(XBValues(i))
       XBLabels(i) = sprintf("%5.1f", y)+"~C~"+sprintf("%5.1f", x)
    end do

;************************************************
; graphic resources
;************************************************  
    
    res                      = True
    res@gsnMaximize          = True  
  
    res@cnFillOn             = True            ; turn on color fill
    res@cnLinesOn            = True            ; turn lines on/off         ; True is default
    res@cnLineLabelsOn       = True            ; turn line labels on/off   ; True is default

    wks = gsn_open_wks(wksType,wksFilePath)          
  
;************************************************
; Loop over variables; generate 3 X-Sections
;************************************************
   x        := sfile->$varName(nv)$
   printVarSummary(x)
   x_regrid := ESMF_regrid_with_weights(x, wgtFilePath, False)
   printVarSummary(x_regrid)
   
    do nv=0,nvar-1
       x        := sfile->$varName(nv)$
       printVarSummary(x)
       x_regrid := ESMF_regrid_with_weights(x, wgtFilePath, False)
       printVarSummary(x_regrid)

       res@gsnMajorLonSpacing = 10
       res@tiMainString = "Pressure x Longitude at "+latXsec+"N"
       plot  = gsn_csm_pres_hgt(wks,x_regrid(:,{latXsec},{latLonStrt:latLonLast}),res) ; (b)
       delete(res@gsnMajorLonSpacing)

       res@gsnMajorLatSpacing = 10
       res@tiMainString = "Pressure x Latitude at "+lonXsec+"E"
       plot  = gsn_csm_pres_hgt(wks,x_regrid(:,{lonLatStrt:lonLatLast},{lonXsec}),res)                       ; (c)
       delete(res@gsnMajorLatSpacing)
       
       res@tmXBMode   = "Explicit"
       res@tmXBValues = XBValues
       res@tmXBLabels = XBLabels
       res@tiMainString = "Pressure x User_Specified_Locations"
       xsecUser = linint2_points_Wrap(x_regrid&lon ,x_regrid&lat,x_regrid, False \           
                                     ,lonXsecUser, latXsecUser, 2)
       printVarSummary(xsecUser)
       plot = gsn_csm_pres_hgt(wks,xsecUser,res)                                      ; (d)
       delete( [/ res@tmXBMode, res@tmXBValues, res@tmXBLabels /] )
    end do
