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

begin
;---Read in netCDF file:
  a  = addfile("/home/dp/Documents/FWP/NARR/3D_grib1_exp/merged_AWIP32.1979010100.3D.nc", "r")
  f  = getfilevarnames(a)
  print(f)

  H500 = a->H500
  H500GradX = a->H500GradX
  H500GradY = a->H500GradY

  printVarSummary(H500)
  printVarSummary(H500GradX)
  printVarSummary(H500GradY)



;---Save resulting plot as H500_OR_WA.png:
  wks = gsn_open_wks("png","H500_OR_WA")

;---Specify lon lat subset:
  lat_min =   39
  lat_max =   52
  lon_min =   233;-114
  lon_max =   246;-127
  

;---H500:
  H500_sub  := H500({lat_min:lat_max},{lon_min:lon_max})
  printVarSummary(H500_sub)

;---Setting some resources
  res                 = True
  res@gsnMaximize     = True     ; maximize plot in frame
  res@cnFillOn        = True     ; turn on contour fill
  res@cnLinesOn       = False    ; turn off contour lines
  res@cnLineLabelsOn  = False    ; turn off contour line labels
  res@tiMainString    = "Plotting lat/lon subset of data"
  res@pmTitleZone     = 4        ; Moves title down
  res@gsnAddCyclic    = False    ; The subset lon data doesn't extend 360 degress (not cyclical)
  
;---Recalculate values for new min/max lon/lon and create new plot
  res@gsnDraw       = False 
  res@gsnFrame      = False
  res@mpMinLatF     = min(H500_sub&lat)-2
  res@mpMaxLatF     = max(H500_sub&lat)+2
  res@mpMinLonF     = min(H500_sub&lon)-2
  res@mpMaxLonF     = max(H500_sub&lon)+2
  res@mpCenterLonF  = (res@mpMinLonF + res@mpMaxLonF) / 2.
  res@pmTickMarkDisplayMode = "Always"  ; nicer map tickmarks

;---Plot:
  plot = gsn_csm_contour_map(wks,H500_sub,res)
  draw(plot)
  frame(wks)



;---H500 Grad X:
  wks = gsn_open_wks("png","H500_Grad_X_OR_WA")

  H500GradX_sub  := H500GradX({lat_min:lat_max},{lon_min:lon_max})
  printVarSummary(H500GradX_sub)
;---Setting some resources
  res                 = True
  res@gsnMaximize     = True     ; maximize plot in frame
  res@cnFillOn        = True     ; turn on contour fill
  res@cnLinesOn       = False    ; turn off contour lines
  res@cnLineLabelsOn  = False    ; turn off contour line labels
  res@tiMainString    = "Plotting lat/lon subset of zonal gradient"
  res@pmTitleZone     = 4        ; Moves title down
  res@gsnAddCyclic    = False    ; The subset lon data doesn't extend 360 degress (not cyclical)
  
;---Recalculate values for new min/max lon/lon and create new plot
  res@gsnDraw       = False 
  res@gsnFrame      = False
  res@mpMinLatF     = min(H500_sub&lat)-2
  res@mpMaxLatF     = max(H500_sub&lat)+2
  res@mpMinLonF     = min(H500_sub&lon)-2
  res@mpMaxLonF     = max(H500_sub&lon)+2
  res@mpCenterLonF  = (res@mpMinLonF + res@mpMaxLonF) / 2.
  res@pmTickMarkDisplayMode = "Always"  ; nicer map tickmarks

;---Plot:
  plot = gsn_csm_contour_map(wks,H500GradX_sub,res)
  draw(plot)
  frame(wks)



;---H500 Grad Y:
  wks = gsn_open_wks("png","H500_Grad_Y_OR_WA")

  H500GradY_sub  := H500GradY({lat_min:lat_max},{lon_min:lon_max})
  printVarSummary(H500GradY_sub)
;---Setting some resources
  res                 = True
  res@gsnMaximize     = True     ; maximize plot in frame
  res@cnFillOn        = True     ; turn on contour fill
  res@cnLinesOn       = False    ; turn off contour lines
  res@cnLineLabelsOn  = False    ; turn off contour line labels
  res@tiMainString    = "Plotting lat/lon subset of meridional gradient"
  res@pmTitleZone     = 4        ; Moves title down
  res@gsnAddCyclic    = False    ; The subset lon data doesn't extend 360 degress (not cyclical)
  
;---Recalculate values for new min/max lon/lon and create new plot
  res@gsnDraw       = False 
  res@gsnFrame      = False
  res@mpMinLatF     = min(H500_sub&lat)-2
  res@mpMaxLatF     = max(H500_sub&lat)+2
  res@mpMinLonF     = min(H500_sub&lon)-2
  res@mpMaxLonF     = max(H500_sub&lon)+2
  res@mpCenterLonF  = (res@mpMinLonF + res@mpMaxLonF) / 2.
  res@pmTickMarkDisplayMode = "Always"  ; nicer map tickmarks

;---Plot:
  plot = gsn_csm_contour_map(wks,H500GradY_sub,res)
  draw(plot)
  frame(wks)


end