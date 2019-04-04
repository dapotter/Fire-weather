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
;---Read in netCDF file.
  ;a  = addfile("ts_Amon_CESM1-CAM5_historical_r1i1p1_185001-200512.nc","r")
  a  = addfile("/home/dp/Documents/FWP/gridMET/erc_1979.nc", "r")
  ;ts = a->ts(0,:,:)
  ts = a->$"energy_release_component-g"$(0,:,:)

;---Print information about ts
  printVarSummary(ts)       ; 192 x 288

  wks = gsn_open_wks("png","ERC_OR_WA")

;---To subscript with longitude values < 0, you must "flip" the longitudes
  ;ts = lonFlip(ts)     ; convert from 0:360 to -180:180
  ;ts = lonPivot(ts, -90)
;---Print information about ts
  printVarSummary(ts)       ; lon: [-180..178.75]

  lat_min =   39
  lat_max =   52
  lon_min =   -114      ; Now we can subscript using 
  lon_max =   -127      ; negative longitudes
  ts_sub  := ts({lat_min:lat_max},{lon_min:lon_max})
  printVarSummary(ts_sub)        ; 43 x 57
  
;---Setting some resources
  res                       = True

  res@gsnMaximize           = True     ; maximize plot in frame

  res@cnFillOn              = True     ; turn on contour fill
  res@cnLinesOn             = False    ; turn off contour lines
  res@cnLineLabelsOn        = False    ; turn off contour line labels

  res@tiMainString          = "Plotting lat/lon subset of data"
  res@pmTitleZone           = 4          ; Moves title down
  ;---Subsetting values, the lon doesn't extend 360 degress (not cyclical)
  res@gsnAddCyclic          = False
  
;---Recalculate values for new min/max lon/lon and create new plot
  res@gsnDraw               = False 
  res@gsnFrame              = False
  res@mpMinLatF         = min(ts_sub&lat)-2
  res@mpMaxLatF         = max(ts_sub&lat)+2
  res@mpMinLonF         = min(ts_sub&lon)-2
  res@mpMaxLonF         = max(ts_sub&lon)+2
  res@mpCenterLonF      = (res@mpMinLonF + res@mpMaxLonF) / 2.
  res@pmTickMarkDisplayMode = "Always"  ; nicer map tickmarks

  
;---Attach lat/lon grid lines of subsetted data.
  gsres                  = True;True
  gsres@gsnCoordsAsLines = False;True
  gsres@gsnCoordsAttach  = False;True

;---Attach two markers showing two lat,lon corners of interest
  mkres               = True
  mkres@gsMarkerIndex = 16     ; filled dot
  mkres@gsMarkerColor = "black"
  mkres@gsMarkerSizeF = 15

  plot = gsn_csm_contour_map(wks,ts_sub,res)
  ;gsn_coordinates(wks,plot,ts_sub,gsres)
  mkid2 = gsn_add_polymarker(wks,plot,(/lon_min,lon_max/),(/lat_min,lat_max/),mkres)
  draw(plot)
  frame(wks)

end

