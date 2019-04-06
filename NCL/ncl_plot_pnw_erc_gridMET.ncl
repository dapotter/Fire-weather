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
  a  = addfile("/home/dp/Documents/FWP/gridMET/erc_1979.nc", "r")
  erc = a->$"energy_release_component-g"$(0,:,:)
  printVarSummary(erc)

;---Save resulting plot as ERC_OR_WA.png:
  wks = gsn_open_wks("png","ERC_OR_WA")

;---Specify lon lat subset:
  lat_min =   39
  lat_max =   52
  lon_min =   -114
  lon_max =   -127
  
  erc_sub  := erc({lat_min:lat_max},{lon_min:lon_max})
  printVarSummary(erc_sub)
  
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
  res@mpMinLatF     = min(erc_sub&lat)-2
  res@mpMaxLatF     = max(erc_sub&lat)+2
  res@mpMinLonF     = min(erc_sub&lon)-2
  res@mpMaxLonF     = max(erc_sub&lon)+2
  res@mpCenterLonF  = (res@mpMinLonF + res@mpMaxLonF) / 2.
  res@pmTickMarkDisplayMode = "Always"  ; nicer map tickmarks

;---Plot:
  plot = gsn_csm_contour_map(wks,erc_sub,res)
  draw(plot)
  frame(wks)
end

