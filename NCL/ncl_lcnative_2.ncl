;--------------------------------------------
; lcnative_2.ncl
;--------------------------------------------
;
; These files are loaded by default in NCL V6.2.0 and newer
; load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/gsn_code.ncl"   
; load "$NCARG_ROOT/lib/ncarg/nclscripts/csm/gsn_csm.ncl"   

begin

;--------------------------------------------
; open file and read in data
;--------------------------------------------

;  fili  = "merged_AWIP32.1979110621_H500.3D.grib1"
;  fili  = "merged_AWIP32.1979110621.3D.grib1" ; Can use the full .3D file but need to specify
					       ; vpt(14,:,:) at the bottom of the file

;  fili  = "/home/dp/Documents/FWP/gridMET/erc_1979.nc"
  fili  = "/home/dp/Documents/FWP/gridMET/erc_1979.grb"
  f     = addfile (fili, "r")

;  vpt   = f->VPTMP_252_HYBL
;  vpt   = f->tpag10	; Doesn't work because the _H500.3D grib file created 
			; by wgrib doesn't use tpag10 in its variable info, only 
			; netCDF files contain this.
;  vpt   = f->HGT_221_ISBL
;  lat2d     = f->gridlat_221
;  lon2d     = f->gridlon_221

  vpt       = f->PRES_GDS0_SFC
  lat2d     = f->g0_lat_1
  lon2d     = f->g0_lon_2

;--------------------------------------------
; create plot
;--------------------------------------------
  wks = gsn_open_wks("png","lcnative_2")      ; send graphics to PNG file

  res                   = True              ; plot mods desired
  res@tiMainString      = "Native Lambert Conformal"

;--------------------------------------------
; lambert conformal projections are limited using
; the corners method rather than the latlon method
; seen for cylindrical equidistant projections
;--------------------------------------------
  res@mpLimitMode       = "Corners"         ; choose range of map
  res@mpLeftCornerLatF  = lat2d@corners(0)
  res@mpLeftCornerLonF  = lon2d@corners(0)
  res@mpRightCornerLatF = lat2d@corners(2)
  res@mpRightCornerLonF = lon2d@corners(2)

;--------------------------------------------
; native lambert grids read in from grib files are
; different than those read in from netCDF files.
; NCL automatically looks for the parallel and 
; meridian information on the grib file and attaches
; this information as attributes to the lat2d array.
;--------------------------------------------
  res@mpProjection          = "LambertConformal"
  res@mpLambertParallel1F   = lat2d@mpLambertParallel1F
  res@mpLambertParallel2F   = lat2d@mpLambertParallel2F
  res@mpLambertMeridianF    = lat2d@Lov       ; ncl adds from grib file

  res@pmTickMarkDisplayMode = "Always"
  res@mpFillOn              = False           ; turn off map fill
  res@mpOutlineDrawOrder    = "PostDraw"      ; draw continental outline last
  res@mpOutlineBoundarySets = "GeophysicalAndUSStates" ; state boundaries

;--------------------------------------------
; usually, when data is placed onto a map, it is TRANSFORMED to the specified
; projection. Since this model is already on a native lambert conformal grid,
; we want to turn OFF the transformation.
;--------------------------------------------
  res@tfDoNDCOverlay = True
; res@tfDoNDCOverlay = "NDCViewport"          ; NCL V6.5.0 or later
;--------------------------------------------
  res@cnFillOn              = True            ; color plot desired
  res@cnLinesOn             = False           ; turn off contour lines
  res@cnFillPalette         = "gui_default"   ; set color map
  res@gsnAddCyclic          = False           ; regional data 

;  plot = gsn_csm_contour_map(wks,vpt(9,:,:),res)
  plot = gsn_csm_contour_map(wks,vpt(:,:),res)
end
