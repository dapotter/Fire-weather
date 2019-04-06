;---------------------------------------------------------------------
; Run this to print a .grib1 file's variables and attributes before
;---------------------------------------------------------------------

begin
    ;---Files:
    ;f = addfile("merged_AWIP32.1979010321.3D.grib1", "r")
    ;f = addfile("/home/dp/Documents/FWP/NARR/3D_grib1_exp/merged_AWIP32.1979010121.3D.grib1","r")
    ;f = addfile("/home/dp/Documents/FWP/NARR/pbl_grib1_exp/merged_AWIP32.1979010121.RS.pbl.grib1","r")
    ;f = addfile("/home/dp/Documents/FWP/NARR/flx_grib1_exp/merged_AWIP32.1979010121.RS.flx.grib1","r")
    ;f = addfile("/home/dp/Documents/FWP/NARR/sfc_grib1_exp/merged_AWIP32.1979010100.RS.sfc.grib1", "r")
     f = addfile("/home/dp/Documents/FWP/gridMET/erc_1979.grb", "r")
    
    ;---Variables:
    ;v = f->HGT_221_ISBL    ; 3D file - Geopotential height
    ;v = f->CAPE_221_SPDY   ; pbl file - Convective available potential energy
    ;v = f->PRMSL_221_MSL   ; flx file - Pressure reduced to mean sea level
    ;l = f->lv_ISBL0        ; 3D file - level variable
    ;l = f->lv_SPDY0        ; pbl file - level variable
    ;x = f->gridlon_221     ; all files
    ;y = f->gridlat_221     ; all files


    ;---Printing file and variable information:
    print("File atts, and var dims, atts: ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    file_atts = getfileatts(f)
    print("file atts:"+file_atts)
    ;file_var_dims = getfilevardims(f, "HGT_221_ISBL")
    ;file_var_atts = getfilevaratts(f, "HGT_221_ISBL")
    ;file_var_dims = getfilevardims(f, "CAPE_221_SPDY")
    ;file_var_atts = getfilevaratts(f, "CAPE_221_SPDY")
    ;print("file_var_dims:"+file_var_dims)
    ;print("file_var_atts:"+file_var_atts)
    
    
    ;---Variable summaries
    print("Variable summaries: ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    ;printVarSummary(v)
    ;printVarSummary(l)
    ;printVarSummary(x)
    ;printVarSummary(y)


    ;---Prints file each variable's name, type, dimensions, 
    ;---dimension sizes, and attributes
    print("Var names, dims, atts: ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~")
    vnames = getfilevarnames(f)           ; all variable names
    nvars  = dimsizes(vnames)
    attnames = getfileatts(f)

    ;---Prints var names and summaries
    do nv = 0,nvars-1
        print("~~~~~~~~~~~~~~~~~~~~~~~~~ Var Summary ~~~~~~~~~~~~~~~~~~~~~~~~~")
        ;---Var name and summary:
        print("Variable '" + vnames(nv) + "'")
        if(.not.any(ismissing(vnames))) then
            printFileVarSummary (f,vnames(nv))
        end if
    end do

    do nv = 0,nvars-1
        vtype  = getfilevartypes(f,vnames(nv))
        vatts  := getfilevaratts(f,vnames(nv))       ; The := is needed because these
        dnames := getfilevardimnames(f,vnames(nv))   ; variables may be a different size
        dsizes := getfilevardimsizes(f,vnames(nv))   ; the next time through the loop

        print("============================ Var: ============================")
        
        ;---Var name and summary:
        print("Variable '" + vnames(nv) + "'")
        
        ;---Var type and dimension names:
        print("  type            : " + vtype)
        if(.not.any(ismissing(dnames))) then
            print("  dimension names : " + str_join(dnames,","))
        else
            print("  dimension names : ")
        end if
        
        ;---Var dimension sizes and attributes:
        print("  dimension sizes : " + str_join(""+dsizes,","))
        if(.not.any(ismissing(vatts))) then
            print("  attributes      : " + str_join(vatts,","))
        else
            print("  attributes      : ")
        end if
        
    end do
end
