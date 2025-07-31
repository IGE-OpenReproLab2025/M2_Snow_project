#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# GK20200511

import glob
import netCDF4
import numpy as np
import matplotlib.pyplot as plt
# from scipy.signal import savgol_filter

scenario = 'ssp585'
racineCMIP6 = "/data/gkrinner/CMIP6"
racineMasques = "/home/gkrinner/CMIP6/masques"

Yref1 = 1850
Yref2 = 1900

nmon = 12

Tbinmin = -.5
Tbinmax = 5.
dtbin = .25
nbin = int((Tbinmax - Tbinmin)/dtbin + 1)
Tbincen = np.linspace(Tbinmin,Tbinmax,nbin)

taboo = []

epsilon = 0.001

def get_models(filelist):
    models = []
    for ific,fichier in enumerate(filelist):
        model = fichier.rsplit("/",1)[1].split("_")[2]
        if (model not in models): 
            models.append(model)
    models.sort()
    return(models)
    
filelist = glob.glob(racineCMIP6+"/CMIP/historical/Amon/tas/*.nc")
models_Thist = get_models(filelist)

filelist = glob.glob(racineCMIP6+"/CMIP/historical/SImon/siconc/*.nc")
models_Shist = get_models(filelist)

filelist = glob.glob(racineCMIP6+"/ScenarioMIP/"+scenario+"/Amon/tas/*.nc")
models_Tscen = get_models(filelist)

filelist = glob.glob(racineCMIP6+"/ScenarioMIP/"+scenario+"/SImon/siconc/*.nc")
models_Sscen = get_models(filelist)

models = []
for imod, model in enumerate(models_Thist):
    if (model not in models) and (model not in taboo) and (model in models_Shist) and (model in models_Tscen) and (model in models_Sscen):
        models.append(model)

print(models)

for imod, model in enumerate(models):

  # if ( model == "E3SM-1-1"):

    print("Model : {} {}".format(imod,model))

    # areacella
    #
    champ = 'areacella'
    fichier = glob.glob(racineMasques+'/'+champ+'/'+champ+'_'+model+'.nc')
    if (fichier == []):
        fichier = glob.glob(racineMasques+'/pseudo_'+champ+'/'+champ+'_'+model+'.nc')
    if (fichier == []):
        print("Il manque {} pour {} !".format(champ,model))
        quit()
    print(fichier) #GKtest
    f = netCDF4.Dataset(fichier[0])
    lon = f.variables['lon'][:]
    nlon = lon.size
    lat = f.variables['lat'][:]
    nlat = lat.size
    areacella = f.variables[champ][:,:]
    aireterre = np.sum(areacella)
    # print("Aire Terre : {}".format(aireterre))
    f.close()

    # lire Thist + Tscen, extraire les temps, calculer les temperatures moyennes globales
    #
    champ = 'tas'
    fichier = glob.glob(racineCMIP6+"/CMIP/historical/Amon/"+champ+"/*_"+model+"_*.nc")
    print(fichier) #GKtest
    fh = netCDF4.Dataset(fichier[0])
    fichier = glob.glob(racineCMIP6+"/ScenarioMIP/"+scenario+"/Amon/"+champ+"/*_"+model+"_*.nc")
    print(fichier) #GKtest
    fs = netCDF4.Dataset(fichier[0])
    nlon2 = fh.variables['lon'][:].size
    nlat2 = fh.variables['lat'][:].size
    if (nlon2 != nlon) or (nlat2 != nlat):
        print("{} : Nombre de longitudes ou latitude faux ! {} {} {} {}".format(model,nlon,nlon2,nlat,nlat2))
        quit()
    tname = fh.variables[champ].dimensions[0]
    timeh = fh.variables[tname][:]
    nth = timeh.shape[0]
    times = fs.variables[tname][:]
    nts = times.shape[0]
    #
    nt = nth + nts
    year = np.empty((nt),int)
    month = np.empty((nt),int)
    GSAT = np.empty((nt),float)
    #
    t_unit = fh.variables[tname].units
    t_cal = fh.variables[tname].calendar
    tvalue = netCDF4.num2date(timeh,units = t_unit,calendar = t_cal)
    for it in range(nth):
        year[it] = tvalue[it].year
        month[it] = tvalue[it].month
        tas = fh.variables[champ][it,:,:]
        tas *= areacella
        GSAT[it] = np.sum(tas[:,:])/aireterre
    #
    t_unit = fs.variables[tname].units
    t_cal = fs.variables[tname].calendar
    tvalue = netCDF4.num2date(times,units = t_unit,calendar = t_cal)
    for it in range(nts):
        year[nth+it] = tvalue[it].year
        month[nth+it] = tvalue[it].month
        tas = fs.variables[champ][it,:,:]
        tas *= areacella
        GSAT[nth+it] = np.sum(tas[:,:])/aireterre
    #
    fh.close()
    fs.close()
    #
    iyref1 = np.argmax(year >= Yref1)
    iyref2 = nt - np.argmax(year[::-1] <= Yref2)
    GSATref = np.average(GSAT[iyref1:iyref2])
    print("  GSAT reference : {}".format(GSATref))
    GSAT[:] -= GSATref
    #
    # GSATsmooth = savgol_filter(GSAT, 12, 3) # window size arg2, polynomial order arg3
    # GSAT smoothed: exponential smoothing, tau=60 = 5 yrs
    tau = 60
    GSATsmooth = np.empty((nt),float)
    GSATsmooth[0] = np.average(GSAT[0:tau])
    for it in range(1,nt):
        GSATsmooth[it] = 1./tau * GSAT[it] + (1. - 1./tau) * GSATsmooth[it-1]

    # lire Shist + Sscen
    #
    fichier = glob.glob(racineCMIP6+"/CMIP/historical/SImon/siconc/*_"+model+"_*.nc")
    print(fichier) #GKtest
    fh = netCDF4.Dataset(fichier[0])
    fichier = glob.glob(racineCMIP6+"/ScenarioMIP/"+scenario+"/SImon/siconc/*_"+model+"_*.nc")
    print(fichier) #GKtest
    fs = netCDF4.Dataset(fichier[0])
    # la grille peut etre differente de celle de T
    nlon = fh.variables['siconc'].shape[2]
    nlat = fh.variables['siconc'].shape[1]
    #
    n_inbin = np.zeros((nbin,nmon),int)
    sicbin = np.zeros((nbin,nmon,nlat,nlon),float) 
    #
    sicread = np.empty((nlat,nlon),float)
    spval = fh.variables["siconc"].missing_value
    for it in range(nt):
        if (it < nth):
            sicread = fh.variables["siconc"][it,:,:]
        else:
            sicread = fs.variables["siconc"][it-nth,:,:]
        ibin = np.argmin(np.abs(GSATsmooth[it]-Tbincen[:]))
        imon = month[it] - 1
        sicbin[ibin,imon,:,:] += sicread[:,:]
        n_inbin[ibin,imon] += 1
    #
    units = fh.variables["siconc"].units
    stdname = fh.variables["siconc"].standard_name
    longname = fh.variables["siconc"].long_name

    # normaliser les bins
    #
    for ibin in range(nbin):
        for imon in range(nmon):
            if (n_inbin[ibin,imon] > 0):
                sicbin[ibin,imon,:,:] /= n_inbin[ibin,imon]
            else:
                sicbin[ibin,imon,:,:] = spval

    # pour graphique warming level
    levels = [ 10., 25., 50., 75., 90.]
    nlev = len(levels)

    swl = np.empty((nlev,nmon,nlat,nlon),float)
    for ilev,level in enumerate(levels):
        xgtl = np.where( (sicbin > level) & (np.abs(sicbin/spval -1.) > epsilon), dtbin, 0)
        swl[ilev,:,:,:] = np.sum(xgtl, axis = 0)
        swl[ilev,:,:,:] = np.where( (np.abs(sicbin[int(nbin/2),:,:,:]/spval -1.) > epsilon), swl[ilev,:,:,:], spval) + Tbinmin

    # ecrire
    #
    f = netCDF4.Dataset('SiconcWarming/SiconcWarming_'+model+'_historical+'+scenario+'.nc',mode='w',format='NETCDF4_CLASSIC')
    #
    monname = 'month'
    month_dim = f.createDimension(monname, nmon)
    months = f.createVariable(monname, float, (monname,))
    months.units = 'months'
    months.long_name = 'month'
    months.standard_name = 'month'
    months.axis = 'T'
    #
    binname = 'GSAT'
    tbin_dim = f.createDimension(binname, nbin)
    tbins = f.createVariable(binname, float, (binname,))
    tbins.units = 'K'
    tbins.long_name = 'GSAT wrt. 1850-1900'
    tbins.standard_name = 'GSAT'
    tbins.axis = 'Z'
    #
    levelname = 'Level'
    lev_dim = f.createDimension(levelname, nlev)
    levs = f.createVariable(levelname, float, (levelname,))
    levs.units = '%'
    levs.long_name = 'Level'
    levs.standard_name = 'Level'
    levs.axis = 'Z'
    #
    # copier les dimensions spatiales de siconc (fichier original)
    dimlist = []
    for idim in range(1,3):
        dimlist.append(fh.variables['siconc'].dimensions[idim])
    try:
        dimlist.append(fh.variables['type'].dimensions[0])
    except:
        pass
    for dname, the_dim in fh.dimensions.items():
        if (dname in dimlist):
            f.createDimension(dname, len(the_dim))
    # copier les variables qui sont des coordonnees de siconc (car elles sont aussi des coordonnees de variables de sortie)
    #
    coordinates = fh.variables['siconc'].getncattr('coordinates')
    coordlist = coordinates.split()
    # Traitement special pour INM-CM4-8, INM-CM5-0 et E3SM-1-1:
    if ( model == 'INM-CM4-8') or (model == 'INM-CM5-0') or (model == 'E3SM-1-1'):
        coordlist = ['type', 'lat', 'lon']
    for v_name, varin in fh.variables.items():
        for coordinate in coordlist:
            if (v_name == coordinate) and (v_name != 'time'):
                #
                dimensions = varin.dimensions
                outVar = f.createVariable(v_name, varin.datatype, dimensions)
                #
                attributes = {k: varin.getncattr(k) for k in varin.ncattrs()}
                try:
                    del attributes["_FillValue"]
                except:
                    pass
                outVar.setncatts(attributes)
                #
                outVar[:] = varin[:]
    #
    sicbins = f.createVariable('siconcbin',float,(binname,monname,dimlist[0],dimlist[1]))
    sicbins.missing_value = spval
    sicbins.units = units
    sicbins.standard_name = stdname
    sicbins.long_name = longname
    sicbins.coordinates = coordinates
    #
    swls = f.createVariable('Limit',float,(levelname,monname,dimlist[0],dimlist[1]))
    swls.missing_value = spval
    swls.units = 'K'
    swls.standard_name = 'Limit'
    swls.long_name = 'Limit'
    swls.coordinates = coordinates
    #
    months[:] = month[:nmon]
    tbins[:] = Tbincen[:]
    levs[:] = levels[:]
    sicbins[:,:,:,:] = sicbin[:,:,:,:]
    swls[:,:,:,:] = swl[:,:,:,:]
    #
    f.close()
    #
    fh.close()
    fs.close()
