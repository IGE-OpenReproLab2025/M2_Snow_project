# All the functions created 

from .libs import (
    os,
    glob, 
    xe,
    np,
    pd,
    plt,
    mcolors,
    xr,
    sns, 
    ccrs,
    cfeature,
    calendar,
    find_contours
)

def regrid_model_files(
    models,
    ref_grid,
    output_dir,
    filename_template,        
    input_dirs,               
    output_name_template,     
    method="nearest_s2d"
):
    os.makedirs(output_dir, exist_ok=True)

    for model in models:
        file_found = False
        for input_dir in input_dirs:
            file_pattern = os.path.join(input_dir, filename_template.format(model=model))
            files = glob.glob(file_pattern)
            if files:
                file_path = files[0]
                file_found = True
                break

        if not file_found:
            print(f"Aucun fichier trouvé pour le modèle {model}")
            continue

        print(f"Traitement du modèle {model} : {file_path}")

        with xr.open_dataset(file_path) as ds:
            regridder = xe.Regridder(ds, ref_grid, method=method, ignore_degenerate=True)
            ds_regridded = regridder(ds)

            output_filename = output_name_template.format(model=model)
            output_path = os.path.join(output_dir, output_filename)
            ds_regridded.to_netcdf(output_path)
            print(f"Fichier sauvegardé : {output_path}")

            del ds_regridded, regridder

def regrid_model_files_bis(
    models,
    ref_grid,
    output_dir,
    filename_template,
    input_dirs,
    output_name_template,
    method="nearest_s2d"
):
    os.makedirs(output_dir, exist_ok=True)

    for model in models:
        file_found = False
        for input_dir in input_dirs:
            file_pattern = os.path.join(input_dir, filename_template.format(model=model))
            files = glob.glob(file_pattern)
            if files:
                file_path = files[0]
                file_found = True
                break

        if not file_found:
            print(f"Aucun fichier trouvé pour le modèle {model}")
            continue

        print(f"Traitement du modèle {model} : {file_path}")

        with xr.open_dataset(file_path) as ds:
            # Étape 1 : Détecter noms de latitude/longitude
            lat_name = None
            lon_name = None
            for possible_lat in ['lat', 'latitude', 'nav_lat', 'navLatitude']:
                if possible_lat in ds:
                    lat_name = possible_lat
                    break
            for possible_lon in ['lon', 'longitude', 'nav_lon', 'navLongitude']:
                if possible_lon in ds:
                    lon_name = possible_lon
                    break

            if lat_name is None or lon_name is None:
                print(f"Coordonnées lat/lon non trouvées pour le fichier : {file_path}")
                continue

            # Étape 2 : Renommer pour standardiser
            ds = ds.rename({lat_name: 'lat', lon_name: 'lon'})
            if 'lat' not in ds.coords or 'lon' not in ds.coords:
                ds = ds.set_coords(['lat', 'lon'])

            try:
                # Étape 3 : Regridding
                regridder = xe.Regridder(ds, ref_grid, method=method, ignore_degenerate=True)
                ds_regridded = regridder(ds)

                # Étape 4 : Sauvegarde
                output_filename = output_name_template.format(model=model)
                output_path = os.path.join(output_dir, output_filename)
                ds_regridded.to_netcdf(output_path)
                print(f"✅ Fichier sauvegardé : {output_path}")

                del ds_regridded, regridder

            except ValueError as e:
                print(f"Erreur lors du regridding de {model} : {e}")
                continue

def recalculer_areacella(models, input_path, output_path, suffix="_hist_reprojete.nc", output_suffix="_gridarea_hist_recalcule.nc"):
    """
    Recalcule les champs 'areacella' à partir d'un ensemble de fichiers NetCDF projetés, en utilisant CDO.
    
    Args:
        models (list): Liste des noms de modèles.
        input_path (str): Répertoire contenant les fichiers NetCDF d'entrée.
        output_path (str): Répertoire où stocker les fichiers de sortie.
        suffix (str): Suffixe des fichiers d'entrée.
        output_suffix (str): Suffixe des fichiers de sortie.
    """
    os.makedirs(output_path, exist_ok=True)

    for model in models:
        input_file = os.path.join(os.path.expanduser(input_path), f"{model}{suffix}")

        # Vérifie si le fichier existe vraiment
        if not os.path.isfile(input_file):
            print(f"❌ Fichier manquant pour {model} : {input_file}")
            continue

        print(f"🔄 Traitement du modèle {model}")

        try:
            # Test de lecture du fichier avec xarray
            with xr.open_dataset(input_file, engine='netcdf4'):
                output_file = os.path.join(output_path, f"{model}{output_suffix}")

                # Construction et exécution de la commande CDO
                cdo_command = f"cdo gridarea {input_file} {output_file}"
                print(f"⚙️ Exécution : {cdo_command}")
                os.system(cdo_command)
        except Exception as e:
            print(f"❗ Erreur lors de l'ouverture du fichier pour le modèle {model} : {e}")

def dictionnaire_to_csv(data_dict, output_csv):
    """
    Convertit un dictionnaire de DataArrays mensuels en CSV avec des noms de mois abrégés (jan, feb, ...).
    
    """
    month_names = ['jan', 'feb', 'mar', 'apr', 'may', 'jun',
                   'jul', 'aug', 'sep', 'oct', 'nov', 'dec']
    
    rows = []

    for model, da in data_dict.items():
        if isinstance(da, xr.DataArray):
            values = da.values
            if len(values) == 12:
                row = {"modèle": model}
                row.update({month_name: val for month_name, val in zip(month_names, values)})
                rows.append(row)
            else:
                print(f"⚠️ Le modèle {model} n'a pas 12 mois : ignoré")
        else:
            print(f"⚠️ Entrée invalide pour {model} : type {type(da)} ignoré")

    df = pd.DataFrame(rows)
    ordered_columns = ["modèle"] + month_names
    df = df[ordered_columns]
    df.to_csv(output_csv, index=False)
    print(f"✅ Fichier CSV sauvegardé : {output_csv}")

def get_data(path_siconc, path_areacella, path_sftlf):
    """Charge les jeux de données NetCDF (glace de mer, superficie, fraction de terre)."""
    ds_siconc = xr.open_dataset(path_siconc)
    ds_area = xr.open_dataset(path_areacella)
    ds_sftlf = xr.open_dataset(path_sftlf)
    return ds_siconc, ds_area, ds_sftlf

def get_data_ref(path_sicbcs, path_areacella, path_landfrac):
    """Charge les jeux de données NetCDF (glace de mer, superficie, fraction de terre)."""
    ds_sicbcs = xr.open_dataset(path_sicbcs)
    ds_area = xr.open_dataset(path_areacella)
    ds_landfrac = xr.open_dataset(path_landfrac)
    return ds_sicbcs, ds_area, ds_landfrac

def extraction_variable_and_monthly_mean_nh(ds_siconc, sftlf= None, variable="siconc", start_year="1995", end_year="2014", lat_min=20):
    """Extrait la variable glace de mer, filtre par latitude, moyenne mensuelle, etc."""
    siconc = ds_siconc[variable].sel(time=slice(start_year, end_year))
    
    if lat_min is not None:
        siconc = siconc.sel(lat=siconc.lat >= lat_min)

    siconc = siconc.where(siconc < 1e19, float("nan"))

    max_val = siconc.max().item()
    if max_val > 1.5:
        print(f"La variable '{variable}' est en %. Conversion en fraction.")
        siconc = siconc / 100

    if sftlf is not None:
        siconc = siconc.where(sftlf < 50)

    siconc = siconc.clip(0, 1)

    seaice_cover = siconc.groupby("time.month").mean(dim="time")
    seaice_cover = (seaice_cover > 0.15).astype(int)

    monthly_mean = seaice_cover.mean(dim=("lat", "lon"))

    return siconc, seaice_cover, monthly_mean

def extraction_variable_and_monthly_mean_sh(ds_siconc, sftlf= None, variable="siconc", start_year="1995", end_year="2014", lat_min=-40):
    """Extrait la variable glace de mer, filtre par latitude, moyenne mensuelle, etc."""
    siconc = ds_siconc[variable].sel(time=slice(start_year, end_year))
    
    if lat_min is not None:
        siconc = siconc.sel(lat=siconc.lat <= lat_min)

    siconc = siconc.where(siconc < 1e19, float("nan"))

    max_val = siconc.max().item()
    if max_val > 1.5:
        print(f"La variable '{variable}' est en %. Conversion en fraction.")
        siconc = siconc / 100

    if sftlf is not None:
        siconc = siconc.where(sftlf < 50)

    siconc = siconc.clip(0, 1)

    seaice_cover = siconc.groupby("time.month").mean(dim="time")
    seaice_cover = (seaice_cover > 0.15).astype(int)

    monthly_mean = seaice_cover.mean(dim=("lat", "lon"))

    return siconc, seaice_cover, monthly_mean

def extraction_variable_and_monthly_mean_ref(ds_sicbcs, sftlf= None, variable="sicbcs", start_year="1995", end_year="2014", lat_min=-40):
    """Extrait la variable glace de mer, filtre par latitude, moyenne mensuelle, etc."""
    sicbcs = ds_sicbcs[variable].sel(time=slice(start_year, end_year))
    
    if lat_min is not None:
        sicbcs = sicbcs.sel(lat=sicbcs.lat <= lat_min)

    sicbcs = sicbcs.where(sicbcs < 1e19, float("nan"))

    max_val = sicbcs.max().item()
    if max_val > 1.5:
        print(f"La variable '{variable}' est en %. Conversion en fraction.")
        sicbcs = sicbcs / 100

    if sftlf is not None:
        sicbcs = sicbcs.where(sftlf < 10)

    sicbcs = sicbcs.clip(0, 1)

    seaice_cover = sicbcs.groupby("time.month").mean(dim="time")
    monthly_mean = seaice_cover.mean(dim=("lat", "lon"))

    return sicbcs, seaice_cover, monthly_mean

def extraction_variable(ds_siconc, sftlf=None, variable="siconcbin", lat_min=20):
    """Version simple de l'extraction, sans moyenne mensuelle."""
    siconc = ds_siconc[variable]

    if lat_min is not None:
        siconc = siconc.sel(lat=siconc.lat >= lat_min)

    siconc = siconc.where(siconc < 1e19, float("nan"))

    max_val = siconc.max().item()
    if max_val > 1.5:
        print(f"La variable '{variable}' est en %. Conversion en fraction.")
        siconc = siconc / 100

    if sftlf is not None:
        siconc = siconc.where(sftlf < 50)

    siconc = siconc.clip(0, 1)

    return siconc

def extraction_variable_sh(ds_siconc, sftlf=None, variable="siconcbin", lat_min=-40):
    """Version simple de l'extraction, sans moyenne mensuelle."""
    siconc = ds_siconc[variable]

    if lat_min is not None:
        siconc = siconc.sel(lat=siconc.lat <= lat_min)

    siconc = siconc.where(siconc < 1e19, float("nan"))

    max_val = siconc.max().item()
    if max_val > 1.5:
        print(f"La variable '{variable}' est en %. Conversion en fraction.")
        siconc = siconc / 100

    if sftlf is not None:
        siconc = siconc.where(sftlf < 50)

    siconc = siconc.clip(0, 1)

    return siconc

def seaice_surface_calculation(seaice_cover, areacella):
    """Calcule la surface de glace de mer en km²."""
    area_km2 = areacella * 1e-6
    seaice_cover_km2 = (seaice_cover * area_km2).groupby("month").sum(dim=["lat", "lon"], skipna=True)
    return seaice_cover_km2

def binary_mask_from_siconc(seaice_cover, threshold=0.15):
    mask = xr.where(seaice_cover > threshold, 1, 0)
    return mask

def plot_binary_mask(binary_mask, month=1, title="Carte couverture neigeuse", ax=None):
    if ax is None:
        plt.figure(figsize=(10, 5))
        ax = plt.axes(projection=ccrs.PlateCarree())
        show_plot = True
    else:
        show_plot = False
        
    binary_mask.isel(month=month).plot.pcolormesh(ax=ax, transform=ccrs.PlateCarree(), cmap='viridis')
    ax.coastlines()
    ax.set_title(title)

    if show_plot:
        plt.show

#def plot_seaice_cover_basic_nh(siconc, time_index=0, title="Carte couverture de glace de mer", ax=None):
   # """Affiche une carte simple de la couverture de glace de mer sans traitement des zéros ni sauvegarde."""
   # if ax is None:
    #    plt.figure(figsize=(10, 5))
    #    ax = plt.axes(projection=ccrs.NorthPolarStereo())
    #    show_plot = True
    #else:
    #    show_plot = False

    #siconc.isel(time=time_index).plot.pcolormesh(
    #    ax=ax,
    #    transform=ccrs.PlateCarree(),
    #    cmap='viridis'
    #)
    #ax.coastlines()
    #ax.gridlines(draw_labels=True)
    #ax.set_title(title)

    #if show_plot:
        #plt.show()

def plot_seaice_cover_basic_nh(siconc, time_index=0, title="Couverture glace de mer"):
    ax = plt.axes(projection=ccrs.NorthPolarStereo())
    siconc.isel(time=time_index).plot.pcolormesh(
        ax=ax,
        transform=ccrs.PlateCarree(),
        cmap='viridis',
        vmin=0, vmax=1,
        cbar_kwargs={"label": "Concentration"}
    )
    ax.set_extent([-180, 180, 40, 90], crs=ccrs.PlateCarree())
    ax.coastlines()
    ax.gridlines()
    ax.set_title(title)
    plt.show()

def plot_seaice_cover_basic_sh(siconc, time_index=0, title="Couverture glace de mer"):
    ax = plt.axes(projection=ccrs.SouthPolarStereo())
    siconc.isel(time=time_index).plot.pcolormesh(
        ax=ax,
        transform=ccrs.PlateCarree(),
        cmap='viridis',
        vmin=0, vmax=1,
        cbar_kwargs={"label": "Concentration"}
    )
    ax.set_extent([-180, 180, -40, -90], crs=ccrs.PlateCarree())
    ax.coastlines()
    ax.gridlines()
    ax.set_title(title)
    plt.show()

def plot_seaice_cover_basic_nh_ref(sicbcs, time_index=0, title="Carte couverture de glace de mer", ax=None):
    """Affiche une carte simple de la couverture de glace de mer sans traitement des zéros ni sauvegarde."""
    if ax is None:
        plt.figure(figsize=(10, 5))
        ax = plt.axes(projection=ccrs.NorthPolarStereo())
        show_plot = True
    else:
        show_plot = False

    sicbcs.isel(time=time_index).plot.pcolormesh(
        ax=ax,
        transform=ccrs.PlateCarree(),
        cmap='viridis'
    )
    ax.coastlines()
    ax.gridlines(draw_labels=True)
    ax.set_title(title)

    if show_plot:
        plt.show()

#def plot_seaice_cover_basic_sh(siconc, time_index=0, title="Carte couverture de glace de mer", ax=None):
#    """Affiche une carte simple de la couverture de glace de mer sans traitement des zéros ni sauvegarde."""
#    if ax is None:
#        plt.figure(figsize=(10, 5))
#        ax = plt.axes(projection=ccrs.SouthPolarStereo())
 #       show_plot = True
#    else:
 #       show_plot = False
#
 #   siconc.isel(time=time_index).plot.pcolormesh(
 #       ax=ax,
 #       transform=ccrs.PlateCarree(),
 #       cmap='viridis'
 #   )
 #   ax.coastlines()
 #   ax.gridlines(draw_labels=True)
 #   ax.set_title(title)
#
 #   if show_plot:
  #      plt.show()

def plot_snow_cover_advanced(snc, time_index=0, title="EC-Earth3", ax=None, save_path=None):
    """Affiche une carte avec gestion des zéros et sauvegarde optionnelle."""
    if ax is None:
        plt.figure(figsize=(8, 8))
        ax = plt.axes(projection=ccrs.NorthPolarStereo())
        show_plot = True
    else:
        show_plot = False

    data = snc.isel(time=time_index)

    cmap = plt.cm.viridis.copy()
    cmap.set_under('white')

    vmin = data.where(data >= 0.1).min().item()
    vmax = data.max().item()

    data.plot.pcolormesh(
        ax=ax,
        transform=ccrs.PlateCarree(),
        cmap=cmap,
        vmin=vmin,
        vmax=vmax
    )

    ax.coastlines()
    ax.gridlines(draw_labels=True)
    ax.set_title(title)

    if save_path:
        plt.savefig(save_path, dpi=300)

    if show_plot:
        plt.show()

def plot_monthly_mean(monthly_mean, xlabel="Mois", ylabel="Average Snow Cover", title="Monthly average snow cover"):
    """Trace la moyenne mensuelle de la couverture neigeuse."""
    plt.plot(range(1, 13), monthly_mean, marker="o")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid()
    plt.show()

def plot_surface_km2(seaice_cover_km2, xlabel="Mois", ylabel="Couverture glace de mer (km²)", title="Surface de glace de mer moyenne mensuelle"):
    """Trace la surface neigeuse par mois en km²."""
    plt.plot(range(1, 13), seaice_cover_km2, marker="o")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid()
    plt.show()

def plot_monthly_seaice_boxplot(
    df_long,
    reference_df=None,
    x_col='Month',
    y_col='seaice_extent',
    ref_label='Reference',
    title='Sea Ice Cover Extent - Historical',
    ylabel='Sea Ice Cover Extent (km²)',
    xlabel='Month',
    color='lightblue',
    ref_color='red',
    figsize=(10, 8),
    save_path=None
):
    """
    Génère un boxplot mensuel de la couverture de glace de mer avec une référence superposée.
    """
    plt.figure(figsize=figsize)
    sns.boxplot(x=x_col, y=y_col, data=df_long, color=color)

    if reference_df is not None:
        sns.scatterplot(
            x=x_col,
            y=y_col,
            data=reference_df,
            color=ref_color,
            s=50,
            label=ref_label,
            zorder=10
        )

    plt.title(title, fontsize=24)
    plt.ylabel(ylabel, fontsize=18)
    plt.xlabel(xlabel, fontsize=18)
    plt.xticks(rotation=45, fontsize=14)
    plt.yticks(fontsize=14)
    
    if reference_df is not None:
        plt.legend(fontsize=14)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"✅ Figure enregistrée : {save_path}")
    else:
        plt.show()

def plot_heatmap_errors(
    data,
    title="Erreur absolue",
    xlabel="Month",
    ylabel="Model",
    cmap="Reds",
    figsize=(15, 10),
    annot=False,
    fontsize_title=30,
    fontsize_labels=20,
    fontsize_ticks=14,
    save_path=None
):
    """
    Affiche une heatmap à partir d'un DataFrame (par ex. erreurs absolues mensuelles par modèle).
    """
    plt.figure(figsize=figsize)
    ax = sns.heatmap(data, annot=annot, cmap=cmap)

    plt.title(title, fontsize=fontsize_title)
    plt.xlabel(xlabel, fontsize=fontsize_labels)
    plt.ylabel(ylabel, fontsize=fontsize_labels)
    plt.xticks(rotation=45, fontsize=fontsize_ticks)
    plt.yticks(fontsize=fontsize_ticks)

    # Mise à jour de la barre de couleur
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=fontsize_ticks)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"✅ Figure enregistrée : {save_path}")
    else:
        plt.show()

def plot_seaice_on_ax(ax, datas, seaice_cover, season_label, sftlf, colors):
    """Affiche plusieurs couches de données de glace de mer sur une même carte."""
    data_crs = ccrs.PlateCarree()
    land_mask = sftlf < 50

    for label, da in datas.items():
        mask = (da == 1) & land_mask
        lon2d, lat2d = np.meshgrid(da.lon, da.lat)
        gsat_label = label.split("_")[1] if "reference" not in label else "0°C-ref"

        ax.contourf(
            lon2d, lat2d, mask,
            levels=[0.15, 1],
            colors=[colors[gsat_label]],
            transform=data_crs,
            alpha=0.6
        )

    if seaice_cover is not None:
        seaice_mask = (seaice_cover > 0.15)
        lon2d, lat2d = np.meshgrid(seaice_cover.lon, seaice_cover.lat)
        ax.contour(
            lon2d, lat2d, seaice_mask,
            levels=[0.15],
            colors='red',
            linewidths=1.3,
            transform=data_crs
        )

    ax.coastlines()
    ax.set_extent([-180, 180, 40, 90], crs=data_crs)
    ax.set_title(f'{season_label}', fontsize=22)
    gl = ax.gridlines(draw_labels=True, color='gray', linewidth=0.5, linestyle='-')
    gl.xlabel_style = {'size': 16}
    gl.ylabel_style = {'size': 16}

def plot_seaice_on_ax_sh(ax, datas, seaice_cover, season_label, sftlf, colors):
    """Affiche plusieurs couches de données de glace de mer sur une même carte."""
    data_crs = ccrs.PlateCarree()
    land_mask = sftlf < 50

    for label, da in datas.items():
        mask = (da == 1) & land_mask
        lon2d, lat2d = np.meshgrid(da.lon, da.lat)
        gsat_label = label.split("_")[1] if "reference" not in label else "0°C-ref"

        ax.contourf(
            lon2d, lat2d, mask,
            levels=[0.15, 1],
            colors=[colors[gsat_label]],
            transform=data_crs,
            alpha=0.6
        )

    if seaice_cover is not None:
        seaice_mask = (seaice_cover > 0.15)
        lon2d, lat2d = np.meshgrid(seaice_cover.lon, seaice_cover.lat)
        ax.contour(
            lon2d, lat2d, seaice_mask,
            levels=[0.15],
            colors='red',
            linewidths=1.3,
            transform=data_crs
        )

    ax.coastlines()
    ax.set_extent([-180, 180, -40, -90], crs=data_crs)
    ax.set_title(f'{season_label}', fontsize=22)
    gl = ax.gridlines(draw_labels=True, color='gray', linewidth=0.5, linestyle='-')
    gl.xlabel_style = {'size': 16}
    gl.ylabel_style = {'size': 16}

def adjust_label(label):
    """Convertit un label type '1.5°C' en '+1.5°C', et '0°C-ref' en 'Observations'."""
    if label == "0°C-ref":
        return "Observations"
    try:
        temp_val = float(label.replace("°C", ""))
        return f"+{temp_val:.1f}°C"
    except ValueError:
        return label

def scores_exp_normalises(valeurs):
    """
   Calculate normalized exponential scores from a table of values
    """
    min_val = np.min(valeurs)
    max_val = np.max(valeurs)
    # Éviter la division par zéro si max_val == min_val
    if max_val == min_val:
        return np.ones_like(valeurs)
    scores_exp = np.exp(-(valeurs - min_val) / (max_val - min_val))
    return scores_exp

def scores_gaussiens(valeurs, sigma_scale=4):
    """
Calculate Gaussian-type scores from a table of values
    """
    min_val = np.min(valeurs)
    max_val = np.max(valeurs)
    mu = min_val
    if max_val == min_val:
        return np.ones_like(valeurs)
    sigma = (max_val - min_val) / sigma_scale
    scores = np.exp(-((valeurs - mu) ** 2) / (2 * sigma ** 2))
    return scores
