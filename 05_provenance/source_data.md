# Data sources and preprocessing evidence

## LGHAP AOD

LGHAP v2: Global monthly mean 1-km gap-free AOD grids (2000-2024), Bai and Li, Zenodo record 15879684: https://zenodo.org/records/15879684 ; DOI https://doi.org/10.5281/zenodo.15879684 . The current study uses January 2015-December 2024. AOD is dimensionless. The provider lists the dataset as CC BY 4.0. Methodological reference: Bai et al. (2024), Earth System Science Data, 16, 2425-2448, https://doi.org/10.5194/essd-16-2425-2024 .

Historical local raw-to-grid evidence identifies nearest-neighbour sampling at each 0.1-degree grid centre, followed by matching monthly covariates. The processed AOD intermediate and frozen response are compared for all 161,160 keys by the verifier with zero numerical discrepancy. The local source archive and raw extraction history are not rerun by this repository check. Product availability alone is not proof of complete raw-data reproduction.

## ERA5

Copernicus Climate Change Service / ECMWF ERA5 monthly averaged single-level data:
https://cds.climate.copernicus.eu/datasets/reanalysis-era5-single-levels-monthly-means?tab=overview

Archived evidence identifies monthly `moda` data, temperature conversion from K to degrees C, and the precipitation `avgad` field in m day-1 converted by multiplying by 1000 to mm day-1. `tp` in the model table is not monthly accumulated mm. The final covariates have been matched to the original Anhui grid. The lightweight verifier does not claim to reconstruct every production operation from raw ERA5.

## NDVI and DEM

MOD13A3.061 monthly NDVI provider: https://lpdaac.usgs.gov/products/mod13a3v061/ . The point-source record uses scale 0.0001 and a fill value of -3000 before scaling. `NDVI_final` is the historical corrected, filled series retained exactly for comparability. Full-period interpolation can use later months, including across years. Original production QC/interpolation logs are incomplete; see the dedicated limitation and saved NDVI-removal sensitivity.

NASADEM elevation provider: https://lpdaac.usgs.gov/products/nasadem_hgtv001/ . `dem_traceable` is the source-traceable elevation (m) at the grid centres; no NTL column is supplied as a model predictor.

## Projection and traceability

The published-candidate table is an exact string-token projection of the historical frozen CSV. It preserves target/features, grid coordinates, keys and row order, adds an explicit original-row `sample_id`, and omits unused/legacy/path-bearing fields. Gzip compression does not change the decompressed tokens. Original and candidate file hashes and the projection contract are recorded in `projection_contract.json`; private original absolute paths are deliberately excluded.
