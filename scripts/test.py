import geopandas as gpd
import os

REPO = r"/mnt/d/Becas y Proyectos/EY Challenge 2024/EY24"
assert os.path.isdir(
    REPO
), "No existe el repositorio. Revisar la variable REPO del codigo run_model"


BUILDING_GDF = gpd.read_parquet(rf"{REPO}/data/data_out/BUILDING_GDF.parquet")
print(BUILDING_GDF.crs)

print("Anduvo")
