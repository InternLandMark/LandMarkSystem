from .blender_dataset import BlenderDataset
from .city_dataset import CityDataset, CityGaussianDataset
from .colmap_dataset import ColmapGaussianDataset
from .matrixcity_dataset import MatrixCityDataset

# from .zhangjiang_dataset import ZhangjiangGaussianDataset

dataset_dict = {
    "city": CityDataset,
    "matrixcity": MatrixCityDataset,
    "blender": BlenderDataset,
}

dataset_dict_gs = {
    "city": CityGaussianDataset,
    "Colmap": ColmapGaussianDataset,  # TODO un-ready
}
