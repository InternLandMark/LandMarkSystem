import os
import shutil
from collections import OrderedDict

import pytest
import torch


class Test3DGSCkptTransform:
    """
    Test register cache
    """

    def setup_class(self):
        self.cur_dir_path = os.path.dirname(os.path.abspath(__file__))

    @pytest.mark.skip
    def test_scaffold_gs_transform(self):
        self.ckpt_dir = "/mnt/hwfile/landmark/checkpoint/landmark_sys/scaffold_gs/ScaffoldGS_matrix_city_blockA"

        self.ckpt = os.path.join(self.ckpt_dir, "checkpoint.pth")
        ckpt = torch.load(self.ckpt, "cpu")
        new_state_dict = OrderedDict()
        trans_map = {
            "opacity_mlp": "anchor_decoder.mlp_opacity",
            "cov_mlp": "anchor_decoder.mlp_cov",
            "color_mlp": "anchor_decoder.mlp_color",
        }
        for key in ckpt.keys():
            if key in trans_map:
                print(f"found raw key {key}")
                for f_key in ckpt[key].keys():
                    new_key = trans_map[key] + "." + f_key
                    new_state_dict[new_key] = ckpt[key][f_key]
            else:
                print(f"not found raw key {key}")
        print(f"{new_state_dict.keys()=}")

        self.new_ckpt_dir = "log/new_test/scaffold_gs/ScaffoldGS_matrix_city_blockA/refactor"
        self.new_ckpt = os.path.join(self.new_ckpt_dir, "state_dict.th")

        os.makedirs(self.new_ckpt_dir, exist_ok=True)
        # copy ply
        self.point_ply = os.path.join(self.ckpt_dir, "point_cloud.ply")
        self.new_point_ply = os.path.join(self.new_ckpt_dir, "point_cloud.ply")
        shutil.copyfile(self.point_ply, self.new_point_ply)
        # write state dict
        torch.save(new_state_dict, self.new_ckpt)

    @pytest.mark.skip
    def test_octree_gs_transform(self):
        self.ckpt_dir = (
            "/mnt/hwfile/landmark/checkpoint/landmark_sys/octree_gs/"
            "OctreeGS_matrix_city_blockA/point_cloud/iteration_40000/"
        )

        self.ckpt = os.path.join(self.ckpt_dir, "checkpoint.pth")
        ckpt = torch.load(self.ckpt, "cpu")
        print(f"{ckpt.keys()=}")
        new_state_dict = OrderedDict()
        trans_map = {
            "opacity_mlp": "anchor_decoder.mlp_opacity",
            "cov_mlp": "anchor_decoder.mlp_cov",
            "color_mlp": "anchor_decoder.mlp_color",
        }
        for key in ckpt.keys():
            if key in trans_map:
                print(f"found raw key {key}")
                for f_key in ckpt[key].keys():
                    new_key = trans_map[key] + "." + f_key
                    new_state_dict[new_key] = ckpt[key][f_key]
            else:
                print(f"not found raw key {key}")
        print(f"{new_state_dict.keys()=}")

        self.new_ckpt_dir = "log/new_test/octree_gs/OctreeGS_matrix_city_blockA/refactor"
        self.new_ckpt = os.path.join(self.new_ckpt_dir, "state_dict.th")

        os.makedirs(self.new_ckpt_dir, exist_ok=True)
        # copy ply
        self.point_ply = os.path.join(self.ckpt_dir, "point_cloud.ply")
        self.new_point_ply = os.path.join(self.new_ckpt_dir, "point_cloud.ply")
        shutil.copyfile(self.point_ply, self.new_point_ply)
        # write state dict
        torch.save(new_state_dict, self.new_ckpt)
