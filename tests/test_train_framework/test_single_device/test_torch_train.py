# pylint: disable=R1735
import logging
import os

import pytest

from benchmarks.nerf.gridnerf.gridnerf_trainer import GridNeRFTrainer
from benchmarks.nerf.instant_ngp.instant_ngp_trainer import NGPTrainer
from benchmarks.nerf.octree_gs.gs_trainer import OctreeGSTrainer
from benchmarks.nerf.origin_gs.gs_trainer import Gaussian3DTrainer
from benchmarks.nerf.scaffold_gs.gs_trainer import ScaffoldGSTrainer
from landmark.nerf_components.configs import BaseConfig


def setup_distributed_environment(world_size, rank):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29500"
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["RANK"] = str(rank)


class TestTorchTrainNoParallel:
    """
    Test torch training with no parallelism.
    """

    cur_dir_path = os.path.dirname(os.path.abspath(__file__))

    @pytest.mark.group_train_gn
    def test_train_gridnerf_no_parallel(self):
        setup_distributed_environment(world_size=1, rank=0)
        path = "benchmarks/nerf/gridnerf/confs/matrixcity_2block_single_train_ci.py"
        config_path = os.path.join(
            self.cur_dir_path,
            "..",
            "..",
            "..",
            path,
        )

        try:
            config = BaseConfig.from_file(config_path)
            if not config:
                logging.error("Configuration file loading failed.")
                pytest.fail("Failed to load the configuration file.")

            trainer = GridNeRFTrainer(config)

            PSNR = trainer.train()
            print("PSNRRRR: ", PSNR)
            assert PSNR >= 21.5, "PSNR is too low; training may have issues. Please check your training code."

        except ValueError as e:
            logging.error("An exception occurred during training: %s", e)
            pytest.fail("Test failed due to an exception: %s", e)

    @pytest.mark.group_train_in
    def test_train_instantNGP_no_parallel(self):
        setup_distributed_environment(world_size=1, rank=0)
        path = "benchmarks/nerf/instant_ngp/confs/matrixcity_2block_single_train_ci.py"
        config_path = os.path.join(
            self.cur_dir_path,
            "..",
            "..",
            "..",
            path,
        )

        try:
            config = BaseConfig.from_file(config_path)
            if not config:
                logging.error("Configuration file loading failed.")
                pytest.fail("Failed to load the configuration file.")

            trainer = NGPTrainer(config)

            PSNR = trainer.train()
            assert PSNR >= 23.2, "PSNR is too low; training may have issues. Please check your training code."

        except ValueError as e:
            logging.error("An exception occurred during training: %s", e)
            pytest.fail("Test failed due to an exception: %s", e)

    @pytest.mark.group_train_na
    def test_train_nerfacto_no_parallel(self):
        path = "benchmarks/nerf/nerfacto/confs/matrixcity_2block_single_train_ci.py"
        config_path = os.path.join(
            self.cur_dir_path,
            "..",
            "..",
            "..",
            path,
        )

        try:
            config = BaseConfig.from_file(config_path)
            if not config:
                logging.error("Configuration file loading failed.")
                pytest.fail("Failed to load the configuration file.")

            trainer = NGPTrainer(config)

            PSNR = trainer.train()
            assert PSNR >= 23.2, "PSNR is too low; training may have issues. Please check your training code."

        except ValueError as e:
            logging.error("An exception occurred during training: %s", e)
            pytest.fail("Test failed due to an exception: %s", e)

    @pytest.mark.group_train_org
    def test_train_origin_gs_no_parallel(self):
        setup_distributed_environment(world_size=1, rank=0)
        path = "benchmarks/nerf/origin_gs/confs/matrixcity_2block_train_ci.py"
        config_path = os.path.join(
            self.cur_dir_path,
            "..",
            "..",
            "..",
            path,
        )

        try:
            config = BaseConfig.from_file(config_path)
            if not config:
                logging.error("Configuration file loading failed.")
                pytest.fail("Failed to load the configuration file.")

            trainer = Gaussian3DTrainer(config)

            PSNR = trainer.train()
            assert PSNR > 24.53, "PSNR is too low; training may have issues. Please check your training code."

        except ValueError as e:
            logging.error("An exception occurred during training: %s", e)
            pytest.fail("Test failed due to an exception: %s", e)

    @pytest.mark.group_train_sg
    def test_train_scaffold_gs_no_parallel(self):
        setup_distributed_environment(world_size=1, rank=0)
        path = "benchmarks/nerf/scaffold_gs/confs/matrixcity_2block_single_train_ci.py"
        config_path = os.path.join(
            self.cur_dir_path,
            "..",
            "..",
            "..",
            path,
        )

        try:
            config = BaseConfig.from_file(config_path)
            if not config:
                logging.error("Configuration file loading failed.")
                pytest.fail("Failed to load the configuration file.")

            trainer = ScaffoldGSTrainer(config)

            PSNR = trainer.train()
            assert PSNR >= 27.02, "PSNR is too low; training may have issues. Please check your training code."

        except ValueError as e:
            logging.error("An exception occurred during training: %s", e)
            pytest.fail("Test failed due to an exception: %s", e)

    @pytest.mark.group_train_ocg
    def test_train_octree_gs_no_parallel(self):
        setup_distributed_environment(world_size=1, rank=0)
        path = "benchmarks/nerf/octree_gs/confs/matrixcity_2block_single_train_ci.py"
        config_path = os.path.join(
            self.cur_dir_path,
            "..",
            "..",
            "..",
            path,
        )

        try:
            config = BaseConfig.from_file(config_path)
            if not config:
                logging.error("Configuration file loading failed.")
                pytest.fail("Failed to load the configuration file.")

            trainer = OctreeGSTrainer(config)

            PSNR = trainer.train()
            assert PSNR > 27.05, "PSNR is too low; training may have issues. Please check your training code."
        except ValueError as e:
            logging.error("An exception occurred during training: %s", e)
            pytest.fail("Test failed due to an exception: %s", e)
