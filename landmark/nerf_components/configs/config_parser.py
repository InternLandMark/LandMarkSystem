import argparse
import sys
from importlib.machinery import SourceFileLoader
from pathlib import Path

from landmark.utils.config import Config


def get_default_parser():
    """Reads user command line and uses an argument parser to parse the input arguments.
    Input arguments include configuration, host, port, world size, local rank, backend for torch.distributed.

    Returns:
       Parser: Returns the parser with the default arguments, the user may add customized arguments into this parser.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="path to the config file")

    return parser


def parse_console_args(unknown_args):
    if unknown_args is None:
        return None
    parsed_args = {}
    i = 0
    while i < len(unknown_args):
        arg = unknown_args[i]
        if arg.startswith("--"):
            key = arg[2:]
            if i + 1 < len(unknown_args) and not unknown_args[i + 1].startswith("--"):
                value = unknown_args[i + 1]
                parsed_args[key] = value
                i += 1
            else:
                parsed_args[key] = True
        i += 1
    return parsed_args


# Copy from InternLM
class BaseConfig(Config):
    """This is a wrapper class for dict objects so that values of which can be
    accessed as attributes.

    Args:
        config (dict): The dict object to be wrapped.
    """

    @staticmethod
    def from_file(filename: str, console_args=None, merge_configs=True):
        """Reads a python file and constructs a corresponding :class:`Config` object.

        Args:
            filename (str): Name of the file to construct the return object.

        Returns:
            :class:`Config`: A :class:`Config` object constructed with information in the file.

        Raises:
            AssertionError: Raises an AssertionError if the file does not exist, or the file is not .py file
        """

        # check config path
        if isinstance(filename, str):
            filepath = Path(filename).absolute()
        elif isinstance(filename, Path):
            filepath = filename.absolute()

        assert filepath.exists(), f"{filename} is not found, please check your configuration path"

        # check extension
        extension = filepath.suffix
        assert extension == ".py", "only .py files are supported"

        # import the config as module
        remove_path = False
        if filepath.parent not in sys.path:
            sys.path.insert(0, (filepath))
            remove_path = True

        module_name = filepath.stem
        source_file = SourceFileLoader(fullname=str(module_name), path=str(filepath))
        module = source_file.load_module()  # pylint: disable=W4902,E1120,W1505

        # load into config
        config = BaseConfig()
        config["config"] = filename

        assert hasattr(module, "config"), f"{filename} does not have config attribute"

        config_list = module.config

        if merge_configs:
            for cfg in config_list:
                cfg.check_args()
                for arg_name, arg_val in cfg:
                    assert arg_name not in config, f"Duplicate attribute {arg_name} in configs."
                    # TODO check duplicated attributes in which configs (frank)
                    config._add_item(arg_name, arg_val)
        else:
            # TODO to be tested (frank)
            for cfg in config_list:
                cfg.check_args()
                cfg_cls_name = cfg.__class__.__name__
                config._add_item(cfg_cls_name, cfg)

        # process console args
        if console_args:
            for arg_name, arg_val in console_args.items():
                assert arg_name in config, f"Unknown argument {arg_name}."
                arg_type = type(config[arg_name])
                assert arg_type in (
                    int,
                    float,
                    str,
                    bool,
                ), f"Unsupported type {arg_type} for argument {arg_name}, please set this argument in config file."

                if arg_type is bool:
                    if isinstance(arg_val, bool):
                        pass
                    elif arg_val.lower() in ["true", "1"]:
                        arg_val = True
                    elif arg_val.lower() in ["false", "0"]:
                        arg_val = False
                    else:
                        raise ValueError(f"Unknown boolean value {arg_val} for argument {arg_name}.")
                # elif arg_type is list:
                #     try:
                #         arg_val = eval(arg_val)
                #     except Exception:
                #         raise ValueError(f"Unknown list value {arg_val} for argument {arg_name}.")
                else:
                    try:
                        arg_val = arg_type(arg_val)
                    except Exception as e:
                        raise ValueError(f"Invalid value {arg_val} for argument {arg_name}.") from e

                config.update({arg_name: arg_val})

        # remove module
        del sys.modules[module_name]
        if remove_path:
            sys.path.pop(0)

        return config

    def save_config(self, file_path: str) -> None:
        # all_attrs = copy.deepcopy(vars(self))
        # for config in self.config_list:
        #     all_attrs.update(vars(config))

        with open(file_path, "w", encoding="utf-8") as file:
            for attr_name in sorted(self):
                file.write(f"{attr_name} = {self[attr_name]}\n")
