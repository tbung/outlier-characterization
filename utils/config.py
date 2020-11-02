import argparse
import toml


def strtobool(val):
    """Convert string to boolean value, respective values are what distutils
    uses."""

    if val in ["y", "yes", "t", "true", "on", "1", 1]:
        return True
    elif val in ["n", "no", "f", "false", "off", "0", 0]:
        return False
    else:
        raise ValueError("Boolean flag value not understood")


class Config:
    def __init__(self):
        pass

    def load(self, path):
        self.__dict__.update(toml.load(path))

    def save(self, path):
        with open(path, "w") as f:
            toml.dump(self.__dict__, f)

    def update(self, data):
        self.__dict__.update(data)

    def parse_args(self):
        """Parses config file and updates with additional args."""

        cf_parser = argparse.ArgumentParser()
        cf_parser.add_argument(
            "-c",
            "--config",
            type=argparse.FileType("r"),
            default="./config/default.toml",
        )
        args, extra_args = cf_parser.parse_known_args()

        params = toml.loads(args.config.read())

        param_parser = argparse.ArgumentParser()

        for param, value in params.items():
            # Boolean flags should accept an argument so that they can have
            # default values. By default python interprets "--flag False" as
            # True so we need to give it a conversion function
            param_parser.add_argument(
                f"--{param}",
                type=type(value) if type(value) is not bool else strtobool,
                default=value,
            )

        params.update(vars(param_parser.parse_args(extra_args)))

        self.update(params)

    def __str__(self):
        return str(self.__dict__)
