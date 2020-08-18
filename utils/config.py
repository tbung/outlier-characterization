import argparse
import toml


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
        cf_parser.add_argument('-c', '--config', type=argparse.FileType('r'),
                               default='./config/default.toml')
        args, extra_args = cf_parser.parse_known_args()

        params = toml.loads(args.config.read())

        param_parser = argparse.ArgumentParser()

        for param, value in params.items():
            param_parser.add_argument(f'--{param}', type=type(value),
                                      default=value)

        params.update(vars(param_parser.parse_args(extra_args)))

        self.update(params)

    def __str__(self):
        return str(self.__dict__)
