import os
from dotenv import load_dotenv, dotenv_values


class BaseConfig:
    """
    Config class to load environment variables from a .env file
    and expose them as properties using python-dotenv.
    """

    def __init__(self, env_path='config/.env', defaults=None, required=None):
        """
        Initialize the BaseConfig class by loading environment variables.

        :param env_path: Path to the .env file
        :param defaults: Dictionary of default values for environment variables
        :param required: List of required environment variable names
        """
        self._env = {}
        self._defaults = defaults or {}
        self._required = required or []
        self._load_env(env_path)
        self._validate_required()

    def _load_env(self, env_path):
        """
        Load environment variables from the specified .env file using python-dotenv.

        :param env_path: Path to the .env file
        """
        if not os.path.exists(env_path):
            raise FileNotFoundError(f"The .env file at '{env_path}' does not exist.")

        # Load environment variables into a dictionary without affecting os.environ
        self._env = dotenv_values(env_path)

        # Apply default values where environment variables are missing
        for key, value in self._defaults.items():
            self._env.setdefault(key, value)

    def _validate_required(self):
        """
        Validate that all required environment variables are present.
        """
        missing_vars = [var for var in self._required if var not in self._env]
        if missing_vars:
            raise EnvironmentError(f"Missing required environment variables: {', '.join(missing_vars)}")

    def _convert_type(self, value):
        """
        Convert string values to appropriate data types.

        :param value: The string value to convert
        :return: Converted value
        """
        if isinstance(value, str):
            # Attempt to convert to integer
            if value.isdigit():
                return int(value)
            # Attempt to convert to float
            try:
                float_val = float(value)
                return float_val
            except ValueError:
                pass
            # Attempt to convert to boolean
            if value.lower() in ('true', 'false'):
                return value.lower() == 'true'
            # Attempt to convert to list (comma-separated)
            if ',' in value:
                return [item.strip() for item in value.split(',')]
        return value

    def __getattr__(self, name):
        """
        Expose environment variables as attributes.

        :param name: Name of the environment variable
        :return: Value of the environment variable
        :raises AttributeError: If the environment variable is not found
        """
        try:
            value = self._env[name]
            return self._convert_type(value)
        except KeyError:
            raise AttributeError(f"'BaseConfig' object has no attribute '{name}'") from None

    def __repr__(self):
        return f"<BaseConfig {self._env}>"