import os
from pathlib import Path


def get_test_resource_directory():
    working_directory = Path(os.getcwd())
    for parent in working_directory.parents:
        if parent.name == "SmartCCG":
            resource_directory = parent / "resources" / "test"
            return resource_directory
