import os
import sys
import git
import pathlib
PROJ_ROOT_PATH = pathlib.Path(git.Repo('.', search_parent_directories=True).working_tree_dir)
PROJ_ROOT =  str(PROJ_ROOT_PATH)
if PROJ_ROOT not in sys.path:
    sys.path.append(PROJ_ROOT)
    
    
# CONSTANT PATHS
# Seed
SEEDS_FOLDER = pathlib.Path(PROJ_ROOT_PATH / "seedfiles" )
pathlib.Path(SEEDS_FOLDER).mkdir(parents=True, exist_ok=True)

# Model Root Folder
MODELS_FOLDER = pathlib.Path(PROJ_ROOT_PATH / "models" )
pathlib.Path(MODELS_FOLDER).mkdir(parents=True, exist_ok=True)

# Weights Root Folder
WEIGHTS_FOLDER = pathlib.Path(PROJ_ROOT_PATH / "weights" )
pathlib.Path(WEIGHTS_FOLDER).mkdir(parents=True, exist_ok=True)