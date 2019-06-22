import pandas as pd
import numpy as np
import os
from pathlib import Path


def read_and_process_structured(structures_path):
    """
    Reads structures files and parse them into a single table
    
    :param structures_path: path where all files are located (type: tr)
    :return: table with all molecule structures (type: pd.DataFrame)
    """

    structures_path = Path(structures_path)

    xyz_files = os.listdir(structures_path)
    xyz_files = list(filter(lambda x: '.xyz' in x, xyz_files))

    processed_atoms = []
    for file in xyz_files:
        structure_path = structures_path / file
        raw_xyz = read_file(structure_path)
        molecule_name = file.split(".")[0]
        n_atoms, atoms = process_xyz(raw_xyz, molecule_name)

        processed_atoms.append(atoms)

    processed_atoms = np.vstack(processed_atoms)
    processed_atoms = pd.DataFrame(processed_atoms)
    processed_atoms.columns = ['molecule_name', 'element', 'x', 'y', 'z', 'atom_index']
    
    return processed_atoms


def process_xyz(raw_xyz, molecule_name):
    """
    Process a raw molecule file in ".xyz" format into a np.array
    
    :param raw_xyz: raw file (type: str)
    :param molecule_name: molecule name (type: str)
    :return: processed molecule (type: np.array)
    """
    
    xyz_lines = raw_xyz.split("\n")

    n_atoms = int(xyz_lines[0])

    molecule_name = np.repeat(molecule_name, n_atoms).reshape([-1, 1])

    atoms = []
    for atom in xyz_lines[2:]:
        atoms.append(atom.split(" "))
    
    atom_index = np.arange(n_atoms).reshape([-1, 1])
    atoms = np.hstack([molecule_name, np.array(atoms), atom_index])

    return n_atoms, atoms


def read_file(file_path, encoding="utf-8"):
    """
    Read a file into a string
    
    :param file_path: path to file (type: str)
    :param encoding: encoding of file (type: str)
    :return: content of file (type: str)
    """
    
    with open(file_path, encoding=encoding) as f:
        raw_xyz = f.read()
    return raw_xyz
