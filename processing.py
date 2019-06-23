import numpy as np


def build_molecule_structure_dict(processed_atoms):
    """
    Converts processed_atoms table into a dict for faster access

    :param processed_atoms: processed atoms table (type: pd.DataFrame)
    :return: dictionary containing processed atoms (type: dict)
    """

    groupbys = processed_atoms.groupby("molecule_name")

    molecule_structure_dict = {}
    for gb_obect in groupbys:
        molecule_name = gb_obect[0]
        molecule_structure_dict[molecule_name] = {
            'elements': gb_obect[1]['element'].values,
            'atom_index': gb_obect[1]['atom_index'].values,
            'atoms': gb_obect[1][['x', 'y', 'z']].values
        }

    return molecule_structure_dict


def get_canvas(molecule_name, atom_from_index, atom_to_index,
               molecule_structure_dict, possible_elements_dict, range_of_values):
    """
    For a given atom in a molecule, gives a 3D matrix with the positioning of all atoms with respet to it

    :param molecule_name: name of molecule in molecule_structure_dic (type: str)
    :param atom_from_index: index of the atom inside the  molecule (type: int)
    :param atom_to_index: index of second atom in the pair (type: int)
    :param molecule_structure_dict: dictionary of processed  molecules (type: str)
    :param possible_elements_dict: possible elements that can be in a molecule ("H", "C", etc)
    :param range_of_values: range of values in the 3D grid (type: list[float])
    :return: 4D array of zeros except for the coordinates where there is an atom.
    Dimensions 1,2,3 are the coordinates, and 4 is the element (type: np.array)
    """

    n = len(range_of_values)
    atoms = molecule_structure_dict[molecule_name]["atoms"]
    atom = molecule_structure_dict[molecule_name]["atoms"][atom_from_index]
    elements = molecule_structure_dict[molecule_name]["elements"]

    canvas = np.zeros([n+1, n+1, n+1, len(possible_elements_dict) + 1])
    canvas = populate_canvas(canvas, atom, atoms, elements, possible_elements_dict, range_of_values)
    canvas = flag_target_atom(canvas, atom_to_index, atoms, range_of_values)

    return canvas


def flag_target_atom(canvas, atom_target_index, atoms, range_of_values):
    """
    Places target atom in last channel of canvas

    :param canvas: 4D matrix to be populated (type: np.array)
    :param atom_target_index: index of second atom in the pair (type: int)
    :param atoms: atoms to place in canvas with respect to atom (type: np.array)
    :param range_of_values: range of values in the 3D grid (type: list[float])
    :return: 4D array of zeros except for the coordinates where there is an atom.
    Dimensions 1,2,3 are the coordinates, and 4 is the element (type: np.array)
    """

    atoms_centered = atoms - atoms[atom_target_index]
    atoms_centered = np.round(atoms_centered, 2)
    atom_target = atoms_centered[atom_target_index]
    point_coordinates = get_coordinates(range_of_values, atom_target)

    canvas[point_coordinates[0],
           point_coordinates[1],
           point_coordinates[2], -1] = 1

    return canvas


def populate_canvas(canvas, atom, atoms, elements, possible_elements_dict, range_of_values):
    """
    Given an all-zeros canvas, populate it

    :param canvas: 4D matrix to be populated (type: np.array)
    :param atom: coordinates of atom of reference (type: np.array)
    :param atoms: atoms to place in canvas with respect to atom (type: np.array)
    :param elements: array with element type for each atom in atoms (type: np.array)
    :param possible_elements_dict: possible elements that can be in a molecule ("H", "C", etc)
    :param range_of_values: range of values in the 3D grid (type: list[float])
    :return: 4D array of zeros except for the coordinates where there is an atom.
    Dimensions 1,2,3 are the coordinates, and 4 is the element (type: np.array)
    """

    atoms_centered = atoms - atom
    atoms_centered = np.round(atoms_centered, 2)

    for element, point in zip(elements, atoms_centered):
        point_coordinates = get_coordinates(range_of_values, point)
        element_index = possible_elements_dict[element]
        canvas[point_coordinates[0],
               point_coordinates[1],
               point_coordinates[2], element_index] = 1

    return canvas


def get_coordinates(range_of_values, point):
    """
    Computes position of the values of point inside of range_of_values

    :param range_of_values: array containing a range of equally-espaced values (type: np.array)
    :param point: array to evaluate (type: np.array)
    :return: coordinates of point inside of array (type: list[int])
    """

    min_value = range_of_values[0]
    max_value = range_of_values[-1]
    n = len(range_of_values)

    def get_index(x):
        return int(np.floor(n * (x - min_value) / (max_value - min_value)))

    return [get_index(x) for x in point]
