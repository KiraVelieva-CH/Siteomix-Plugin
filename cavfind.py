э
"""
class definitions for CavFind
"""
import os
import re
import time
from datetime import datetime
from logging import getLogger

import numpy as np

 
from .ligsite import (do_ligsite, find_cavities, find_cavities_dist, FP_DTYPE, mask_grid, setup_grid)
from .pdb_structure import PDBStructure

logger = getLogger(__name__)

default_settings = {
    "keep_hydrogens": (False, bool),
    "keep_hetatms": (False, bool),
    "keep_waters": (False, bool),
    "resn_water": (("HOH", "WAT", "H2O"), tuple),
    "alternate": ("A", str),
    "grid_spacing": (0.6, float),
    "probe_radius": (1.4, float),
    "softness": (0.5, float),
    "cushion": (0.0, float),
    "radii": ("UA", str),
    "radius_factor": (1.0, float),
    "ligsite_cutoff": (5, int),
    "gap": (0, int),
    "original_ligsite": (False, bool),
    "max_dist": (1.22, float),
    "vol_resol": (3.0, float),
    "min_size": (4, int),
    "max_size": (99999, int),
    "split_files": (False, bool),
    "shape_dmax": (3.0, float),
    "shape_object": ("sele", str),
}

tool_tips = {
    # PDB interpretation
    "keep_hydrogens": "Keep H-atoms for LigSite calculation (default: False).",
    "keep_hetatms": "Keep hetero-components for LigSite calculation (default: False).",
    "keep_waters": "Keep water molecules for LigSite calculation (default: False).",
    "resn_water": "Residue names for 'water'.",
    "alternate": "alternate conformation to be kept (default: 'A'), "
    + "use 'all' to keep all alternates.",
    # Grid setup
    "grid_spacing": "Grid spacing in Angs. (default: 0.6).",
    "probe_radius": "Probe size in Angs. (default: 1.4).",
    "softness": "Soft shell in grid units (default: 0.5).",
    "cushion": "Cushion around the coordinates (default: 0.).",
    "radii": "Atomic radii to use for: 'UA' (united-atoms, default), 'AA' (all-atom).",
    "radius_factor": "Multiplicative factor for atom radii (default: 1.0).",
    # LigSite
    "ligsite_cutoff": "Cutoff value for the LigSite algorithm (default: 5).",
    "gap": "Maximum gap for cavity coalescence (default: 0).",
    "original_ligsite": "Find cavities using a distance cutoff.",
    "max_dist": "Maximum distance used in cavity detection (default: 1.22)",
    "vol_resol": "Resolution for volume estimation, higher is more accurate (default: 3.0).",
    # 'old_cavfind': "Use the old GrowFromSeed algorithm (default: False).",
    "min_size": "Minimum size of a cavity in grid points (default: 4).",
    "max_size": "Maximum size of a cavity in grid points (default: 99999).",
    # 'min_volume': "Minimum size of a cavity in Angs.**3.",
    # 'max_volume': "Maximum size of a cavity in Angs.**3.",
    "split_files": "Generate separate files for each cavity (default: False).",
    # Annotation and Cropping
    "shape_dmax": "Maximum distance of a grid point to the shaping object (default: 3.0).",
    "shape_object": "PyMOL object/selection used for cavity shaping (default: sele)."
}

param_types = {
    # types of setting parameters:
    # 0 ... boolean
    # 1 ... integer
    # 2 ... float
    # 3 ... str
    # 4 ... tuple
    # PDB interpretation
    "keep_hydrogens": 0,
    "keep_hetatms": 0,
    "keep_waters": 0,
    "resn_water": 4,
    "alternate": 3,
    # Grid setup
    "grid_spacing": 2,
    "probe_radius": 2,
    "softness": 2,
    "cushion": 2,
    "radii": 3,
    "radius_factor": 2,
    # LigSite
    "ligsite_cutoff": 1,
    "gap": 1,
    "original_ligsite": 0,
    "max_dist": 2,
    "vol_resol": 2,
    # 'old_cavfind': 0,
    "min_size": 1,
    "max_size": 1,
    # 'min_volume': 2,
    # 'max_volume': 2,
    "split_files": 0,
    "shape_dmax": 2,
    "shape_object": 3
}

# global parameters
GRID_DTYPE = np.int8  # numerical type of the grid object
PROTEIN_FLAG = -100  # flag for grid points occupied by the protein
SOFT_FLAG = -99  # flag for grid points in the 'soft shell' around protein atoms


class CavFind:
    """
    CavFind object
    """

    def __init__(self, name, obj_name, settings=None):
        self.name = name
        self.obj_name = obj_name
        self.time_of_creation = datetime.now()

        if settings is not None:
            self.settings = settings.copy()
        else:
            self.settings = default_settings.copy()

        # structure for which cavities should be calculated (PDBStructure)
        self.pdb = None

        self.coords = None  # atom coordinates as numpy array
        self.radii = None  # atom radii as numpy array
        self.hp = None  # hydrophobicity parameters as numpy array
        self.charges = None  # partial charges

        self.grid = None  # grid object for LigSite algorithm
        self.cavities = None  # cavities detected in structure

    def struct_from_pdb(self, file_obj):
        """
        import a structure, assuming PDB-formatted data
        clean the structure according to setting parameters:
        'keep_hydrogens', 'keep_hetatms', 'keep_waters', 'alternates'

        :param file_obj: PDB data, file, list/tuple of lines, PyMOL pdbstr
        """
        self.pdb = PDBStructure(file_obj)

        keep_hydrogens = self.settings["keep_hydrogens"]
        keep_hetatms = self.settings["keep_hetatms"]
        keep_waters = self.settings["keep_waters"]
        alternate = self.settings["alternate"]
        resn_water = self.settings["resn_water"]

        if (
            not keep_hydrogens
            or not keep_hetatms
            or not keep_waters
            or alternate != "all"
        ):  # remove at least some atoms
            keep_atoms = []
            for atom in self.pdb.atom:

                alt_flag = atom.alternate in (alternate, " ", "") or alternate == "all"
                if not alt_flag:  # only keep atoms with the correct alternate flag
                    continue

                is_hydrogen = re.match("^[0-9H]", atom.name.strip()) is not None
                is_water = atom.residue in resn_water

                if not is_hydrogen and not atom.het:
                    # is not a hydrogen, not a HETATM and has the 'correct' alternate flag
                    keep_atoms.append(atom)
                    continue

                if is_water and not is_hydrogen and keep_waters:
                    # is a water, not a hydrogen and waters should be kept
                    keep_atoms.append(atom)
                    continue

                if atom.het and not is_hydrogen and not is_water and keep_hetatms:
                    # is a HETATM, not a hydrogen and HETATMs should be kept
                    keep_atoms.append(atom)
                    continue

                if is_hydrogen and keep_hydrogens:
                    # is a hydrogen and hydrogen should be kept
                    keep_atoms.append(atom)

            self.pdb.atom = keep_atoms

        rad_column = {"UA": 0, "AA": 1}.get(self.settings["radii"], 0)
        self.pdb.assign_rad_hp_chg(rad_column=rad_column)

        # Generate numpy arrays for coordinates, radii and hydrophobicity parameters
        coords = []
        radii = []
        charges = []
        for atom in self.pdb.atom:
            coords.append((atom.x, atom.y, atom.z))
            radii.append(atom.radius) 
            charges.append(atom.charge)
        self.coords = np.array(coords, dtype=FP_DTYPE)
        self.radii = np.array(radii, dtype=FP_DTYPE)
        self.charges = np.array(charges, dtype=FP_DTYPE)
        logger.info(f"Total charge: {np.sum(self.charges):2f}")

    def detect_binding_site(self, rerun=False, progress_callback=None):
        if not rerun:
            # perform a complete Ligsite calculation
            # setup and mask the grid
            self._emit_message(f"{self.name}:\nsetup grid", progress_callback)

            # setup and mask the grid
            start = time.perf_counter()
            self.grid = setup_grid(
                self.coords,
                self.settings["cushion"],
                self.settings["grid_spacing"],
                0,
                GRID_DTYPE,
            )
            mask_grid(
                self.grid,
                self.coords,
                self.radii,
                self.settings["radius_factor"],
                self.settings["probe_radius"],
                self.settings["softness"],
                PROTEIN_FLAG,
                SOFT_FLAG,
            )
            stop = time.perf_counter()
            logger.info(f"{self.name}: {stop - start:.2f} sec.")

            # run the LigSite algorithm and find cavities
            self._emit_message(
                f"{self.name}:\nanalyse grid\n({np.prod(self.grid.extent)} points)",
                progress_callback,
            )

            start = time.perf_counter()
            do_ligsite(self.grid, PROTEIN_FLAG)
            stop = time.perf_counter()
            logger.info(f"{self.name}: {stop - start:.2f} sec.")

        # detect cavities in the grid (entry point in case of re-run)
        self._emit_message(
            f"{self.name}:\ndetect cavities\n(cutoff: {self.settings['ligsite_cutoff']})",
            progress_callback,
        )

        start = time.perf_counter()
        if self.settings["original_ligsite"]:
            self.cavities = find_cavities_dist(
                self.grid,
                self.settings["ligsite_cutoff"],
                self.settings["max_dist"],
                self.settings["min_size"],
                self.settings["max_size"],
                self.settings["probe_radius"],
                self.settings["vol_resol"],
            )
        else:
            self.cavities = find_cavities(
                self.grid,
                self.settings["ligsite_cutoff"],
                self.settings["gap"],
                self.settings["min_size"],
                self.settings["max_size"],
                self.settings["probe_radius"],
                self.settings["vol_resol"],
            )
        stop = time.perf_counter()
        logger.info(f"{self.name}: {stop - start:.2f} sec.")


        return self  # necessary for asynchronous execution

    @staticmethod
    def _emit_message(message, progress_callback):
        if progress_callback:
            progress_callback.emit(message)
        logger.info(message.replace("\n", " "))

    def _get_pdb_header(self):
        """
        Generate REMARK lines for PDB-output,
        containing the settings used in the calculation
        :return: string with REMARK lines
        """
        header = [
            "REMARK",
            f"REMARK     alternate flag: {self.settings['alternate']}",
        ]

        if not self.settings["keep_hydrogens"]:
            header.append("REMARK                     *** hydrogen atoms excluded")
        if not self.settings["keep_waters"]:
            header.append("REMARK                     *** water molecules excluded")
        if not self.settings["keep_hetatms"]:
            header.append("REMARK                     *** hetero atoms excluded")

        header.extend(
            [
                "REMARK",
                f"REMARK       grid spacing:{self.settings['grid_spacing']:8.3f}",
                f"REMARK            cushion:{self.settings['cushion']:7.2f}",
                f"REMARK       probe radius:{self.settings['probe_radius']:7.2f}",
                f"REMARK           softness:{self.settings['softness']:7.2f}",
                f"REMARK      radius factor:{self.settings['radius_factor']:7.2f}",
                "REMARK",
                f"REMARK      cavity cutoff:{self.settings['ligsite_cutoff']:7.2f}",
                f"REMARK      gap parameter:{self.settings['gap']:7.2f}",
                f"REMARK   min. cavity size:{self.settings['min_size']:7d} points",
                f"REMARK   max. cavity size:{self.settings['max_size']:7d} points",
                "REMARK",
                "",
            ]
        )

        return "\n".join(header)

    def write_cavities(
        self,
        filename="cav.pdb",
        file_format="pdb",
        indices=None,
        annotation=None,
        header="",
    ):
        """

        :param indices:
        :param filename:
        :param file_format:
        :param annotation:
        :param header:
        :return:
        """
        # build list of annotations
        annotation_list = []
        if isinstance(annotation, list) or isinstance(annotation, tuple):
            annotation_list = list(annotation)
        elif isinstance(annotation, str):
            annotation_list = [
                item.strip() for item in annotation.split(",") if item.strip()
            ]

        if "LIG" not in annotation_list:  # always save the LigSite score
            annotation_list.append("LIG")

        # remove unavailable annotations
        annotation_list = [
            key for key in annotation_list if key in self.cavities[0].annotations
        ]

        basename, extension = os.path.splitext(filename)

        if file_format == "pdb":
            if extension != ".pdb":
                basename = filename
                extension = ".pdb"
            pdb_header = self._get_pdb_header()

            for annotation in annotation_list:
                annotation_header = f"REMARK Annotation: {annotation}\n"
                if annotation == "LIG":  
                    fname = f"{basename}{extension}"
                else:
                    fname = f"{basename}_{annotation}{extension}"

                with open(fname, "w") as file:
                    file.write(header)
                    file.write(pdb_header)
                    file.write(annotation_header)
                    for i, cav in enumerate(self.cavities):
                        # either write all cavities or those in 'indices'
                        if indices is None or i in indices:
                            cav.write(file, file_format, annotation)
                    file.write("END\n")
                    logger.info("Cavities written to: {:s}".format(fname))

                if self.settings["split_files"]:
                    # write a separate file for each selected cavity
                    for i, cav in enumerate(self.cavities):
                        # either write all cavities or those in 'indices'
                        if indices is None or i in indices:
                            if annotation == "LIG":  # naming compatibility with CavFind
                                fname = f"{basename}_{i + 1:04d}{extension}"
                            else:
                                fname = (
                                    f"{basename}_{annotation}_{i + 1:04d}{extension}"
                                )

                            with open(fname, "w") as file:
                                file.write(header)
                                file.write(pdb_header)
                                file.write(annotation_header)
                                cav.write(file, file_format, annotation)
                                file.write("END\n")
                                logger.info(f"Cavity #{i + 1:d} written to: {fname}")
        elif file_format == "csv":
            if extension != ".csv":
                basename = filename
                extension = ".csv"

            content = "# columns: x, y, z, " + ", ".join(annotation_list) + "\n"

            for i, cav in enumerate(self.cavities):
                # either write all cavities or those in 'indices'
                if indices is None or i in indices:
                    fname = f"{basename}_{i + 1:04d}{extension}"
                    with open(fname, "w") as file:
                        file.write(header)
                        file.write(f"# number of grid points:{cav.size:6d}\n")
                        file.write(content)
                        cav.write(file, "csv", annotation_list)
                        logger.info(f"Cavity #{i + 1} written to: {fname}")


