"""Module defining base JDFTx input set and generator."""

from __future__ import annotations

import os
from collections import defaultdict
from dataclasses import dataclass, field
from importlib.resources import files as get_mod_path
from pathlib import Path
from typing import TYPE_CHECKING, Any
import logging

import numpy as np
from monty.serialization import loadfn
from pymatgen.core.units import ang_to_bohr, eV_to_Ha
from pymatgen.io.core import InputGenerator, InputSet
from pymatgen.io.jdftx.inputs import JDFTXInfile, JDFTXStructure
from pymatgen.io.vasp import Kpoints

from atomate2 import SETTINGS

if TYPE_CHECKING:
    from pymatgen.core import Structure
    from pymatgen.util.typing import Kpoint, PathLike


_BASE_JDFTX_SET = loadfn(get_mod_path("atomate2.jdftx.sets") / "BaseJdftxSet.yaml")
_BEAST_CONFIG = loadfn(get_mod_path("atomate2.jdftx.sets") / "BeastConfig.yaml")
_PSEUDO_CONFIG = loadfn(get_mod_path("atomate2.jdftx.sets") / "PseudosConfig.yaml")
FILE_NAMES = {"in": "init.in", "out": "jdftx.out"}


class JdftxInputSet(InputSet):
    """
    A class to represent a JDFTx input file as a JDFTx InputSet.

    Parameters
    ----------
    jdftxinput
        A JdftxInput object
    """

    def __init__(self, jdftxinput: JDFTXInfile, jdftxstructure: JDFTXStructure) -> None:
        self.jdftxstructure = jdftxstructure
        self.jdftxinput = jdftxinput

    def write_input(
        self,
        directory: str | Path,
        infile: PathLike = FILE_NAMES["in"],
        make_dir: bool = True,
        overwrite: bool = True,
    ) -> None:
        """Write JDFTx input file to a directory.

        Parameters
        ----------
        directory
            Directory to write input files to.
        make_dir
            Whether to create the directory if it does not already exist.
        overwrite
            Whether to overwrite an input file if it already exists.
        """
        directory = Path(directory)
        if make_dir:
            os.makedirs(directory, exist_ok=True)

        if not overwrite and (directory / infile).exists():
            raise FileExistsError(f"{directory / infile} already exists.")

        jdftxinput = condense_jdftxinputs(self.jdftxinput, self.jdftxstructure)
        print("jdftxinput_write_input:", jdftxinput)
        jdftxinput.write_file(filename=(directory / infile))

    @staticmethod
    def from_directory(
        directory: str | Path,
    ) -> JdftxInputSet:
        """Load a set of JDFTx inputs from a directory.

        Parameters
        ----------
        directory
            Directory to read JDFTx inputs from.
        """
        directory = Path(directory)
        jdftxinput = JDFTXInfile.from_file(directory / "input.in")
        jdftxstructure = jdftxinput.to_JDFTXStructure(jdftxinput)
        return JdftxInputSet(jdftxinput=jdftxinput, jdftxstructure=jdftxstructure)


@dataclass
class JdftxInputGenerator(InputGenerator):
    """A class to generate JDFTx input sets.

    Args:
        user_settings (dict): User JDFTx settings. This allows the user to
            override the default JDFTx settings loaded in the default_settings
            argument.
        coulomb_truncation (bool) = False:
            Whether to use coulomb truncation and calculate the coulomb
            truncation center. Only works for molecules and slabs.
        auto_kpoint_density (int) = 1000:
            Reciprocal k-point density for automatic k-point calculation. If
            k-points are specified in user_settings, they will not be
            overridden.
        potential (None, float) = None:
            Potential vs SHE for GC-DFT calculation.
        calc_type (str) = "bulk":
            Type of calculation used for setting input parameters. Options are:
            ["bulk", "surface", "molecule"].
        pseudopotentials (str) = "GBRV"
        config_dict (dict): The config dictionary used to set input parameters
            used in the calculation of JDFTx tags.
        default_settings: Default JDFTx settings.
    """

    # copy _BASE_JDFTX_SET to ensure each class instance has its own copy
    # otherwise in-place changes can affect other instances
    user_settings: dict = field(default_factory=dict)
    coulomb_truncation: bool = False
    auto_kpoint_density: int = 1000
    potential: None | float = None
    calc_type: str = "bulk"
    pseudopotentials: str = "GBRV"
    config_dict: dict = field(default_factory=lambda: _BEAST_CONFIG)
    
    _default_settings: dict = field(default_factory=lambda: _BASE_JDFTX_SET, repr=False)
    _settings: dict = field(init=False, repr=False, default_factory=dict)

    def __post_init__(self) -> None:
        """Initialize settings by combining defaults with user overrides."""
        valid_calc_types = ["bulk", "surface", "molecule"]
        if self.calc_type not in valid_calc_types:
            raise ValueError(
                f"calc type f{self.calc_type} not in list of supported calc "
                "types: {valid_calc_types}."
            )
        self._settings = self._default_settings.copy()
        # users can pass a jdftx tag as a key with value None
        # to remove the tag from the settings
        filtered_user_settings = {k: v for k, v in self.user_settings.items() if v is not None}

        # remove None keys from user_settings so that they
        # aren't use to update self.settings later.
        for key in self.user_settings:
            if self.user_settings[key] is None and key in self._settings:
                self._settings.pop(key)

        self._settings.update(filtered_user_settings)
        # set default coords-type to Cartesian
        if "coords-type" not in self._settings:
            self._settings["coords-type"] = "Cartesian"
        #self._apply_settings(self.settings)

    # def _apply_settings(
    #     self, settings: dict[str, Any]
    # ) -> None:  # settings as attributes
    #     for key, value in settings.items():
    #         setattr(self, key, value)

    def get_input_set(
        self,
        structure: Structure = None,
        **kwargs
    ) -> JdftxInputSet:
        """Get a JDFTx input set for a structure.

        Parameters
        ----------
        structure
            A Pymatgen Structure.
        **kwargs
            Additional settings to override defaults

        Returns
        -------
        JdftxInputSet
            A JDFTx input set.
        """
        # user_settings = self.user_settings.copy()
        # user_settings.update(kwargs)
        #print("user_settings_get_input_set:", user_settings)
        settings = self._settings.copy()
        print("user_settings:", self.user_settings)
        for key, value in kwargs.items():
            if key not in self.user_settings:
                settings[key] = value

        #self.settings.update(self.user_settings)
        
        print("settings.update:", settings)
        self._set_kgrid(settings, structure)
        self._set_coulomb_interaction(settings, structure)
        self._set_nbands(settings, structure)
        self._set_mu(settings)
        self._set_pseudos(settings)
        self._set_magnetic_moments(settings, structure)
        #self._apply_settings(self.settings) #here
        #print("self.settings:", self.settings)
        jdftx_structure = JDFTXStructure(structure)
        #jdftxinputs = self.settings
        jdftxinput = JDFTXInfile.from_dict(settings)
        print("jdftxinput_get_input_set:", jdftxinput)
        return JdftxInputSet(jdftxinput=jdftxinput, jdftxstructure=jdftx_structure)

    def _set_kgrid(self, settings, structure: Structure):
        """Get k-point grid.

        Parameters
        ----------
        structure
            A pymatgen structure.

        Returns
        -------
        Kpoints
            A tuple of integers specifying the k-point grid.
        """
        # never override k grid definition in user settings
        if "kpoint-folding" in self.user_settings:
            return
        # calculate k-grid with k-point density
        if self.calc_type != "molecule":
            kpoints = Kpoints.automatic_density(
                structure=structure, kppa=self.auto_kpoint_density
            )
            kpoints = kpoints.kpts[0]
            if self.calc_type == "surface":
                kpoints = (kpoints[0], kpoints[1], 1)
        elif self.calc_type == "molecule":
            kpoints = (1, 1, 1)
        kpoint_update = {
            "kpoint-folding": {
                "n0": kpoints[0],
                "n1": kpoints[1],
                "n2": kpoints[2],
            }
        }
        settings.update(kpoint_update)
        return

    def _set_coulomb_interaction(self, settings, structure: Structure) -> JDFTXInfile:
        """
        Set coulomb-interaction and coulomb-truncation for JDFTXInfile.

        Description

        Parameters
        ----------
        structure
            A pymatgen structure

        Returns
        -------
        jdftxinputs
            A pymatgen.io.jdftx.inputs.JDFTXInfile object

        """
        if "coulomb-interaction" in settings:
            return
        if self.calc_type == "bulk":
            settings["coulomb-interaction"] = {
                "truncationType": "Periodic",
            }
            return
        if self.calc_type == "surface":
            settings["coulomb-interaction"] = {
                "truncationType": "Slab",
                "dir": "001",
            }
        elif self.calc_type == "molecule":
            settings["coulomb-interaction"] = {
                "truncationType": "Isolated",
            }
        com = center_of_mass(structure=structure)
        if settings["coords-type"] == "Cartesian":
            com = com @ structure.lattice.matrix * ang_to_bohr
        elif settings["coords-type"] == "Lattice":
            com = com * ang_to_bohr
        settings["coulomb-truncation-embed"] = {
            "c0": com[0],
            "c1": com[1],
            "c2": com[2],
        }
        return

    def _set_nbands(self, settings, structure: Structure) -> None:
        """Set number of bands in DFT calculation."""
        if "elec-n-bands" in settings:
            return
        nelec = 0
        for atom in structure.species:
            nelec += _PSEUDO_CONFIG[self.pseudopotentials][str(atom)]
        nbands_add = int(nelec / 2) + 10
        nbands_mult = int(nelec / 2) * _BEAST_CONFIG["bands_multiplier"]
        settings["elec-n-bands"] = max(nbands_add, nbands_mult)

    def _set_pseudos(self, settings) -> None:
        """Set ion-species tag corresponding to pseudopotentials."""
        if SETTINGS.JDFTX_PSEUDOS_DIR is not None:
            pseudos_str = str(
                Path(SETTINGS.JDFTX_PSEUDOS_DIR) / Path(self.pseudopotentials)
            )
        else:
            pseudos_str = self.pseudopotentials

        add_tags = [
            pseudos_str + "/$ID" + suffix
            for suffix in _PSEUDO_CONFIG[self.pseudopotentials]["suffixes"]
        ]
        # do not override pseudopotentials in settings
        if "ion-species" in self.user_settings:
            return
        settings["ion-species"] = add_tags
        return

    def _set_mu(self, settings) -> None:
        """Set absolute electron chemical potential (fermi level) for GC-DFT."""
        # never override mu in settings
        if "target-mu" in settings or self.potential is None:
            return
        if "pcm-variant" in settings:
            solvent_model = settings["pcm-variant"]
        else:
            solvent_model = "CANDLE"
            logging.warning("No solvent model specified, using CANDLE to set mu.")
        ashep = _BEAST_CONFIG["ASHEP"][solvent_model]
        # calculate absolute potential in Hartree
        mu = (ashep - self.potential) * eV_to_Ha
        settings["target-mu"] = {"mu": mu}
        return

    def _set_magnetic_moments(self, settings, structure: Structure) -> None:
        """Set the magnetic moments for each atom in the structure.

        If the user specified magnetic moments as JDFTx tags, they will
        not be prioritized. The user can also set the magnetic moments in
        the site_params dictionary attribute of the structure. If neither above
        options are set, the code will initialize all metal atoms with +5
        magnetic moments.

        Parameters
        ----------
        structure
            A pymatgen structure

        Returns
        -------
        None
        """
        # check if user set JFDTx magnetic tags and return if true
        if (
            "initial-magnetic-moments" in settings
            or "elec-initial-magnetization" in settings
        ):
            return
        # if magmoms set on structure, build JDFTx tag
        if "magmom" in structure.site_properties:
            if len(structure.species) != len(structure.site_properties["magmom"]):
                raise ValueError(
                    f"length of magmom, {structure.site_properties['magmom']} "
                    "does not match number of species in structure, "
                    f"{len(structure.species)}."
                )
            magmoms = defaultdict(list)
            for magmom, species in zip(
                structure.site_properties["magmom"], structure.species, strict=False
            ):
                magmoms[species].append(magmom)
            tag_str = ""
            for element, magmom_list in magmoms.items():
                tag_str += f"{element} " + " ".join(list(map(str, magmom_list))) + " "
        # set magmoms to +5 for all metals in structure.
        else:
            magmoms = defaultdict(list)
            for species in structure.species:
                if species.is_metal:
                    magmoms[str(species)].append(3)
                else:
                    magmoms[str(species)].append(0)
            tag_str = ""
            for element, magmom_list in magmoms.items():
                tag_str += f"{element} " + " ".join(list(map(str, magmom_list))) + " "
        settings["initial-magnetic-moments"] = tag_str
        return


def condense_jdftxinputs(
    jdftxinput: JDFTXInfile, jdftxstructure: JDFTXStructure
) -> JDFTXInfile:
    """
    Combine JDFTXInfile and JDFTxStructure into complete JDFTXInfile.

    Function combines a JDFTXInfile class with calculation
    settings and a JDFTxStructure that defines the structure
    into one JDFTXInfile instance.

    Parameters
    ----------
        jdftxinput: JDFTXInfile
            A JDFTXInfile object with calculation settings.

        jdftxstructure: JDFTXStructure
            A JDFTXStructure object that defines the structure.

    Returns
    -------
        JDFTXInfile
            A JDFTXInfile that includes the calculation
            parameters and input structure.
    """
    # force Cartesian coordinates
    coords_type = jdftxinput.get("coords-type")
    return jdftxinput + JDFTXInfile.from_str(
        jdftxstructure.get_str(in_cart_coords=(coords_type == "Cartesian"))
    )


def center_of_mass(structure: Structure) -> np.ndarray:
    """
    Calculate center of mass.

    Parameters
    ----------
    structure: Structure
        A pymatgen structure

    Returns
    -------
    np.ndarray
        A numpy array containing the center of mass in fractional coordinates.
    """
    weights = [site.species.weight for site in structure]
    return np.average(structure.frac_coords, weights=weights, axis=0)
