# mypy: ignore-errors

"""Core definition of a JDFTx Task Document"""

import logging
import re
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, TypeVar, Union
from pymatgen.core import Structure
from custodian.qchem.jobs import QCJob
from emmet.core.qchem.calc_types import CalcType, LevelOfTheory, TaskType
from atomate2.jdftx.schemas.calculation import Calculation, CalculationInput, CalculationOutput
from emmet.core.structure import StructureMetadata
from monty.serialization import loadfn
from pydantic import BaseModel, Field

from atomate2.jdftx.schemas.calculation import JDFTxStatus
from atomate2.utils.datetime import datetime_str

__author__ = (
    "Cooper Tezak <cooper.tezak@colorado.edu>"
)

logger = logging.getLogger(__name__)
_T = TypeVar("_T", bound="TaskDoc")
# _DERIVATIVE_FILES = ("GRAD", "HESS")

class OutputDoc(BaseModel):
    initial_structure: Structure = Field(None, description="Input Structure object")
    optimized_structure: Optional[Structure] = Field(
        None, description="Optimized Structure object"
    )
    mulliken: Optional[List[Any]] = Field(
        None, description="Mulliken atomic partial charges and partial spins"
    )
    resp: Optional[Union[List[float], List[Any]]] = Field(
        None,
        description="Restrained Electrostatic Potential (RESP) atomic partial charges",
    )
    nbo: Optional[Dict[str, Any]] = Field(
        None, description="Natural Bonding Orbital (NBO) output"
    )

    frequencies: Optional[Union[Dict[str, Any], List]] = Field(
        None,
        description="The list of calculated frequencies if job type is freq (units: cm^-1)",
    )

    frequency_modes: Optional[Union[List, str]] = Field(
        None,
        description="The list of calculated frequency mode vectors if job type is freq",
    )

    @classmethod
    def from_qchem_calc_doc(cls, calc_doc: Calculation) -> "OutputDoc":
        """
        Create a summary of QChem calculation outputs from a QChem calculation document.

        Parameters
        ----------
        calc_doc
            A QChem calculation document.
        kwargs
            Any other additional keyword arguments

        Returns
        -------
        OutputDoc
            The calculation output summary
        """
        return cls(
            initial_molecule=calc_doc.input.initial_molecule,
            optimized_molecule=calc_doc.output.optimized_molecule,
            # species_hash = self.species_hash, #The three entries post this needs to be checked again
            # coord_hash = self.coord_hash,
            # last_updated = self.last_updated,
            final_energy=calc_doc.output.final_energy,
            dipoles=calc_doc.output.dipoles,
            enthalpy=calc_doc.output.enthalpy,
            entropy=calc_doc.output.entropy,
            mulliken=calc_doc.output.mulliken,
            resp=calc_doc.output.resp,
            nbo=calc_doc.output.nbo_data,
            frequencies=calc_doc.output.frequencies,
            frequency_modes=calc_doc.output.frequency_modes,
        )


class InputDoc(BaseModel):
    structure: Structure = Field(
        None,
        title="Input Structure",
        description="Input molecule and calc details for the QChem calculation",
    )

    parameters: Optional[Dict] = Field(
        None,
        description="JDFTx calculation parameters",
    )


    @classmethod
    def from_qchem_calc_doc(cls, calc_doc: Calculation) -> "InputDoc":
        """
        Create qchem calculation input summary from a qchem calculation document.

        Parameters
        ----------
        calc_doc
            A QChem calculation document.

        Returns
        -------
        InputDoc
            A summary of the input molecule and corresponding calculation parameters
        """
        try:
            lot_val = calc_doc.level_of_theory.value
        except AttributeError:
            lot_val = calc_doc.level_of_theory

        try:
            ct_val = calc_doc.calc_type.value
        except AttributeError:
            ct_val = calc_doc.calc_type
        # TODO : modify this to get the different variables from the task doc.
        return cls(
            initial_molecule=calc_doc.input.initial_molecule,
            rem=calc_doc.input.rem,
            level_of_theory=lot_val,
            task_type=calc_doc.task_type.value,
            tags=calc_doc.input.tags,
            solvation_lot_info=calc_doc.solvation_lot_info,
            # special_run_type = calc_doc.input.special_run_type,
            # smiles = calc_doc.input.smiles,
            calc_type=ct_val,
        )


class CustodianDoc(BaseModel):
    corrections: Optional[List[Any]] = Field(
        None,
        title="Custodian Corrections",
        description="List of custodian correction data for calculation.",
    )

    job: Optional[Union[Dict[str, Any], QCJob]] = Field(
        None,
        title="Custodian Job Data",
        description="Job data logged by custodian.",
    )


class TaskDoc(StructureMetadata):
    """
    Calculation-level details about JDFTx calculations
    """

    dir_name: Optional[Union[str, Path]] = Field(
        None, description="The directory for this JDFTx task"
    )

    task_type: Optional[Union[CalcType, TaskType]] = Field(
        None, description="the type of JDFTx calculation"
    )

    last_updated: str = Field(
        default_factory=datetime_str,
        description="Timestamp for this task document was last updated",
    )

    calc_inputs: Optional[CalculationInput] = Field(
        {}, description="JDFTx calculation inputs"
    )

    calc_outputs: Optional[CalculationOutput] = Field(
        None,
        description="JDFTx calculation outputs",
    )

    state: Optional[JDFTxStatus] = Field(
        None, description="State of this JDFTx calculation"
    )

    # implemented in VASP and Qchem. Do we need this?
    # it keeps a list of all calculations in a given task.
    # calcs_reversed: Optional[List[Calculation]] = Field(
    # None,
    # title="Calcs reversed data",
    # description="Detailed data for each JDFTx calculation contributing to the task document.",
    # )


    @classmethod
    def from_directory(
        cls: Type[_T],
        dir_name: Union[Path, str],
        store_additional_json: bool = True,
        additional_fields: Dict[str, Any] = None,
        **jdftx_calculation_kwargs,
    ) -> _T:
        """
        Create a task document from a directory containing JDFTx files.

        Parameters
        ----------
        dir_name
            The path to the folder containing the calculation outputs.
        store_additional_json
            Whether to store additional json files in the calculation directory.
        additional_fields
            Dictionary of additional fields to add to output document.
        **qchem_calculation_kwargs
            Additional parsing options that will be passed to the
            :obj:`.Calculation.from_qchem_files` function.

        Returns
        -------
        TaskDoc
            A task document for the JDFTx calculation
        """
        logger.info(f"Getting task doc in: {dir_name}")

        additional_fields = {} if additional_fields is None else additional_fields
        dir_name = Path(dir_name)
        calc_doc = Calculation.from_files(
            dir_name=dir_name,
            jdftxinput_file="inputs.in",
            jdftxoutput_file="output.out"
            )
        # task_files = _find_qchem_files(dir_name)

        # if len(task_files) == 0:
        #     raise FileNotFoundError("No JDFTx files found!")

        ### all logic for calcs_reversed ###
        # critic2 = {}
        # custom_smd = {}
        # calcs_reversed = []
        # for task_name, files in task_files.items():
        #     if task_name == "orig":
        #         continue
        #     else:
        #         calc_doc = Calculation.from_qchem_files(
        #             dir_name,
        #             task_name,
        #             **files,
        #             **qchem_calculation_kwargs,
        #         )
        #         calcs_reversed.append(calc_doc)
                # all_qchem_objects.append(qchem_objects)

        # Lists need to be reversed so that newest calc is the first calc, all_qchem_objects are also reversed to match
        # calcs_reversed.reverse()

        # all_qchem_objects.reverse()

        ### Custodian stuff ###
        # custodian = _parse_custodian(dir_name)
        # additional_json = None
        # if store_additional_json:
        #     additional_json = _parse_additional_json(dir_name)
        #     for key, _ in additional_json.items():
        #         if key == "processed_critic2":
        #             critic2["processed"] = additional_json["processed_critic2"]
        #         elif key == "cpreport":
        #             critic2["cp"] = additional_json["cpreport"]
        #         elif key == "YT":
        #             critic2["yt"] = additional_json["yt"]
        #         elif key == "bonding":
        #             critic2["bonding"] = additional_json["bonding"]
        #         elif key == "solvent_data":
        #             custom_smd = additional_json["solvent_data"]

        # orig_inputs = (
        #     CalculationInput.from_qcinput(_parse_orig_inputs(dir_name))
        #     if _parse_orig_inputs(dir_name)
        #     else {}
        # )

        # dir_name = get_uri(dir_name)  # convert to full path

        doc = cls.from_structure(
            meta_structure=calc_doc.output.structure,
            dir_name=dir_name,
            calc_outputs=calc_doc.output,
            calc_inputs=calc_doc.input
            # task_type=
            # state=_get_state()
        )

        doc = doc.model_copy(update=additional_fields)
        return doc

    @staticmethod
    def get_entry(
        calcs_reversed: List[Calculation], task_id: Optional[str] = None
    ) -> Dict:
        """
        Get a computed entry from a list of QChem calculation documents.

        Parameters
        ----------
        calcs_reversed
            A list of QChem calculation documents in reverse order.
        task_id
            The job identifier

        Returns
        -------
        Dict
            A dict of computed entries
        """
        entry_dict = {
            "entry_id": task_id,
            "task_id": task_id,
            "charge": calcs_reversed[0].output.molecule.charge,
            "spin_multiplicity": calcs_reversed[0].output.molecule.spin_multiplicity,
            "level_of_theory": calcs_reversed[-1].input.level_of_theory,
            "solvent": calcs_reversed[-1].input.solv_spec,
            "lot_solvent": calcs_reversed[-1].input.lot_solv_combo,
            "custom_smd": calcs_reversed[-1].input.custom_smd,
            "task_type": calcs_reversed[-1].input.task_spec,
            "calc_type": calcs_reversed[-1].input.calc_spec,
            "tags": calcs_reversed[-1].input.tags,
            "molecule": calcs_reversed[0].output.molecule,
            "composition": calcs_reversed[0].output.molecule.composition,
            "formula": calcs_reversed[
                0
            ].output.formula.composition.aplhabetical_formula,
            "energy": calcs_reversed[0].output.final_energy,
            "output": calcs_reversed[0].output.as_dict(),
            "critic2": calcs_reversed[
                0
            ].output.critic,  # TODO: Unclear about orig_inputs
            "last_updated": calcs_reversed[0].output.last_updated,
        }

        return entry_dict


def get_uri(dir_name: Union[str, Path]) -> str:
    """
    Return the URI path for a directory.

    This allows files hosted on different file servers to have distinct locations.

    Parameters
    ----------
    dir_name : str or Path
        A directory name.

    Returns
    -------
    str
        Full URI path, e.g., "fileserver.host.com:/full/payj/of/fir_name".
    """
    import socket

    fullpath = Path(dir_name).absolute()
    hostname = socket.gethostname()
    try:
        hostname = socket.gethostbyaddr(hostname)[0]
    except (socket.gaierror, socket.herror):
        pass
    return f"{hostname}:{fullpath}"


def _parse_custodian(dir_name: Path) -> Optional[Dict]:
    """
    Parse custodian.json file.

    Calculations done using custodian have a custodian.json file which tracks the makers
    performed and any errors detected and fixed.

    Parameters
    ----------
    dir_name
        Path to calculation directory.

    Returns
    -------
    Optional[Dict]
        The information parsed from custodian.json file.
    """
    filenames = tuple(dir_name.glob("custodian.json*"))
    if len(filenames) >= 1:
        return loadfn(filenames[0], cls=None)
    return None


def _parse_orig_inputs(
    dir_name: Path,
) -> Dict[str, Any]:
    """
    Parse original input files.

    Calculations using custodian generate a *.orig file for the inputs. This is useful
    to know how the calculation originally started.

    Parameters
    ----------
    dir_name
        Path to calculation directory.

    Returns
    -------
    Dict[str, Any]
        The original molecule, rem, solvent and other data.
    """
    orig_inputs = {}
    orig_file_path = next(dir_name.glob("*.orig*"), None)

    if orig_file_path:
        orig_inputs = QCInput.from_file(orig_file_path)

    return orig_inputs


def _parse_additional_json(dir_name: Path) -> Dict[str, Any]:
    """Parse additional json files in the directory."""
    additional_json = {}
    for filename in dir_name.glob("*.json*"):
        key = filename.name.split(".")[0]
        if key not in ("custodian", "transformations"):
            if key not in additional_json:
                additional_json[key] = loadfn(filename, cls=None)
    return additional_json


# TODO currently doesn't work b/c has_jdftx_completed method is not implemented
def _get_state(calc: Calculation) -> JDFTxStatus:
    """Get state from calculation document of JDFTx task."""
    if calc.has_jdftx_completed:
        return JDFTxStatus.SUCCESS
    else:
        return JDFTxStatus.FAILED

