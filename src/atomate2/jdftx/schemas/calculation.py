"""Core definitions of a JDFTx calculation document."""

# mypy: ignore-errors

import logging
from pathlib import Path
from typing import Optional, Union

from pydantic import BaseModel, Field
from pymatgen.core.structure import Structure
from pymatgen.core.trajectory import Trajectory
from pymatgen.io.jdftx.inputs import JDFTXInfile
from pymatgen.io.jdftx.joutstructure import JOutStructure
from pymatgen.io.jdftx.outputs import JDFTXOutfile, JDFTXOutputs

from atomate2.jdftx.schemas.enums import CalcType, SolvationType, TaskType

__author__ = "Cooper Tezak <cote3804@colorado.edu>"
logger = logging.getLogger(__name__)


class Convergence(BaseModel):
    """Schema for calculation convergence."""

    converged: bool = Field(
        default=True, description="Whether the JDFTx calculation converged"
    )
    geom_converged: Optional[bool] = Field(
        default=True, description="Whether the ionic/lattice optimization converged"
    )
    elec_converged: Optional[bool] = Field(
        default=True, description="Whether the last electronic optimization converged"
    )
    geom_converged_reason: Optional[str] = Field(
        None, description="Reason ionic/lattice convergence was reached"
    )
    elec_converged_reason: Optional[str] = Field(
        None, description="Reason electronic convergence was reached"
    )

    @classmethod
    def from_jdftxoutfile(cls, jdftxoutfile: JDFTXOutfile) -> "Convergence":
        """Initialize Convergence from JDFTxOutfile."""
        converged = jdftxoutfile.converged
        jstrucs = jdftxoutfile.jstrucs
        geom_converged = jstrucs.geom_converged
        geom_converged_reason = jstrucs.geom_converged_reason
        elec_converged = jstrucs.elec_converged
        elec_converged_reason = jstrucs.elec_converged_reason
        return cls(
            converged=converged,
            geom_converged=geom_converged,
            geom_converged_reason=geom_converged_reason,
            elec_converged=elec_converged,
            elec_converged_reason=elec_converged_reason,
        )

    @classmethod
    def from_jdftxoutputs(cls, jdftxoutputs: JDFTXOutputs) -> "Convergence":
        """Initialize Convergence from JDFTxOutfile."""
        converged = jdftxoutput.converged
        jstrucs = jdftxoutput.jstrucs
        geom_converged = jstrucs.geom_converged
        geom_converged_reason = jstrucs.geom_converged_reason
        elec_converged = jstrucs.elec_converged
        elec_converged_reason = jstrucs.elec_converged_reason
        return cls(
            converged=converged,
            geom_converged=geom_converged,
            geom_converged_reason=geom_converged_reason,
            elec_converged=elec_converged,
            elec_converged_reason=elec_converged_reason,
        )


class RunStatistics(BaseModel):
    """JDFTx run statistics."""

    total_time: Optional[float] = Field(
        0, description="Total wall time for this calculation"
    )

    @classmethod
    def from_jdftxoutfile(cls, jdftxoutfile: JDFTXOutfile) -> "RunStatistics":
        """Initialize RunStatistics from JDFTXOutfile."""
        t_s = jdftxoutfile.t_s if hasattr(jdftxoutfile, "t_s") else None

        return cls(total_time=t_s)


class CalculationInput(BaseModel):
    """Document defining JDFTx calculation inputs."""

    structure: Structure = Field(
        None, description="input structure to JDFTx calculation"
    )
    jdftxinfile: dict = Field(None, description="input tags in JDFTx in file")

    @classmethod
    def from_jdftxinput(cls, jdftxinput: JDFTXInfile) -> "CalculationInput":
        """
        Create a JDFTx InputDoc schema from a JDFTXInfile object.

        Parameters
        ----------
        jdftxinput
            A JDFTXInfile object.

        Returns
        -------
        CalculationInput
            The input document.
        """
        return cls(
            structure=jdftxinput.structure,
            jdftxinfile=jdftxinput.as_dict(),
        )


class CalculationOutput(BaseModel):
    """Document defining JDFTx calculation outputs."""

    structure: Optional[Structure] = Field(
        None,
        description="optimized geometry of the structure after calculation",
    )
    parameters: Optional[dict] = Field(
        None,
        description="JDFTXOutfile dictionary from last JDFTx run",
    )
    forces: Optional[list] = Field(None, description="forces from last ionic step")
    energy: float = Field(None, description="Final energy")
    energy_type: str = Field(
        "F", description="Type of energy returned by JDFTx (e.g., F, G)"
    )
    mu: float = Field(None, description="Fermi level of last electronic step")
    lowdin_charges: Optional[list] = Field(
        None, description="Lowdin charges from last electronic optimizaiton"
    )
    total_charge: float = Field(
        None,
        description=(
            "Total system charge from last electronic step in number" "of electrons"
        ),
    )
    stress: Optional[list[list]] = Field(
        None, description="Stress from last lattice optimization step"
    )
    cbm: Optional[float] = Field(
        None,
        description="Conduction band minimum / LUMO from last electronic optimization",
    )
    vbm: Optional[float] = Field(
        None, description="Valence band maximum /HOMO from last electonic optimization"
    )
    trajectory: Optional[Trajectory] = (
        Field(None, description="Ionic trajectory from last JDFTx run"),
    )
    eigenvals: Optional[np.ndarray] = (
        Field(
            None,
            description="Kohn-Sham eigenvalues for each band-state in "
            "array of shape (state, band)"
            )
    )
    bandProjections: Optional[np.ndarray] = (
        Field(
            None,
            description="Complex projections of atomic orbitals onto band-states in "
            "array of shape (state, band, orbital)"
            )
    )
    orb_label_list: Optional[tuple[str, ...]] = (
        Field(
            None,
            description="List of descriptive orbital tags in the order they appear in bandProjections"
        )
    )

    @classmethod
    def from_jdftxoutputs(
        cls, jdftxoutputs: JDFTXOutputs, **kwargs
    ) -> "CalculationOutput":
        """
        Create a JDFTx output document from a JDFTXOutputs object.

        Parameters
        ----------
        jdftxoutputs
            A JDFTXOutputs object.

        Returns
        -------
        CalculationOutput
            The output document.
        """
        jdftxoutfile = jdftxoutputs.outfile
        optimized_structure: Structure = jdftxoutfile.structure
        forces = jdftxoutfile.forces.tolist() if hasattr(jdftxoutfile, "forces") else None
        if hasattr(jdftxoutfile, "stress"):
            stress = None if jdftxoutfile.stress is None else jdftxoutfile.stress.tolist()
        else:
            stress = None
        energy = jdftxoutfile.e
        energy_type = jdftxoutfile.eopt_type
        mu = jdftxoutfile.mu
        lowdin_charges = optimized_structure.site_properties.get("charges", None)
        # total charge in number of electrons (negative of oxidation state)
        total_charge = (
            jdftxoutfile.total_electrons_uncharged - jdftxoutfile.total_electrons
        )
        cbm = jdftxoutfile.lumo
        vbm = jdftxoutfile.homo
        structure = joutstruct_to_struct(joutstruct=optimized_structure)
        if kwargs.get("store_trajectory", True):
            trajectory: Trajectory = jdftxoutfile.trajectory
            trajectory = trajectory.as_dict()
        else:
            trajectory = None
        
        return cls(
            structure=structure,
            forces=forces,
            energy=energy,
            energy_type=energy_type,
            mu=mu,
            lowdin_charges=lowdin_charges,
            total_charge=total_charge,
            stress=stress,
            cbm=cbm,
            vbm=vbm,
            trajectory=trajectory,
            parameters=jdftxoutfile.to_dict(),
            eigenvals=jdftxoutputs.eigenvals,
            bandProjections=jdftxoutputs.bandProjections,
            orb_label_list=jdftxoutputs.orb_label_list,
        )

    @classmethod
    def from_jdftxoutfile(
        cls, jdftxoutfile: JDFTXOutfile, **kwargs
    ) -> "CalculationOutput":
        """
        Create a JDFTx output document from a JDFTXOutfile object.

        Parameters
        ----------
        jdftxoutput
            A JDFTXOutfile object.

        Returns
        -------
        CalculationOutput
            The output document.
        """
        optimized_structure: Structure = jdftxoutfile.structure
        forces = jdftxoutfile.forces.tolist() if hasattr(jdftxoutfile, "forces") else None
        if hasattr(jdftxoutfile, "stress"):
            stress = None if jdftxoutfile.stress is None else jdftxoutfile.stress.tolist()
        else:
            stress = None
        energy = jdftxoutfile.e
        energy_type = jdftxoutfile.eopt_type
        mu = jdftxoutfile.mu
        lowdin_charges = optimized_structure.site_properties.get("charges", None)
        # total charge in number of electrons (negative of oxidation state)
        total_charge = (
            jdftxoutfile.total_electrons_uncharged - jdftxoutfile.total_electrons
        )
        cbm = jdftxoutfile.lumo
        vbm = jdftxoutfile.homo
        structure = joutstruct_to_struct(joutstruct=optimized_structure)
        if kwargs.get("store_trajectory", True):
            trajectory: Trajectory = jdftxoutfile.trajectory
            trajectory = trajectory.as_dict()
        else:
            trajectory = None
        return cls(
            structure=structure,
            forces=forces,
            energy=energy,
            energy_type=energy_type,
            mu=mu,
            lowdin_charges=lowdin_charges,
            total_charge=total_charge,
            stress=stress,
            cbm=cbm,
            vbm=vbm,
            trajectory=trajectory,
            parameters=jdftxoutfile.to_dict(),
        )


class Calculation(BaseModel):
    """Full JDFTx calculation inputs and outputs."""

    dir_name: str = Field(None, description="The directory for this JDFTx calculation")
    input: CalculationInput = Field(
        None, description="JDFTx input settings for the calculation"
    )
    output: CalculationOutput = Field(
        None, description="The JDFTx calculation output document"
    )
    converged: Convergence = Field(None, description="JDFTx job conversion information")
    run_stats: RunStatistics = Field(0, description="Statistics for the JDFTx run")
    calc_type: CalcType = Field(None, description="Calculation type (e.g. PBE)")
    task_type: TaskType = Field(
        None, description="Task type (e.g. Lattice Optimization)"
    )
    solvation_type: SolvationType = Field(
        None, description="Type of solvation model used (e.g. LinearPCM CANDLE)"
    )

    @classmethod
    def from_files(
        cls,
        dir_name: Union[Path, str],
        jdftxinfile_rel_path: Union[Path, str],
        jdftxoutfile_rel_path: Union[Path, str],
        jdftxinfile_from_file_kwargs: Optional[dict] = None,
        jdftxoutputs_from_calc_dir_kwargs: Optional[dict] = None,
        calculationinput_from_jdftxinput_kwargs: Optional[dict] = None,
        calculationoutput_from_jdftxoutputs_kwargs: Optional[dict] = None,
        # task_name  # do we need task names? These are created by Custodian
    ) -> "Calculation":
        """
        Create a JDFTx calculation document from a directory and file paths.

        Parameters
        ----------
        dir_name
            The directory containing the JDFTx calculation outputs.
        jdftxinput_file
            Path to the JDFTx in file relative to dir_name.
        jdftxoutput_file
            Path to the JDFTx out file relative to dir_name.
        jdftxinput_kwargs
            Additional keyword arguments that will be passed to the
            :obj:`.JDFTXInFile.from_file` method
        jdftxoutput_kwargs
            Additional keyword arguments that will be passed to the
            :obj:`.JDFTXOutputs.from_calc_dir` method

        Returns
        -------
        Calculation
            A JDFTx calculation document.
        """
        jdftxinfile_path = dir_name / jdftxinfile_rel_path
        jdftxoutfile_path = dir_name / jdftxoutfile_rel_path

        kwarg_ref = {
            "input": jdftxinfile_from_file_kwargs,
            "output": jdftxoutputs_from_calc_dir_kwargs,
            "cinput": calculationinput_from_jdftxinput_kwargs,
            "coutput": calculationoutput_from_jdftxoutputs_kwargs,
        }
        for key, kwargs in kwarg_ref.items():
            if kwargs is None:
                kwarg_ref[key] = {}

        jdftxinput = JDFTXInfile.from_file(jdftxinfile_path, **kwarg_ref["input"])
        if not "outfile_name" in jdftxoutput_kwargs:
            jdftxoutput_kwargs["outfile_name"] = jdftxoutfile_rel_path
        jdftxoutputs = JDFTXOutputs.from_calc_dir(
            dir_name,
            **kwarg_ref["output"],
            )
        jdftxoutfile = jdftxoutput.outfile

        input_doc = CalculationInput.from_jdftxinput(jdftxinput, **kwarg_ref["cinput"])
        output_doc = CalculationOutput.from_jdftxoutputs(jdftxoutputs, **kwarg_ref["coutput"])
        logging.log(logging.DEBUG, f"{output_doc}")
        converged = Convergence.from_jdftxoutfile(jdftxoutfile)
        run_stats = RunStatistics.from_jdftxoutfile(jdftxoutfile)

        calc_type = _calc_type(output_doc)
        task_type = _task_type(output_doc)
        solvation_type = _solvation_type(input_doc)

        return cls(
            dir_name=str(dir_name),
            input=input_doc,
            output=output_doc,
            converged=converged,
            run_stats=run_stats,
            calc_type=calc_type,
            task_type=task_type,
            solvation_type=solvation_type,
        )


def _task_type(
    outputdoc: CalculationOutput,
) -> TaskType:
    """Return TaskType for JDFTx calculation."""
    jdftxoutput: dict = outputdoc.parameters
    if not jdftxoutput.get("geom_opt"):
        return TaskType("Single Point")
    if jdftxoutput.get("geom_opt_type") == "lattice":
        return TaskType("Lattice Optimization")
    if jdftxoutput.get("geom_opt_type") == "ionic":
        return TaskType("Ionic Optimization")
    # TODO implement MD and frequency task types. Waiting on output parsers

    return TaskType("Unknown")


def _calc_type(
    outputdoc: CalculationOutput,
) -> CalcType:
    jdftxoutput = outputdoc.parameters
    xc = jdftxoutput.get("xc_func", None)
    return CalcType(xc)


def _solvation_type(inputdoc: CalculationInput) -> SolvationType:
    jdftxinput: JDFTXInfile = inputdoc.jdftxinfile
    fluid = jdftxinput.get("fluid", None)
    if fluid is None:
        return SolvationType("None")
    fluid_solvent = jdftxinput.get("pcm-variant")
    fluid_type = fluid.get("type")
    solvation_type = f"{fluid_type} {fluid_solvent}"
    return SolvationType(solvation_type)


def joutstruct_to_struct(joutstruct: JOutStructure) -> Structure:
    """Convert JOutStructre to Structure."""
    lattice = joutstruct.lattice
    cart_coords = joutstruct.cart_coords
    species = joutstruct.species
    struct = Structure(
        lattice=lattice,
        coords=cart_coords,
        species=species,
        coords_are_cartesian=True,
    )
    for prop, values in joutstruct.site_properties.items():
        for isite, site in enumerate(struct):
            site.properties[prop] = values[isite]
    return struct
