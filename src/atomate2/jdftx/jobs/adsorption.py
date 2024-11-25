"""Core jobs for running JDFTx calculations."""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from jobflow import job, Response, Flow
from pymatgen.core.surface import (
    SlabGenerator,
    get_symmetrically_distinct_miller_indices,
    Slab
)

from pymatgen.core import Element, Molecule, Structure
from pymatgen.analysis.adsorption import AdsorbateSiteFinder


from atomate2.jdftx.jobs.base import BaseJdftxMaker
#from atomate2.jdftx.schemas.adsorption import AdsorptionDocument #need to create this schema
from atomate2.jdftx.sets.core import (
    IonicMinSetGenerator,
)

from atomate2.jdftx.sets.base import JdftxInputGenerator

logger = logging.getLogger(__name__)

@dataclass
class SurfaceMinMaker(BaseJdftxMaker):
    """Maker to create surface relaxation job."""
    name: str = "surface_ionic_min"
    input_set_generator: JdftxInputGenerator = field(
        default_factory= lambda: IonicMinSetGenerator(
            coulomb_truncation = True,
            auto_kpoint_density = 1000,
            calc_type="surface",
        )
    )

class MolMinMaker(BaseJdftxMaker):
    """Maker to create molecule relaxation job."""
    name: str = "surface_ionic_min"
    input_set_generator: JdftxInputGenerator = field(
        default_factory=IonicMinSetGenerator(
            coulomb_truncation = True,
            calc_type="molecule",
        )
    )

def get_boxed_molecule(molecule: Molecule) -> Structure:
    """Get the molecule structure.

    Parameters
    ----------
    molecule: Molecule
        The molecule to be adsorbed.

    Returns
    -------
    Structure
        The molecule structure.
    """
    return molecule.get_boxed_structure(10, 10, 10)


def _get_miller_indices(bulk_structure: Structure, max_index) -> list:
    """Returns a list of Crystallographic orientations (hkl)"""
    miller_indices = get_symmetrically_distinct_miller_indices(
        bulk_structure, max_index=max_index
    )
    #can add miller indices not wanted
    return list(miller_indices)




@job
def generate_slabs(
    bulk_structure: Structure,
    min_slab_size: float, #default in flow
    surface_idx: tuple, #default in flow
    min_vacuum_size: float, #default in flow
    min_lw: float, #default in flow
    in_unit_planes: bool = True,
    center_slab: bool = True,
    lll_reduce: bool = True,
    reorient_lattice: bool = True,
    super_cell: list = [2, 2, 1]

) -> list:

    slab_generator = SlabGenerator(
        bulk_structure,
        surface_idx,
        min_slab_size,
        min_vacuum_size,
        center_slab,
        in_unit_planes,
        lll_reduce,
        reorient_lattice
        
    )
    slabs = slab_generator.get_slabs()

    if not slabs:
        raise ValueError("No slabs could be generated with the given parameters")
    
    slabs = [slab for slab in slabs if not (slab.is_polar() and not slab.is_symmetric())]

    for slab in slabs:
        slab.make_supercell(super_cell)

    logger.info(f"Generated {len(slabs)} slabs for {surface_idx} surface")

    return slabs

@job
def run_slabs_job(
    slab_structures: list[Structure],
    min_maker: SurfaceMinMaker,
) -> Response:
    

    termination_jobs = []
    slab_outputs = defaultdict(list)

    for i, slab in enumerate(slab_structures):
        slab_job = min_maker.make(structure=slab)
        job.append_name(f"slab_{i}")
        termination_jobs.append(slab_job)

        slab_outputs["configuration_number"].append(i)
        slab_outputs["relaxed_structures"].append(slab_job.output.structure)
        slab_outputs["energies"].append(slab_job.output.output.energy)
        slab_outputs["forces"].append(slab_job.output.output.forces)

    slab_flow = Flow(jobs=termination_jobs, output=slab_outputs, name="slab_flow")
    return Response(replace=slab_flow)

@job
def pick_slab(slab_structures: list[Structure],
            slab_calcs_outputs: dict) -> Structure:
    """Pick the slab with the lowest surface energy."""
    slab_energies = slab_calcs_outputs["energies"]
    min_idx = slab_energies.index(min(slab_energies))
    return slab_structures[min_idx]

@job
def generate_ads_slabs()