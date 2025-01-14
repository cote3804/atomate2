"""Core jobs for running JDFTx calculations."""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING


from jobflow import job, Response, Flow
from pymatgen.core.surface import (
    SlabGenerator,
    Slab
)

from pymatgen.core import Molecule, Structure
from pymatgen.analysis.adsorption import AdsorbateSiteFinder


from atomate2.jdftx.jobs.base import BaseJdftxMaker
#from atomate2.jdftx.schemas.adsorption import AdsorptionDocument #need to create this schema
from atomate2.jdftx.sets.core import (
    IonicMinSetGenerator,
)

from atomate2.jdftx.sets.base import JdftxInputGenerator

from atomate2.jdftx.jobs.base import BaseJdftxMaker
from atomate2.jdftx.sets.core import IonicMinSetGenerator

if TYPE_CHECKING:
    from atomate2.jdftx.sets.core import JdftxInputGenerator

logger = logging.getLogger(__name__)


@dataclass
class SurfaceMinMaker(BaseJdftxMaker):
    """Maker to create surface relaxation job."""

    name: str = "surface_ionic_min"
    input_set_generator: JdftxInputGenerator = field(
        default_factory=lambda: IonicMinSetGenerator(
            coulomb_truncation=True,
            auto_kpoint_density=1000,
            calc_type="surface",
        )
    )
@dataclass
class MolMinMaker(BaseJdftxMaker):
    """Maker to create molecule relaxation job."""
    name: str = "molecule_ionic_min"
    input_set_generator: JdftxInputGenerator = field(
        default_factory= lambda: IonicMinSetGenerator(
            coulomb_truncation = True,
            calc_type="molecule",
        )
    )

def get_boxed_molecules(molecules: list[Molecule]) -> list[Structure]:
    """Get the molecule structure.

    Parameters
    ----------
    molecules: list[Molecule]
        The molecules to be adsorbed.

    Returns
    -------
    list[Structure]
        The molecule structures.
    """
    molecule_structures = defaultdict(list)

    for i, molecule in enumerate(molecules):
        boxed_molecule = molecule.get_boxed_structure(10, 10, 10)
        molecule_structures[i] = boxed_molecule

    return  molecule_structures


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
    super_cell: list = [2, 2, 1],

) -> list:
    
    slab_configs = []

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

    for i, slab in enumerate(slabs):
        oriented_bulk = slab.oriented_unit_cell
        config = {
            "configuration_number": i,
            "oriented_unit_cell": oriented_bulk,
            "structure": slab,
        }
        slab_configs.append(config)
    return slab_configs

@job
def generate_ads_slabs(
    slab: Slab,
    adsorbates: list[Molecule],
    min_displacement: float,
    site_type: list[str],  # could be bridge, hollow
    symm_reduce: float = 1e-2
) -> list[Structure]:
    """Generate structures with adsorbates placed on the slab surface.
    
    Parameters
    ----------
    slab : Slab
        The optimized slab structure
    adsorbates : list[Structure]
        The adsorbates structures
    min_displacement : float
        Minimum distance between adsorbate and surface
    site_type : list[str]
        Type of adsorption site to consider
        
    Returns
    -------
    list[Structure]
        List of structures with adsorbates placed at different sites
    """
    asf = AdsorbateSiteFinder(slab)
    ads_configs = []

    for adsorbate in adsorbates:
        adsorbate_formula = adsorbate.composition.reduced_formula

        sites = asf.find_adsorption_sites(
            distance=min_displacement,
            positions=site_type,
            symm_reduce=symm_reduce
            )
        
        for site_type in site_type:
            for i, site in enumerate(sites[site_type]):
                ads_struct = asf.add_adsorbate(adsorbate, site)

                config = {
                    "adsorbate": adsorbate_formula,
                    "site_type": site_type,
                    "site_index": i,
                    "structure": ads_struct,
                    "site_coords": site
                }
                ads_configs.append(config)
        
    return ads_configs

@job
def run_molecule_job(
    molecule_structures: list[Structure],
    molecules: list[Molecule],
    min_maker: MolMinMaker,
) -> Response:
    
    molecule_jobs = []
    molecule_outputs = defaultdict(list)

    for i, molecule in enumerate(molecule_structures): #changed from in molecule_structures.item()
        molecule_job = min_maker.make(structure=molecule)
        job_name = molecules[i].composition.reduced_formula
        molecule_job.append_name(f"_molecule_{job_name}")
        molecule_jobs.append(molecule_job)

        molecule_outputs["configuration_number"].append(i)
        molecule_outputs["formula"].append(job_name)
        molecule_outputs["relaxed_structures"].append(molecule_job.output.calc_outputs.structure)
        molecule_outputs["energies"].append(molecule_job.output.calc_outputs.energy)
        molecule_outputs["forces"].append(molecule_job.output.calc_outputs.forces)

    molecule_flow = Flow(jobs=molecule_jobs, output=molecule_outputs, name="molecule_flow")
    return Response(replace=molecule_flow)

@job
def generate_dict(
    molecules_outputs:dict
) -> dict:
    molecule_energies = {}
    for i in range(len(molecules_outputs["configuration_number"])):
        molecule_energies[molecules_outputs["formula"][i]] = molecules_outputs["energies"][i]
    return molecule_energies

@job
def run_slabs_job(
    slabs_output: list[dict],
    min_maker: SurfaceMinMaker,
    bulk_structure: Structure,
    bulk_energy: float,
    calculate_surface_energy: bool = False,
) -> Response:
    
    if calculate_surface_energy and (bulk_structure is None or bulk_energy is None):
        raise ValueError(
            "bulk_structure and bulk_energy must be provided if calculate_surface_energy is True"
     )

    termination_jobs = []
    slab_outputs = defaultdict(list)
    slab_structures = [slab["structure"] for slab in slabs_output]

    for i, slab in enumerate(slab_structures):
        slab_job = min_maker.make(structure=slab)
        slab_job.append_name(f"slab_{i}")
        termination_jobs.append(slab_job)


        slab_outputs["configuration_number"].append(i)
        slab_outputs["relaxed_structures"].append(slab_job.output.calc_outputs.structure)
        slab_outputs["energies"].append(slab_job.output.calc_outputs.energy)
        slab_outputs["forces"].append(slab_job.output.calc_outputs.forces)

        

        # if calculate_surface_energy:
        #     surface_energy = calculate_surface_energies(
        #         slab_structure=slab,
        #         bulk_structure=bulk_structure,
        #         slab_energy=slab_job.output.calc_outputs.energy,
        #         bulk_energy=bulk_energy,
        #         slab_area=slab.surface_area
        #     )
        #     slab_outputs["surface_energies"].append(surface_energy)

    slab_flow = Flow(jobs=termination_jobs, output=slab_outputs, name="slab_flow")
    return Response(replace=slab_flow)

@job
def run_ads_job(
    ads_configs: list[dict],
    relax_maker: SurfaceMinMaker
) -> Response:
    
    ads_jobs = []
    ads_outputs = defaultdict(list)

    for i, config in enumerate(ads_configs):
        ads_job = relax_maker.make(structure=config["structure"])
        job_name = f"ads_{config['adsorbate']}_{config['site_type']}_{config['site_index']}"
        ads_job.append_name(job_name)
        ads_jobs.append(ads_job)

        ads_outputs["configuration_number"].append(i)
        ads_outputs["adsorbate"].append(config["adsorbate"])
        ads_outputs["site_type"].append(config["site_type"])
        ads_outputs["site_index"].append(config["site_index"])
        ads_outputs["relaxed_structures"].append(ads_job.output.calc_outputs.structure)
        ads_outputs["energies"].append(ads_job.output.output.calc_outputs.energy)
        ads_outputs["forces"].append(ads_job.output.output.calc_outputs.forces)

    ads_flow = Flow(jobs=ads_jobs, output=ads_outputs, name="ads_flow")
    return Response(replace=ads_flow)


@job
def calculate_adsorption_energy(
    ads_outputs: dict,
    slab_energy: float,
    molecule_energy: dict[str, float]
) -> dict:
    
    results = defaultdict(list)

    for i in range(len(ads_outputs["configuration_number"])):
        adsorbate = ads_outputs["adsorbate"][i]
        mol_energy = molecule_energy[adsorbate]
        ads_energy = ads_outputs["energies"][i] - (slab_energy+mol_energy)

        results["config_number"].append(i)
        results["adsorbate"].append(adsorbate)
        results["site_type"].append(ads_outputs["site_type"][i])
        results["site_index"].append(ads_outputs["site_index"][i])
        results["adsorption_energy"].append(ads_energy)
        results["structure"].append(ads_outputs["relaxed_structures"][i])
    
    return results

@job
def pick_slab(
    slabs_outputs: dict,
    surface_energies: list
    ) ->dict:
    """Pick the slab with the lowest energy."""

    if "surface_energies" is None:
        raise ValueError("Surface energies not found. Ensure surface energy calculation was enabled.")
    
    # min_energy_idx = min(
    #     range(len(slab_outputs["surface_energies"])),
    #     key=lambda i: slab_outputs["surface_energies"][i]
    # ) #this was for when surface energies were part of slabs_outputs

    min_energy_idx = min(enumerate(surface_energies), key=lambda x: x[1])[0] #surface energies are list here

    selected_slab = {
        "configuration_number": slabs_outputs["configuration_number"][min_energy_idx],
        "relaxed_structure": slabs_outputs["relaxed_structures"][min_energy_idx],
        "energy": slabs_outputs["energies"][min_energy_idx],
        "forces": slabs_outputs["forces"][min_energy_idx],
        "surface_energy": surface_energies[min_energy_idx]

    }

    logger.info(
        f"Selected slab configuration {selected_slab['configuration_index']} "
        f"with surface energy {selected_slab['surface_energy']:.3f} eV/Å²" #need to check units..
    )
    
    return selected_slab


# def calculate_surface_energies( #changed from calculate_surface_energy 
#         slab_structure: Slab,
#         bulk_structure: Structure,
#         slab_energy: float,
#         bulk_energy: float,
#         slab_area: float
# ) -> float:
#     bulk_composition = bulk_structure.composition.get_el_amt_dict()
#     slab_composition = slab_structure.composition.get_el_amt_dict()
#     total_bulk_atoms = bulk_structure.composition.num_atoms

#     bulk_mole_fractions = {
#         element: count / total_bulk_atoms
#         for element, count in bulk_composition.items()
#     }

#     for ref_element in bulk_composition:
#         slab_bulk_ratio = (
#             slab_composition[ref_element] /
#             (bulk_mole_fractions[ref_element] * total_bulk_atoms)
#         )

#         for element in slab_composition:
#             excess_deficiency = {
#                 element: round(
#                     (bulk_mole_fractions[element] * slab_composition[ref_element] /
#                      bulk_mole_fractions[ref_element]) - slab_composition[element],
#                      2
#                 )
#             }
#         metal_bulk_energies = {
#             "Ir": -76.494584,
#         } #just have Ir for now, see comment below
#         if all(value == int(value) for value in excess_deficiency.values()):
#             corrections = {
#                 element: metal_bulk_energies[element] * factor #need to decide how to get the metal bulk energies (energy per atom of bulk metal).
#                 #Ideally, this would be a fully formed dict already, or should we calculate them each time witht he same params as the bulk? I think having a fully formed dict would be best.
#                 for element, factor in excess_deficiency.items()
#                 if element != ref_element
#             }

#             surface_energy = (
#                 slab_energy -
#                 (slab_bulk_ratio * bulk_energy) +
#                 sum(corrections.values())
#             ) / (2 * slab_area)

#             return surface_energy
            

@job
def generate_surface_energies(
    slabs_outputs: list[Slab],
    bulk_structure: Structure,
    bulk_energy: float,
) -> list[float]:
    """
    Calculate surface energies for multiple oxide slabs using oxygen as reference.
    All slabs should contain the same elements as the bulk structure. 
    
    Args:
        slab_structures: List of surface slab structures
        bulk_structure: Bulk structure reference
        slab_energies: List of DFT energies for each slab (eV)
        bulk_energy: DFT energy of bulk (eV)
        slab_areas: Surface areas of each slab (Å²)
    
    Returns:
        list[float]: Surface energies (eV/Å²) for each slab
    """
    slab_structures = slabs_outputs["relaxed_structures"]
    slab_energies = slabs_outputs["energies"]
    
    if len(slab_structures) != len(slab_energies):
        raise ValueError("Number of slabs, energies must match")
        
    bulk_composition = bulk_structure.composition.get_el_amt_dict()
    total_bulk_atoms = bulk_structure.composition.num_atoms
    
    # Calculate bulk mole fractions once since same for all slabs
    bulk_mole_fractions = {
        element: count / total_bulk_atoms
        for element, count in bulk_composition.items()
    }
    
    surface_energies = []
    ref_element = "O"
    metal_bulk_energies = {
        "Ir": -76.494584,
    } #just have Ir for now, see comment below
    
    for slab, slab_energy in zip(slab_structures, slab_energies):
        slab_composition = slab.composition.get_el_amt_dict()
        slab_area = slab.surface_area

        # Calculate slab/bulk ratio using oxygen reference
        slab_bulk_ratio = (
            slab_composition[ref_element] /
            (bulk_mole_fractions[ref_element] * total_bulk_atoms)
        )
        
        # Calculate excess/deficiency for non-oxygen elements
        excess_deficiency = {
            element: round(
                (bulk_mole_fractions[element] * slab_composition[ref_element] /
                bulk_mole_fractions[ref_element]) - slab_composition[element],
                2
            )
            for element in slab_composition
            if element != ref_element
        }
        
        if not all(value == int(value) for value in excess_deficiency.values()):
            raise ValueError(f"Non-integer excess/deficiency factors found for slab {len(surface_energies)}")
            
        corrections = {
            element: metal_bulk_energies[element] * factor #need to decide how to get the metal bulk energies (energy per atom of bulk metal).
#                 #Ideally, this would be a fully formed dict already, or should we calculate them each time witht he same params as the bulk? I think having a fully formed dict would be best.

            for element, factor in excess_deficiency.items()
        }
        
        surface_energy = (
            slab_energy -
            (slab_bulk_ratio * bulk_energy) +
            sum(corrections.values())
        ) / (2 * slab_area)
        
        surface_energies.append(surface_energy)
    
    return surface_energies
        






