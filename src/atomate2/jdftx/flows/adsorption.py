"""Flow for calculating surface adsorption energies."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from jobflow import Flow, Job, Maker
from pymatgen.core.structure import Molecule, Structure
from pathlib import Path
from atomate2.jdftx.jobs.base import BaseJdftxMaker
from atomate2.jdftx.jobs.adsorption import (
    get_boxed_molecules,
    run_molecule_job,
    make_dict,
    generate_slabs,
    generate_ads_slabs,
    run_slabs_job,
    pick_slab,
    run_ads_job,
    calculate_adsorption_energy,
    MolMinMaker,
    SurfaceMinMaker

)
from atomate2.jdftx.jobs.core import IonicMinMaker

@dataclass
class AdsorptionMaker(BaseJdftxMaker):
    mol_relax_maker: Maker | None = field(default_factory=MolMinMaker)
    bulk_relax_maker: Maker | None = field(default_factory=IonicMinMaker)
    slab_relax_maker: Maker | None = field(default_factory=SurfaceMinMaker)
    min_slab_size: float = 4.0
    min_vacuum: float = 20.0
    min_lw: float = 10.0
    surface_idx: tuple[int, int, int] = (1, 0, 0)
    max_index: int = 1
    site_type: list[str] = field(default_factory=lambda: ["ontop", "bridge", "hollow"])
    min_displacement: float = 2.0

    def make(
        self,
        molecules: list[Molecule],
        bulk: Structure,
        prev_dir_mol: str | Path | None = None,
        prev_dir_bulk: str | Path | None = None,
    ) -> Flow:
        
        molecule_structures =  get_boxed_molecules(molecules=molecules)

        jobs: list[Job] = []

        if self.mol_relax_maker:
            mol_optimize_job = run_molecule_job(
                molecule_structures,
                min_maker=self.mol_relax_maker
            )
            jobs += [mol_optimize_job]

        molecules_calc_outputs = mol_optimize_job.output

        molecule_energies_dict = make_dict(molecules_outputs=molecules_calc_outputs)
        jobs += [molecule_energies_dict]



        slab_from_unoptimized_bulk = generate_slabs(bulk_structure=bulk,
                                        min_slab_size=self.min_slab_size,
                                        surface_idx=self.surface_idx,
                                        min_vacuum_size=self.min_vacuum,
                                        min_lw=self.min_lw,
                                        )
        
        jobs += [slab_from_unoptimized_bulk]
        oriented_bulk = slab_from_unoptimized_bulk.output[0]["oriented_unit_cell"]
        print("oriented_bulk:", oriented_bulk)

        if self.bulk_relax_maker:
            bulk_optimize_job = self.bulk_relax_maker.make(
                oriented_bulk
            )
            bulk_optimize_job.append_name("bulk_relax_job")
            jobs += [bulk_optimize_job]


            optimized_bulk = bulk_optimize_job.output.calc_outputs.structure
            optimized_bulk_energy= bulk_optimize_job.output.calc_outputs.energy 

        else:   
            optimized_bulk = oriented_bulk # if no bulk relax

        generate_slab_structures = generate_slabs(
            bulk_structure=optimized_bulk,
            min_slab_size=self.min_slab_size,
            surface_idx=self.surface_idx,
            min_vacuum_size=self.min_vacuum,
            min_lw=self.min_lw,
        ) 

        jobs += [generate_slab_structures]
        slabs_output = generate_slab_structures.output


        run_slab_calcs = run_slabs_job(
            slabs_ouput=slabs_output,
            min_maker=self.slab_relax_maker,
            bulk_structure=optimized_bulk,
            bulk_energy=optimized_bulk_energy,
            calculate_surface_energy=True)

        jobs += [run_slab_calcs]
        slab_calcs_outputs = run_slab_calcs.output
        slab_calcs_structures = slab_calcs_outputs["relaxed_structures"]
        slab_calcs_energies = slab_calcs_outputs["energies"]
        slab_calcs_surface_energies = slab_calcs_outputs["surface_energies"]

        selected_slab = pick_slab(slab_outputs=slab_calcs_outputs)
        jobs += [selected_slab]

        slab_structure = selected_slab.output["relaxed_structure"]
        slab_energy = selected_slab.output["energy"]

        ads_structures = generate_ads_slabs(
            slab=slab_structure,
            adsorbates=molecules,
            min_displacement=self.min_displacement,
            site_type=self.site_type)

        jobs += [ads_structures]

        run_ads_calcs = run_ads_job(
            ads_configs=ads_structures.output,
            relax_maker=self.slab_relax_maker
        )

        jobs += [run_ads_calcs]

        ads_energies = calculate_adsorption_energy(
            ads_outputs=run_ads_calcs.output,
            slab_energy=slab_energy,
            molecule_energies=molecule_energies_dict.output
        )
        jobs += [ads_energies]

        return Flow(
            jobs=jobs,
            output=ads_energies.output
        )





        