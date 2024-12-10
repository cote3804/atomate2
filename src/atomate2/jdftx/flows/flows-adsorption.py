"""Flow for calculating surface adsorption energies."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from jobflow import Flow, Job, Maker
from pymatgen.core.structure import Molecule, Structure
from pathlib import Path
from atomate2.jdftx.jobs.adsorption import (
    generate_slabs,
    generate_slab,
    generate_ads_slabs,
    run_slabs_job,
    pick_slab,
    MolMinMaker,
    SurfaceMinMaker

)
from atomate2.jdftx.jobs.core import IonicMinMaker

@dataclass
class AdsorptionMaker(Maker):
    mol_relax_maker: Maker | None = field(default_factory=MolMinMaker)
    #mol_static_maker: Maker | None = field(default_factory=MolStaticMaker)
    bulk_relax_maker: Maker | None = field(default_factory=IonicMinMaker)
    slab_relax_maker: Maker | None = field(default_factory=SurfaceMinMaker)
    #slab_static_maker: Maker | None = field(default_factory=SlabStaticMaker)
    min_slab_size: float = 4.0
    min_vacuum: float = 20.0
    min_lw: float = 10.0
    surface_idx: tuple[int, int, int] = (1, 0, 0)
    max_index: int = 1

    def make(
        self,
        molecule: Molecule,
        bulk: Structure,
        prev_dir_mol: str | Path | None = None,
        prev_dir_bulk: str | Path | None = None,
    ) -> Flow:
        
        molecule_structure = molecule.get_boxed_structure(10, 10, 10)

        jobs: list[Job] = []

        if self.mol_relax_maker:
            mol_optimize_job = self.mol_relax_maker.make(
                molecule_structure
            )
            mol_optimize_job.append_name("mol_relax_job") #why not keep job name idk
            jobs += [mol_optimize_job]

        else: #added for testing bc no SP molecule maker
            mol_static_job = self.mol_static_maker.make(
            mol_optimize_job.output.structure, prev_dir=prev_dir_mol
        )  
            mol_static_job.append_name("mol_static_job")
            jobs += [mol_static_job]

        molecule_dft_energy = mol_optimize_job.output.output.energy 

        #building bulk to optimize
        slab_from_bulk = generate_slab(bulk_structure=bulk,
                                        surface_idx=self.surface_idx,
                                        min_vacuum_size=self.min_vacuum
                                        )
        oriented_bulk = slab_from_bulk[0].oriented_unit_cell

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

        jobs += [generate_slab_structures] #will return a list of Slab objects.
        slab_structures = generate_slab_structures.output 

        run_slab_calcs = run_slabs_job(slab_structures=slab_structures)

        jobs += [run_slab_calcs]
        slab_calcs_outputs = run_slab_calcs.output
        slab_calcs_structures = slab_calcs_outputs["relaxed_structures"]
        slab_calcs_energies = slab_calcs_outputs["energies"]

        slab_struct = pick_slab(slab_structures=slab_calcs_structures,
                                bulk_structure=optimized_bulk,
                                bulk_energy=optimized_bulk_energy,
                                slab_energies=slab_calcs_energies)

        jobs += [slab_struct]
        slab_structure = slab_struct.output

        generate_ads_slabs_structures = generate_ads_slabs(
            slab=slab_structure,
            molecule_structure=molecule)








        # if self.slab_relax_maker:
        #     slab_optimize_job = self.slab_relax_maker.make(slab_structure, prev_dir=None) 
        #     slab_optimize_job.append_name("slab_relax_job")
        #     jobs += [slab_optimize_job]

        # slab_static_job = self.slab_static_maker.make(
        #     slab_optimize_job.output.structure, prev_dir=None
        # )
        # slab_static_job.append_name("slab_static_job") #can do full relax + sp or single point on slab
        # jobs += [slab_static_job]

        # slab_dft_energy = slab_static_job.output.output.energy

        # optimized_slab = slab_static_job.output.structure

        # generate_adslabs_structures = generate_adslabs(
        #     optimized_slab, molecule_structure=molecule,
        #     min_lw=self.min_lw) #need to work on this method
        # jobs += [generate_adslabs_structures]
        # adslab_structures = generate_adslabs_structures.output

        # run_ads_calculation = run_adslabs_job(
        #     adslab_structures=adslab_structures,
        #     relax_maker=self.slab_relax_maker,
        #     static_maker=self.slab_static_maker,
        # )
        # jobs += [run_ads_calculation]





        