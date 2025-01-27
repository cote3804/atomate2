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
    generate_dict,
    generate_slabs,
    generate_ads_slabs,
    run_slabs_job,
    generate_surface_energies,
    pick_slab,
    run_ads_job,
    calculate_adsorption_energy,
    MolMinMaker,
    SurfaceMinMaker

)
from atomate2.jdftx.jobs.core import IonicMinMaker
from IrO2.database.queries import start_helper

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
        
        molecule_structure = start_helper.get_example_molecule_struct()
        surface_structure = start_helper.get_example_surface()