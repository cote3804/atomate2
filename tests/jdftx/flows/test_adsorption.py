from pymatgen.core.structure import Molecule, Structure
from atomate2.jdftx.flows.adsorption import AdsorptionMaker
from atomate2.jdftx.sets.core import SinglePointSetGenerator
from atomate2.jdftx.flows.adsorption import AdsorptionMaker
from atomate2.jdftx.jobs.adsorption import MolMinMaker
from atomate2.jdftx.jobs.adsorption import SurfaceMinMaker
from atomate2.jdftx.jobs.core import IonicMinMaker
from jobflow import run_locally



def test_adsorption_flow(mock_jdftx, jdftx_test_dir):
    ref_paths = {
        "molecule_ionic_min_molecule_H2O": "molecule_min_test",
        "ionic_minbulk_relax_job": "bulk_min_test",
        "surface_ionic_minslab_0": "slab_min_test",
        "surface_ionic_minads_H2O_ontop_0": "ads_min_test",
        }
    
    IrO2_structure = Structure.from_file("/pscratch/sd/s/soge8904/jobflow/IrO2/POSCAR_IrO2")

    print(IrO2_structure)

    species = ["O", "H", "H"]
    coordinates = [
        [0.0, 0.0, 0.0],    # Oxygen at origin
        [0.96, 0.0, 0.0],   # First hydrogen (bonded to O)
    [-0.26, 0.92, 0.0]  # Second hydrogen (bonded to O, ~104.5° angle)
    ]

# Create the Molecule object
    molecule = Molecule(species, coordinates)

    mock_jdftx(ref_paths)

    generator = SinglePointSetGenerator()
    generator.auto_kpoint_density = 100

    generator_mol = SinglePointSetGenerator()

    generator_bulk = SinglePointSetGenerator()
    generator_bulk.auto_kpoint_density = 100

    slab_relax_maker = SurfaceMinMaker(
        input_set_generator=generator)

    mol_relax_maker = MolMinMaker(
    input_set_generator=generator_mol
    )

    bulk_relax_maker = IonicMinMaker(
        input_set_generator=generator_bulk
    )

    flow = AdsorptionMaker(
        input_set_generator=generator,
        mol_relax_maker=mol_relax_maker,
        bulk_relax_maker=bulk_relax_maker,
        slab_relax_maker=slab_relax_maker,
        site_type=["ontop"],
        min_slab_size=1.0)

    floww = flow.make(molecules=[molecule], bulk=IrO2_structure)

    responses = run_locally(floww)
    import IPython
    IPython.embed()
    

