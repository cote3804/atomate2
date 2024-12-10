from jobflow_remote.utils.examples import add
from jobflow_remote import submit_flow
from jobflow_remote import submit_flow
from pymatgen.core.structure import Structure, Molecule
from atomate2.jdftx.flows.adsorption import AdsorptionMaker
from atomate2.jdftx.jobs.adsorption import MolMinMaker
from atomate2.jdftx.jobs.adsorption import SurfaceMinMaker
from atomate2.jdftx.jobs.core import IonicMinMaker
from atomate2.jdftx.sets.core import SinglePointSetGenerator



IrO2_structure = Structure.from_file("/Users/sophi/DATA/POSCAR_IrO2")

cmd = "/global/cfs/cdirs/m4025/Software/Perlmutter/JDFTx/build-gpu/jdftx_gpu"

species = ["O", "H", "H"]
coordinates = [
    [0.0, 0.0, 0.0],    # Oxygen at origin
    [0.96, 0.0, 0.0],   # First hydrogen (bonded to O)
    [-0.26, 0.92, 0.0]  # Second hydrogen (bonded to O, ~104.5° angle)
]

# Create the Molecule object
molecule = Molecule(species, coordinates)

generator = SinglePointSetGenerator(
    user_settings={
                   }
)
generator.auto_kpoint_density = 100
#print(generator.as_dict())

slab_relax_maker = SurfaceMinMaker(
    input_set_generator=generator,
    run_jdftx_kwargs={"jdftx_cmd": cmd}
)

mol_relax_maker = MolMinMaker(
    input_set_generator=generator,
    run_jdftx_kwargs={"jdftx_cmd": cmd}
)

bulk_relax_maker = IonicMinMaker(
    input_set_generator=generator,
    run_jdftx_kwargs={"jdftx_cmd": cmd}
)

flow = AdsorptionMaker(
    input_set_generator=generator,
    run_jdftx_kwargs={"jdftx_cmd": cmd},
    mol_relax_maker=mol_relax_maker,
    bulk_relax_maker=bulk_relax_maker,
    slab_relax_maker=slab_relax_maker,
    site_type=["ontop"],
    min_slab_size=2.0
)

floww = flow.make(molecules=[molecule], bulk=IrO2_structure)

resources = {
    "nodes": 1, 
    "ntasks": 1, 
    # "gpus_per_job":1, 
    # "time_limit": 600, # seconds
    "account": "m4025_g",
    "constraint": "gpu",
    "cpus_per_task": 32,
    "time": "06:00:00",
    # "gpus_per_task": 1,
    "gres": "gpu:1",
    "qos": "regular"
    # "queue_name": "regular",
}

print(submit_flow(floww, worker="perlmutter", project="IrO2_soge", resources=resources))