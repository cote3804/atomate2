from jobflow_remote import submit_flow
from jobflow import Flow
from pymatgen.core.structure import Structure, Molecule
from atomate2.jdftx.flows.adsorption import AdsorptionMaker
from atomate2.jdftx.jobs.adsorption import MolMinMaker
from atomate2.jdftx.jobs.adsorption import SurfaceMinMaker
from atomate2.jdftx.jobs.core import IonicMinMaker
from jobflow import run_locally, JobStore, JobflowSettings
from maggma.stores.mongolike import MemoryStore
from monty.serialization import dumpfn
from atomate2.jdftx.sets.core import SinglePointSetGenerator
from jobflow import SETTINGS

from jobflow import SETTINGS
print("Config location:", SETTINGS.CONFIG_FILE)
print("Current settings:", SETTINGS.model_dump())

IrO2_structure = Structure.from_file("/Users/sophi/DATA/POSCAR_IrO2")

#cmd = "/global/cfs/cdirs/m4025/Software/Perlmutter/JDFTx/build-gpu/jdftx_gpu"
cmd = 'docker run -t --rm -v "$(pwd):/root/research" jdftx bash -c "jdftx"'

species = ["O", "H", "H"]
coordinates = [
    [0.0, 0.0, 0.0],    # Oxygen at origin
    [0.96, 0.0, 0.0],   # First hydrogen (bonded to O)
    [-0.26, 0.92, 0.0]  # Second hydrogen (bonded to O, ~104.5° angle)
]

# Create the Molecule object
molecule = Molecule(species, coordinates)

generator = SinglePointSetGenerator(
    user_settings={"ion-species": ["/usr/local/share/jdftx/pseudopotentials/GBRV/ir_pbe_v1.2.uspp",
                   "/usr/local/share/jdftx/pseudopotentials/GBRV/o_pbe_v1.2.uspp",
                   "/usr/local/share/jdftx/pseudopotentials/GBRV/h_pbe_v1.4.uspp"]
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
#print(flow.input_set_generator.as_dict())

# flow.slab_relax_maker.input_set_generator.auto_kpoint_density = 100
# flow.slab_relax_maker.input_set_generator.user_settings = {"ionic-minimize": {"nIterations": 0}}
# flow.bulk_relax_maker.input_set_generator.auto_kpoint_density = 100
# flow.bulk_relax_maker.input_set_generator.user_settings = {"ionic-minimize": {"nIterations": 0},
#                                                            "ion-species": "/usr/local/share/jdftx/pseudopotentials/GBRV/$ID_pbe_v1.uspp"}
# flow.mol_relax_maker.input_set_generator.user_settings = {"ionic-minimize": {"nIterations": 0},
#                                                           "ion-species": "/usr/local/share/jdftx/pseudopotentials/GBRV/$ID_pbe_v1.uspp"}

# flow.bulk_relax_maker.input_set_generator.user_settings = {"ion-species": "/usr/local/share/jdftx/pseudopotentials/GBRV/$ID_pbe_v1.uspp"}
#flow.slab_relax_maker.run_jdftx_kwargs = {"jdftx_cmd": cmd}



floww = flow.make(molecules=[molecule], bulk=IrO2_structure)


#store = JobStore(MemoryStore(), additional_stores={"data": MemoryStore()})
#settings = JobflowSettings(JOB_STORE=store)

run_locally(floww, create_folders=True)
outputs = list(SETTINGS.JOB_STORE.query(load=True))
dumpfn(outputs, "outputs.json")