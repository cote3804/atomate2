from jobflow_remote.utils.examples import add
from atomate2.jdftx.flows.adsorption import AdsorptionMaker
from atomate2.jdftx.jobs.core import IonicMinMaker
from atomate2.jdftx.jobs.adsorption import MolMinMaker
from atomate2.jdftx.jobs.adsorption import SurfaceMinMaker
from atomate2.jdftx.sets.core import SinglePointSetGenerator
from pymatgen.core import Structure, Molecule
from jobflow import job, run_locally, JobStore
from maggma.stores import MongoStore
import os

#os.chdir("/global/homes/s/soge8904/jobflow/jdftx")


collection_name = "tests"
store = MongoStore(
    database="beast_fireworks_database",
    collection_name=collection_name,
    port=27017,
    host="mongodb07.nersc.gov",
    username="beast_fireworks_database_admin",
    password="A3LvqspsU4XdQ^",
    mongoclient_kwargs={"directConnection":True}
)
store = JobStore(docs_store=store, additional_stores={"data": store})
#source source /global/cfs/cdirs/m4025/sophie/jobflow/jdftx_env.sh
cmd = "/global/cfs/cdirs/m4025/Software/Perlmutter/JDFTx/build-gpu/jdftx"
print(cmd)

IrO2_structure = Structure.from_file("/pscratch/sd/s/soge8904/jobflow/IrO2/POSCAR_IrO2")


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




if __name__ == "__main__":
    run_locally(
        floww, 
        store=store
    )