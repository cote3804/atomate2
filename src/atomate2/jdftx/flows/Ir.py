from atomate2.jdftx.jobs.core import IonicMinMaker
from pymatgen.core import Structure
from atomate2.jdftx.sets.core import SinglePointSetGenerator
from jobflow import run_locally
from jobflow import job, run_locally, JobStore
from maggma.stores import MongoStore

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

cmd = "srun -n 1 --gpu-bind=single:1 /global/cfs/cdirs/m4025/Software/Perlmutter/JDFTx/build-gpu/jdftx_gpu"

Ir_structure = Structure.from_file("/pscratch/sd/s/soge8904/jobflow/IrO2/POSCAR_Ir")

generator = SinglePointSetGenerator()

print(Ir_structure)

maker = IonicMinMaker(
    input_set_generator=generator,
    run_jdftx_kwargs={"jdftx_cmd": cmd}
)

job_bulk = maker.make(Ir_structure)

response = run_locally(job_bulk, store=store, create_folders=True)