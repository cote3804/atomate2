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

IrO2_structure = Structure.from_file("/pscratch/sd/s/soge8904/jobflow/IrO2/POSCAR_IrO2")

generator = SinglePointSetGenerator()
generator.auto_kpoint_density = 100
# generator.user_settings={'lattice': {'R00': 0.0, 'R01': 0.0, 'R02': 8.528849994534,
#                           'R10': 8.528849994534, 'R11': 0.0, 'R12': 0.0,
#                           'R20': 0.0, 'R21': 6.00728999615, 'R22': 0.0},
# }

print(IrO2_structure)

maker = IonicMinMaker(
    input_set_generator=generator,
    run_jdftx_kwargs={"jdftx_cmd": cmd}
)

job_bulk = maker.make(IrO2_structure)

response = run_locally(job_bulk, store=store, create_folders=True)


from IPython import embed; embed()
