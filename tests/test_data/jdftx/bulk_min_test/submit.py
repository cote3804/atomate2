# from atomate2.jdftx.jobs.core import IonicMinMaker
# from pymatgen.core import Structure
# from atomate2.jdftx.sets.core import SinglePointSetGenerator
# from jobflow import run_locally
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

# cmd = "srun -n 1 --gpu-bind=single:1 /global/cfs/cdirs/m4025/Software/Perlmutter/JDFTx/build-gpu/jdftx_gpu"

# IrO2_structure = Structure.from_file("/pscratch/sd/s/soge8904/jobflow/IrO2/POSCAR_IrO2")

from atomate2.jdftx.jobs.adsorption import SurfaceMinMaker
from pymatgen.core import Structure
from atomate2.jdftx.sets.core import SinglePointSetGenerator
from jobflow import run_locally
from pymatgen.core.surface import SlabGenerator
from atomate2.jdftx.jobs.core import IonicMinMaker



cmd = "srun -n 1 --gpu-bind=single:1 /global/cfs/cdirs/m4025/Software/Perlmutter/JDFTx/build-gpu/jdftx_gpu"

from pymatgen.core import Lattice, Element, PeriodicSite 
# Lattice parameters
# Conversion factor from Bohr to Angstrom
bohr_to_angstrom = 0.52917721092

# Lattice parameters (converted to Angstrom)
lattice_matrix = [
    [0.000000000000, 0.000000000000, 8.528849994534 * bohr_to_angstrom],
    [8.528849994534 * bohr_to_angstrom, 0.000000000000, 0.000000000000],
    [0.000000000000, 6.007289996150 * bohr_to_angstrom, 0.000000000000]
]

# Define the species and coordinates (converted to Angstrom)
species = ["Ir", "Ir", "O", "O", "O", "O"]
coordinates = [
    [4.264424641093 * bohr_to_angstrom, 4.264424641093 * bohr_to_angstrom, 3.003645584499 * bohr_to_angstrom],
    [0.000000083481 * bohr_to_angstrom, 0.000000083481 * bohr_to_angstrom, 6.007289745975 * bohr_to_angstrom],
    [5.896306617340 * bohr_to_angstrom, 5.896306617340 * bohr_to_angstrom, 6.007289745975 * bohr_to_angstrom],
    [1.631882059728 * bohr_to_angstrom, 6.896967222458 * bohr_to_angstrom, 3.003645584499 * bohr_to_angstrom],
    [2.632542664846 * bohr_to_angstrom, 2.632542664846 * bohr_to_angstrom, 6.007289745975 * bohr_to_angstrom],
    [6.896967222458 * bohr_to_angstrom, 1.631882059728 * bohr_to_angstrom, 3.003645584499 * bohr_to_angstrom]
]


# Create the Structure object
lattice = Lattice(lattice_matrix)
print(lattice)
structure = Structure(lattice, species, coordinates, coords_are_cartesian=True)


generator = SinglePointSetGenerator()
generator.auto_kpoint_density = 100

print(structure)

maker = IonicMinMaker(
    input_set_generator=generator,
    run_jdftx_kwargs={"jdftx_cmd": cmd}
)

job_bulk = maker.make(structure)

response = run_locally(job_bulk, store=store, create_folders=True)


from IPython import embed; embed()
