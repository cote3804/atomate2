from atomate2.jdftx.jobs.adsorption import MolMinMaker
from atomate2.jdftx.sets.core import SinglePointSetGenerator
from pymatgen.core import Molecule
from jobflow import job, run_locally

cmd = "srun -n 1 --gpu-bind=single:1 /global/cfs/cdirs/m4025/Software/Perlmutter/JDFTx/build-gpu/jdftx_gpu"


species = ["O", "H", "H"]
coordinates = [
    [0.0, 0.0, 0.0],    # Oxygen at origin
    [0.96, 0.0, 0.0],   # First hydrogen (bonded to O)
    [-0.26, 0.92, 0.0]  # Second hydrogen (bonded to O, ~104.5° angle)
]

# Create the Molecule object
molecule = Molecule(species, coordinates)

molecule = molecule.get_boxed_structure(10, 10, 10)
print(molecule)

generator = SinglePointSetGenerator()

maker = MolMinMaker(input_set_generator=generator,
                    run_jdftx_kwargs={"jdftx_cmd": cmd})

job_mol = maker.make(molecule)

response = run_locally(job_mol, create_folders=True)