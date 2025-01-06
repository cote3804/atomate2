from atomate2.jdftx.jobs.adsorption import SurfaceMinMaker
from pymatgen.core import Structure
from atomate2.jdftx.sets.core import SinglePointSetGenerator
from jobflow import run_locally
from pymatgen.core.surface import SlabGenerator



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

maker = SurfaceMinMaker(
    input_set_generator=generator,
    run_jdftx_kwargs={"jdftx_cmd": cmd}
)
slab_generator = SlabGenerator(structure, miller_index=(1,0,0), min_slab_size=1.0, min_vacuum_size=20, center_slab=True, in_unit_planes=True, lll_reduce=True, reorient_lattice=True)
slabs = slab_generator.get_slabs()
slab = slabs[0]
print(slab)
super_slab = slab.make_supercell([2,2,1])
print(super_slab)

# from IPython import embed
# embed() 


slab_job = maker.make(super_slab)

response = run_locally(slab_job, create_folders=True)
