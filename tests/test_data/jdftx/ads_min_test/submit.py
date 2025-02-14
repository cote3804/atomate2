from atomate2.jdftx.jobs.surface import SurfaceMinMaker
from pymatgen.core import Structure, Molecule
from atomate2.jdftx.sets.core import SinglePointSetGenerator
from jobflow import run_locally
from pymatgen.core.surface import SlabGenerator
from pymatgen.analysis.adsorption import AdsorbateSiteFinder



cmd = "srun -n 1 --gpu-bind=single:1 /global/cfs/cdirs/m4025/Software/Perlmutter/JDFTx/build-gpu/jdftx_gpu"

IrO2_structure = Structure.from_file("/pscratch/sd/s/soge8904/jobflow/IrO2/POSCAR_IrO2")

slab_generator = SlabGenerator(IrO2_structure, miller_index=(1,0,0), min_slab_size=3, min_vacuum_size=20, center_slab=True)
slabs = slab_generator.get_slabs()
slab = slabs[0]
super_slab = slab.make_supercell([2,2,1])

species = ["O", "H", "H"]
coordinates = [
    [0.0, 0.0, 0.0],    # Oxygen at origin
    [0.96, 0.0, 0.0],   # First hydrogen (bonded to O)
    [-0.26, 0.92, 0.0]  # Second hydrogen (bonded to O, ~104.5° angle)
]

# Create the Molecule object
molecule = Molecule(species, coordinates)

ads_slab_generator = AdsorbateSiteFinder(slab)

ads_coord = ads_slab_generator.find_adsorption_sites(positions="ontop")

# from IPython import embed
# embed() # Add this after line 28
coords = ads_coord["ontop"][0]

ads_slab = ads_slab_generator.add_adsorbate(molecule=molecule, ads_coord=coords)

generator = SinglePointSetGenerator()
generator.auto_kpoint_density = 100

maker = SurfaceMinMaker(
    input_set_generator=generator,
    run_jdftx_kwargs={"jdftx_cmd": cmd}
)

ads_slab_job = maker.make(ads_slab)

response = run_locally(ads_slab_job, create_folders=True)