from jobflow_remote.utils.examples import add
from jobflow_remote import submit_flow
from jobflow import Flow
from atomate2.jdftx.jobs.core import SinglePointMaker
from atomate2.jdftx.jobs.base import BaseJdftxMaker
from atomate2.jdftx.jobs.surface import MolMinMaker
from atomate2.jdftx.sets.base import JdftxInputGenerator
from pymatgen.core.structure import Structure, Molecule

coords = [
    [0.0, 0.0, 0.0],  # O
    [0.757, 0.586, 0.0],  # H
    [-0.757, 0.586, 0.0],  # H
]
species = ["O", "H", "H"]
water_mol = Molecule(species, coords)

mol_box = water_mol.get_boxed_structure(10, 10, 10)

cmd = "/global/cfs/cdirs/m4025/Software/Perlmutter/JDFTx/build-gpu/jdftx_gpu"

maker = MolMinMaker(
        run_jdftx_kwargs= {
        "jdftx_cmd": cmd, 
        "jdftx_job_kwargs":{"input_file":"init.in"},
        },
    task_document_kwargs = {},
    write_input_set_kwargs = {"infile": "init.in"}
)

job = maker.make(mol_box)

flow = Flow([job])

resources = {
    "nodes": 1, 
    "ntasks": 1, 
    # "gpus_per_job":1, 
    # "time_limit": 600, # seconds
    "account": "m4025_g",
    "constraint": "gpu",
    "cpus_per_task": 32,
    "time": "00:10:00",
    # "gpus_per_task": 1,
    "gres": "gpu:1",
    "partition": "debug"
    # "queue_name": "regular",
}

print(submit_flow(flow, worker="perlmutter", project="IrO2_soge", resources=resources))