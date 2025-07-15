from dataclasses import dataclass, field

from atomate2.ase.jobs import AseRelaxMaker
from atomate2.ase.md import AseMDMaker
from ase.calculators.calculator import Calculator
from emmet.core.vasp.calculation import StoreTrajectoryOption
from atomate2.ase.md import MDEnsemble
_ASE_DATA_OBJECTS = ["*.traj", "*.json.gz", "trajectory*"]

@dataclass
class MaceRelaxMaker(AseRelaxMaker):

#    model_path: str = "/pscratch/sd/s/soge8904/kestrel/MLIP/models_training/all_data_06_14_2025/seed_137/MACE_all_data_06_14_2025_stagetwo_compiled.model"
#    model_path: str = "/Users/sophi/DATA/IrO2/MLIP/all_data_06_14_2025/MACE_all_data_06_14_2025_stagetwo_compiled.model"
    model_path: str = "/scratch/soge8904/MLIP/models_training/all_data_06_14_2025/seed_137/MACE_all_data_06_14_2025_stagetwo_compiled.model"
    device: str = "cuda"
    relax_cell: bool = False
    ionic_step_data = None

    relax_kwargs: dict = field(default_factory=lambda: {
        "fmax": 0.01, 
        "traj_file": "mace_min.traj",
        "verbose": True 
    })

    optimizer_kwargs: dict = field(default_factory=lambda: {
        "optimizer": "BFGS"
    })
    store_trajectory: StoreTrajectoryOption = StoreTrajectoryOption.NO

    @property
    def calculator(self) -> Calculator:
        """MACE calculator."""
        try:
            from mace.calculators import MACECalculator
        except ImportError as e:
            raise ImportError(
                "MACE is required for MACE relaxation jobs. "
                "Please install MACE: pip install mace-torch"
            ) from e
        
        if not self.model_path:
            raise ValueError("model_path must be specified for MACE calculator")
        
        return MACECalculator(
            model_paths=self.model_path,
            device=self.device,
            **self.calculator_kwargs
        )
    
@dataclass
class MaceMDMaker(AseMDMaker):

    #model_path: str = "/pscratch/sd/s/soge8904/kestrel/MLIP/models_training/all_data_06_14_2025/seed_137/MACE_all_data_06_14_2025_stagetwo_compiled.model"
    model_path: str = "/scratch/soge8904/MLIP/models_training/all_data_06_14_2025/seed_137/MACE_all_data_06_14_2025_stagetwo_compiled.model"

    device: str = "cuda"
    store_trajectory: StoreTrajectoryOption = StoreTrajectoryOption.NO

    time_step: float = 1
    n_steps: int = 100
    ensemble: MDEnsemble = MDEnsemble.npt
    dynamics: str = "npt"
    temperature: float = 300.0
    pressure: float = 0.00101325 #kilobars?
    ionic_step_data: None = None
    verbose: bool = True
    # mask = [[1, 1, 0],
    #     [1, 1, 0],
    #     [0, 0, 1]]
    # ase_md_kwargs = {"ttime": 100*units.fs,
    #             "pfactor": 75*units.fs**2,
    #             "mask": mask}

    @property
    def calculator(self) -> Calculator:
        """MACE calculator."""
        try:
            from mace.calculators import MACECalculator
        except ImportError as e:
            raise ImportError(
                "MACE is required for MACE relaxation jobs. "
                "Please install MACE: pip install mace-torch"
            ) from e
        
        if not self.model_path:
            raise ValueError("model_path must be specified for MACE calculator")
        
        return MACECalculator(
            model_paths=self.model_path,
            device=self.device,
            **self.calculator_kwargs
        )

