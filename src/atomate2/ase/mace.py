from dataclasses import dataclass, field

from atomate2.ase.jobs import AseRelaxMaker
from ase.calculators.calculator import Calculator

@dataclass
class MaceRelaxMaker(AseRelaxMaker):

    relax_cell = False
#    model_path: str = "/scratch/soge8904/MLIP/models_training/all_data_06_14_2025/seed_137/MACE_all_data_06_14_2025_stagetwo_compiled.model"
    model_path: str = "/Users/sophi/DATA/IrO2/MLIP/all_data_06_14_2025/MACE_all_data_06_14_2025_stagetwo_compiled.model"
    
    device: str = "cuda"

    relax_kwargs: dict = field(default_factory=lambda: {
        "fmax": 0.01, 
    })

    optimizer_kwargs: dict = field(default_factory=lambda: {
        "optimizer": "BFGS"
    })

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