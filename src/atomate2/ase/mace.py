from dataclasses import dataclass, field

from atomate2.ase.jobs import AseRelaxMaker
from ase.calculators.calculator import Calculator
from jobflow.core import job
import json

_ASE_DATA_OBJECTS = ["*.traj", "*.json.gz", "trajectory*"]

@dataclass
class MaceRelaxMaker(AseRelaxMaker):

    relax_cell = False
    model_path: str = "/scratch/soge8904/MLIP/models_training/all_data_06_14_2025/seed_137/MACE_all_data_06_14_2025_stagetwo_compiled.model"
#    model_path: str = "/Users/sophi/DATA/IrO2/MLIP/all_data_06_14_2025/MACE_all_data_06_14_2025_stagetwo_compiled.model"
    
    device: str = "cuda"

    relax_kwargs: dict = field(default_factory=lambda: {
        "fmax": 0.01, 
        "traj_file": "mace_min.traj",
        "verbose": True 
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
    
    def run_ase(self, mol_or_struct, prev_dir=None):
        """Override run_ase to debug the result."""
        print("=== RUNNING ASE CALCULATION ===")
        
        # Run the normal calculation
        result = super().run_ase(mol_or_struct, prev_dir)
        
        print("=== ASE CALCULATION COMPLETED ===")
        print(f"Result type: {type(result)}")
        print(f"Result attributes: {dir(result)}")
        
        # Check result object size
        try:
            # Try to serialize the result to see how big it is
            result_dict = result.__dict__ if hasattr(result, '__dict__') else str(result)
            result_json = json.dumps(result_dict, default=str)
            result_size = len(result_json.encode('utf-8'))
            print(f"ASE Result size: {result_size:,} bytes ({result_size/1024/1024:.2f} MB)")
            
            # Check individual fields
            if hasattr(result, '__dict__'):
                print("Result field sizes:")
                for key, value in result.__dict__.items():
                    try:
                        field_json = json.dumps(value, default=str)
                        field_size = len(field_json.encode('utf-8'))
                        if field_size > 10000:  # > 10KB
                            print(f"  {key}: {field_size:,} bytes")
                            if field_size > 1024*1024:  # > 1MB
                                print(f"    ^^^ LARGE FIELD! {field_size/1024/1024:.2f} MB")
                    except Exception as e:
                        print(f"  {key}: Cannot serialize - {e}")
            
        except Exception as e:
            print(f"Cannot analyze result size: {e}")
        
        return result