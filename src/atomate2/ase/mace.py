from dataclasses import dataclass, field

from atomate2.ase.jobs import AseRelaxMaker
from ase.calculators.calculator import Calculator
from emmet.core.vasp.calculation import StoreTrajectoryOption
from jobflow import job
import json
import os

_ASE_DATA_OBJECTS = ["*.traj", "*.json.gz", "trajectory*"]

@dataclass
class MaceRelaxMaker(AseRelaxMaker):

    relax_cell = False
    model_path: str = "/pscratch/sd/s/soge8904/kestrel/MLIP/models_training/all_data_06_14_2025/seed_137/MACE_all_data_06_14_2025_stagetwo_compiled.model"
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
    store_trajectory: StoreTrajectoryOption = StoreTrajectoryOption.FULL

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
    
    @job(data=_ASE_DATA_OBJECTS)
    def make(self, mol_or_struct, prev_dir=None):
        """Make job with output debugging."""
        print("=== MAKE METHOD STARTED ===")
        print(f"Working directory: {os.getcwd()}")
        print(f"Data patterns for GridFS: {['*.traj', '*.json.gz', '*.log']}")
        
        # Check files before
        files_before = set(os.listdir('.'))
        print(f"Files before calculation: {files_before}")
        
        # Run the ASE calculation  
        result = self.run_ase(mol_or_struct, prev_dir=prev_dir)
        
        # Check files after
        files_after = set(os.listdir('.'))
        new_files = files_after - files_before
        print(f"New files created: {new_files}")
        
        # Check file sizes
        for file in new_files:
            if os.path.isfile(file):
                size = os.path.getsize(file)
                print(f"  {file}: {size:,} bytes ({size/1024/1024:.2f} MB)")
        
        # Create task document
        from atomate2.ase.schemas import AseTaskDoc
        
        print("=== CREATING TASK DOCUMENT ===")
        task_doc = AseTaskDoc.to_mol_or_struct_metadata_doc(
            getattr(self.calculator, "name", type(self.calculator).__name__),
            result,
            self.steps,
            relax_kwargs=self.relax_kwargs,
            optimizer_kwargs=self.optimizer_kwargs,
            relax_cell=self.relax_cell,
            fix_symmetry=self.fix_symmetry,
            symprec=self.symprec if self.fix_symmetry else None,
            ionic_step_data=self.ionic_step_data,
            store_trajectory=self.store_trajectory,
            tags=self.tags,
        )
        
        print(f"Task document type: {type(task_doc)}")
        
        # Check task document size
        try:
            task_dict = task_doc.dict() if hasattr(task_doc, 'dict') else task_doc.__dict__
            task_json = json.dumps(task_dict, default=str)
            task_size = len(task_json.encode('utf-8'))
            print(f"TASK DOCUMENT SIZE: {task_size:,} bytes ({task_size/1024/1024:.2f} MB)")
            
            if task_size > 15*1024*1024:
                print("🚨 TASK DOCUMENT TOO LARGE FOR MONGODB!")
            
            # Find the largest fields in task document
            print("Task document field sizes:")
            large_fields = []
            for key, value in task_dict.items():
                try:
                    field_json = json.dumps(value, default=str)
                    field_size = len(field_json.encode('utf-8'))
                    if field_size > 100*1024:  # > 100KB
                        large_fields.append((key, field_size))
                        print(f"  {key}: {field_size:,} bytes ({field_size/1024/1024:.2f} MB)")
                except Exception as e:
                    print(f"  {key}: Cannot serialize - {e}")
            
            # Show the problematic fields
            if large_fields:
                print("🚨 LARGE FIELDS THAT SHOULD BE IN GRIDFS:")
                for key, size in sorted(large_fields, key=lambda x: x[1], reverse=True):
                    print(f"  {key}: {size/1024/1024:.2f} MB")
            
        except Exception as e:
            print(f"Cannot analyze task document: {e}")
        
        print("=== MAKE METHOD COMPLETED ===")
        return task_doc
