from atomate2.ase.jobs import AseRelaxMaker
from mace.calculators.mace import MACECalculator

class MACEMaker(AseRelaxMaker):
    name: str = "MACE-mh-1 OC20 Head Calculator"

    @property
    def calculator(self):
        return MACECalculator(
            model_paths="/anvil/projects/x-chm250094/cooper/huggingface/mace-mh-1.model",
            head="oc20_usemppbe",
        )