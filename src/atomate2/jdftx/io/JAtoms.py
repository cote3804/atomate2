from ase import Atoms, Atom
from pymatgen.core.units import bohr_to_ang, Ha_to_eV
from dataclasses import dataclass
from typing import Optional, List
import numpy as np


class JEiter():
    '''
    Class object for storing logged electronic minimization data for a single SCF cycle
    '''
    iter_type: str = None
    etype: str = None
    #
    iter: int = None
    E: float = None
    grad_K: float = None
    alpha: float = None
    linmin: float = None
    t_s: float = None
    #
    mu: float = None
    nElectrons: float = None
    abs_magneticMoment: float = None
    tot_magneticMoment: float = None
    subspaceRotationAdjust: float = None
    #
    converged: bool = False
    converged_reason: str = None

    @classmethod
    def _from_lines_collect(cls, lines_collect: list[str], iter_type: str, etype: str):
        instance = cls()
        instance.iter_type = iter_type
        instance.etype = etype
        _iter_flag = f"{iter_type}: Iter: "
        for i, line_text in enumerate(lines_collect):
            if instance.is_line0(i, line_text):
                instance.read_line0(line_text, _iter_flag)
            elif instance.is_fillings_line(i, line_text):
                instance.read_fillings_line(line_text)
            elif instance.is_subspaceadjust_line(i, line_text):
                instance.read_subspaceadjust_line(line_text)
            elif instance.is_converged_line(i, line_text):
                instance.read_converged_line(line_text)
        return instance


    def is_line0(self, i: int, line_text: str) -> bool:
        is_line = i == 0
        return is_line
    

    def read_line0(self, line_text: str, _iter_flag: str) -> None:
        if not "Iter: " in _iter_flag:
            raise ValueError("_iter_flag must contain 'Iter: '")
        elif not _iter_flag in line_text:
            raise ValueError(f"Provided _iter_flag {_iter_flag} does not appear in provided line {line_text}")
        else:
            self.iter = self._get_colon_var_t1(line_text, "Iter: ")
            self.E = self._get_colon_var_t1(line_text, f"{self.etype}: ") * Ha_to_eV
            self.grad_K = self._get_colon_var_t1(line_text, "|grad|_K: ")
            self.alpha = self._get_colon_var_t1(line_text, "alpha: ")
            self.linmin = self._get_colon_var_t1(line_text, "linmin: ")
            self.t_s = self._get_colon_var_t1(line_text, "t[s]: ")
    

    def is_fillings_line(self, i: int, line_text: str) -> bool:
        is_line = "FillingsUpdate" in line_text
        return is_line
    

    def read_fillings_line(self, fillings_line: str) -> None:
        assert "FillingsUpdate:" in fillings_line
        self.set_mu(fillings_line)
        self.set_nElectrons(fillings_line)
        if "magneticMoment" in fillings_line:
            self.set_magdata(fillings_line)
    

    def is_subspaceadjust_line(self, i: int, line_text: str) -> bool:
        is_line = "SubspaceRotationAdjust" in line_text
        return is_line
    

    def read_subspaceadjust_line(self, line_text: str) -> None:
        self.subspaceRotationAdjust = self._get_colon_var_t1(line_text, "SubspaceRotationAdjust: set factor to")
    

    def is_converged_line(self, i: int, line_text: str) -> bool:
        is_line = f"{self.iter_type}: Converged" in line_text
        return is_line
    

    def read_converged_line(self, line_text: str) -> None:
        self.converged = True
        self.converged_reason = line_text.split("(")[1].split(")")[0]


    def set_magdata(self, fillings_line: str) -> None:
        _fillings_line = fillings_line.split("magneticMoment: [ ")[1].split(" ]")[0].strip()
        self.abs_magneticMoment = self._get_colon_var_t1(_fillings_line, "Abs: ")
        self.tot_magneticMoment = self._get_colon_var_t1(_fillings_line, "Tot: ")


    def _get_colon_var_t1(self, linetext: str, lkey: str) -> float | None:
        '''
        Reads a float from an elec minimization line assuming value appears as
        "... lkey value ..."
        '''
        colon_var = None
        if lkey in linetext:
            colon_var = float(linetext.split(lkey)[1].strip().split(" ")[0])
        return colon_var


    def set_mu(self, fillings_line: str) -> None:
        self.mu = self._get_colon_var_t1(fillings_line, "mu: ") * Ha_to_eV


    def set_nElectrons(self, fillings_line: str) -> None:
        self.nElectrons = self._get_colon_var_t1(fillings_line, "nElectrons: ")




class JEiters(list):
    '''
    Class object for collecting and storing a series of SCF cycles done between
    geometric optimization steps
    '''
    iter_type: str = None
    etype: str = None
    _iter_flag: str = None

    @classmethod
    def from_text_slice(cls, text_slice: list[str], iter_type: str = "ElecMinimize", etype: str = "F"):
        super().__init__([])
        instance = cls()
        instance._iter_flag = f"{iter_type}: Iter:"
        instance.iter_type = iter_type
        instance.etype = etype
        instance.parse_text_slice(text_slice)
        return instance

    # def __init__(self, text_slice: list[str], iter_type: str = "ElecMinimize", etype: str = "F"):
    #     super().__init__()
    #     self._iter_flag = f"{iter_type}: Iter:"
    #     self.iter_type = iter_type
    #     self.etype = etype
    #     self.parse_text_slice(text_slice)

    def parse_text_slice(self, text_slice: list[str]) -> None:
        lines_collect = []
        for line_text in text_slice:
            if len(line_text.strip()):
                if self._iter_flag in line_text:
                    if len(lines_collect):
                        self.append(JEiter._from_lines_collect(lines_collect, self.iter_type, self.etype))
                        lines_collect = []
                lines_collect.append(line_text)
            else:
                break
        if len(lines_collect):
            self.append(JEiter._from_lines_collect(lines_collect, self.iter_type, self.etype))
            lines_collect = []


    

@dataclass
class JAtoms(Atoms):
    '''
    A mutant of the ase Atoms class for flexiblity in holding JDFTx optimization data
    '''
    iter_type: str = None
    etype: str = None
    eiter_type: str = None
    emin_flag: str = None
    #
    Ecomponents: dict = None
    elecMinData: JEiters = None
    stress: np.ndarray = None
    strain: np.ndarray = None
    #
    iter: int = None
    E: float = None
    grad_K: float = None
    alpha: float = None
    linmin: float = None
    t_s: float = None
    #
    line_types = ["emin", "lattice", "strain", "stress", "posns", "forces", "ecomp", "lowdin", "opt"]

    @classmethod
    def from_text_slice(cls, text_slice: list[str],
                        eiter_type: str = "ElecMinimize",
                        iter_type: str = "IonicMinimize",
                        emin_flag: str = "---- Electronic minimization -------"):
        '''
        Create a JAtoms object from a slice of an out file's text corresponding
        to a single step of a native JDFTx optimization
        '''
        instance = cls()
        instance.eiter_type = eiter_type
        instance.iter_type = iter_type
        instance.emin_flag = emin_flag
        #
        line_collections = instance.init_line_collections()
        for i, line in enumerate(text_slice):
            read_line = False
            for line_type in line_collections:
                sdict = line_collections[line_type]
                if sdict["collecting"]:
                    lines, collecting, collected = instance.collect_generic_line(line, sdict["lines"])
                    sdict["lines"] = lines
                    sdict["collecting"] = collecting
                    sdict["collected"] = collected
                    read_line = True
                    break
            if not read_line:
                for line_type in line_collections:
                    if not line_collections[line_type]["collected"]:
                        if instance.is_generic_start_line(line, line_type):
                            line_collections[line_type]["collecting"] = True
                            line_collections[line_type]["lines"].append(line)
                            break

        # Ecomponents needs to be parsed before emin to set etype
        instance.parse_ecomp_lines(line_collections["ecomp"]["lines"])
        instance.parse_emin_lines(line_collections["emin"]["lines"])
        # Lattice must be parsed before posns/forces incase of direct coordinates
        instance.parse_lattice_lines(line_collections["lattice"]["lines"])
        instance.parse_posns_lines(line_collections["posns"]["lines"])
        instance.parse_forces_lines(line_collections["forces"]["lines"])
        # Strain and stress can be parsed in any order
        instance.parse_strain_lines(line_collections["strain"]["lines"])
        instance.parse_stress_lines(line_collections["stress"]["lines"])
        # Lowdin must be parsed after posns
        instance.parse_lowdin_lines(line_collections["lowdin"]["lines"])
        # Opt line must be parsed after ecomp
        instance.parse_opt_lines(line_collections["opt"]["lines"])

    def init_line_collections(self) -> dict:
        ''' #TODO: Move line_collections to be used as a class variable
        '''
        line_collections = {}
        for line_type in self.line_types:
            line_collections[line_type] = {"lines": [],
                                           "collecting": False,
                                           "collected": False}
        return line_collections
        
            
    def is_emin_start_line(self, line_text: str) -> bool:
        is_line = self.emin_flag in line_text
        return is_line
    

    def parse_emin_lines(self, emin_lines: list[str]) -> None:
        if len(emin_lines):
            self.elecMinData = JEiters(emin_lines, iter_type=self.eiter_type, etype=self.etype)


    def is_lattice_start_line(self, line_text: str) -> bool:
        is_line = "# Lattice vectors:" in line_text
        return is_line
    
    
    def parse_lattice_lines(self, lattice_lines: list[str]) -> None:
        R = None
        if len(lattice_lines):
            R = self._bracket_num_list_str_of_3x3_to_nparray(lattice_lines, i_start=2)
            R = R.T * bohr_to_ang
        self.set_cell(R)

    
    def is_strain_start_line(self, line_text: str) -> bool:
        is_line = "# Strain tensor in" in line_text
        return is_line
    
    
    def parse_strain_lines(self, strain_lines: list[str]) -> None:
        ST = None
        if len(strain_lines):
            ST = self._bracket_num_list_str_of_3x3_to_nparray(strain_lines, i_start=1)
            ST = ST.T * 1 # Conversion factor?
        self.strain = ST
    
    def is_stress_start_line(self, line_text: str) -> bool:
        is_line = "# Stress tensor in" in line_text
        return is_line
    

    def parse_stress_lines(self, stress_lines: list[str]) -> bool:
        ST = None
        if len(stress_lines):
            ST = self._bracket_num_list_str_of_3x3_to_nparray(stress_lines, i_start=1)
            ST = ST.T * 1 # Conversion factor?
        self.stress = ST
    
    
    def is_posns_start_line(self, line_text: str) -> bool:
        is_line = "# Ionic positions" in line_text
        return is_line
    
    
    def parse_posns_lines(self, posns_lines: list[str]) -> None:
        nAtoms = len(posns_lines) - 1
        coords_type = posns_lines[0].split("positions in")[1].strip().split()[0]
        posns = []
        names = []
        for i in range(nAtoms):
            line = posns_lines[i+1]
            name = line.split()[1].strip()
            posn = np.array([float(x.strip()) for x in line.split()[2:5]])
            names.append(name)
            posns.append(posn)
        posns = np.array(posns)
        if coords_type != "cartesian":
            posns = np.dot(posns, self.cell)
        else:
            posns *= bohr_to_ang
        for i in range(nAtoms):
            self.append(Atom(names[i], posns[i]))

    
    def is_forces_start_line(self, line_text: str) -> bool:
        is_line = "# Forces in" in line_text
        return is_line
    
    
    def parse_forces_lines(self, forces_lines: list[str]) -> None:
        nAtoms = len(forces_lines) - 1
        coords_type = forces_lines[0].split("Forces in")[1].strip().split()[0]
        forces = []
        for i in range(nAtoms):
            line = forces_lines[i+1]
            force = np.array([float(x.strip()) for x in line.split()[2:5]])
            forces.append(force)
        forces = np.array(forces)
        if coords_type != "cartesian":
            forces = np.dot(forces, self.cell) # TODO: Double check this conversion
            # (since self.cell is in Ang, and I need the forces in eV/ang, how
            # would you convert forces from direct coordinates into cartesian?)
        else:
            forces *= bohr_to_ang
        forces *= Ha_to_eV
        self.forces = forces

    
    def is_ecomp_start_line(self, line_text: str) -> bool:
        is_line = "# Energy components" in line_text
        return is_line
    
    
    def parse_ecomp_lines(self, ecomp_lines: list[str]) -> None:
        self.Ecomponents = {}
        for line in ecomp_lines:
            if " = " in line:
                lsplit = line.split(" = ")
                key = lsplit[0].strip()
                val = float(lsplit[1].strip())
                self.Ecomponents[key] = val
        if self.etype is None:
            self.etype = key


    
    def is_lowdin_start_line(self, line_text: str) -> bool:
        is_line = "#--- Lowdin population analysis ---" in line_text
        return is_line
    

    
    def parse_lowdin_lines(self, lowdin_lines: list[str]) -> None:
        charges_dict = {}
        moments_dict = {}
        for line in lowdin_lines:
            if self.is_charges_line(line):
                charges_dict = self.parse_lowdin_line(line, charges_dict)
            elif self.is_moments_line(line):
                moments_dict = self.parse_lowdin_line(line, moments_dict)
        names = self.get_chemical_symbols()
        charges = np.zeros(len(names))
        moments = np.zeros(len(names))
        for el in charges_dict:
            idcs = [i for i in range(len(names)) if names[i] == el]
            for i, idx in enumerate(idcs):
                charges[idx] += charges_dict[el][i]
                moments[idx] += moments_dict[el][i]
        self.charges = charges
        self.magnetic_moments = moments


    def is_charges_line(self, line_text: str) -> bool:
        is_line = "oxidation-state" in line_text
        return is_line
    
    
    def is_moments_line(self, line_text: str) -> bool:
        is_line = "magnetic-moments" in line_text
        return is_line
    

    def parse_lowdin_line(self, lowdin_line: str, lowdin_dict: dict[str, float]) -> dict[str, float]:
        tokens = [v.strip() for v in lowdin_line.strip().split()]
        name = tokens[2]
        vals = [float(x) for x in tokens[3:]]
        lowdin_dict[name] = vals
        return lowdin_dict
        
    
    def is_opt_start_line(self, line_text: str) -> bool:
        is_line = line_text.starts_with(f"{self.iter_type}: Iter")
        return is_line
    
    
    def parse_opt_lines(self, opt_lines: list[str]) -> None:
        opt_line = opt_lines[0]
        iter = int(self._get_colon_var_t1(opt_line, "Iter:"))
        self.iter = iter
        E = self._get_colon_var_t1(opt_line, self.etype)
        self.E = E * Ha_to_eV
        grad_K = self._get_colon_var_t1(opt_line, "|grad|_K: ")
        self.grad_K = grad_K
        alpha = self._get_colon_var_t1(opt_line, "alpha: ")
        self.alpha = alpha
        linmin = self._get_colon_var_t1(opt_line, "linmin: ")
        self.linmin = linmin
        t_s = self._get_colon_var_t1(opt_line, "t[s]: ")
        self.t_s = t_s


    
    def is_generic_start_line(self, line_text: str, line_type: str) -> bool:
        ''' I am choosing to map line_type to a function this way because
        I've had horrible experiences with storing functions in dictionaries
        in the past
        '''
        if line_type == "lowdin":
            return self.is_lowdin_start_line(line_text)
        elif line_type == "opt":
            return self.is_opt_start_line(line_text)
        elif line_type == "ecomp":
            return self.is_ecomp_start_line(line_text)
        elif line_type == "forces":
            return self.is_forces_start_line(line_text)
        elif line_type == "posns":
            return self.is_posns_start_line(line_text)
        elif line_type == "stress":
            return self.is_stress_start_line(line_text)
        elif line_type == "strain":
            return self.is_strain_start_line(line_text)
        elif line_type == "lattice":
            return self.is_lattice_start_line(line_text)
        elif line_type == "emin":
            return self.is_emin_start_line(line_text)
        else:
            raise ValueError(f"Unrecognized line type {line_type}")

    
    def collect_generic_line(self, line_text: str, generic_lines: list[str]) -> tuple[list[str], bool, bool]:
        collecting = True
        collected = False
        if not len(line_text.strip()):
            collecting = False
            collected = True
        else:
            generic_lines.append(line_text)
        return generic_lines, collecting, collected
    
    
    def _bracket_num_list_str_of_3_to_nparray(self, line: str) -> np.ndarray:
        return np.array([float(x) for x in line.split()[1:-1]])
    
    
    def _bracket_num_list_str_of_3x3_to_nparray(self, lines: list[str], i_start=0) -> np.ndarray:
        out = np.zeros([3,3])
        for i in range(3):
            out[i,:] += self._bracket_num_list_str_of_3_to_nparray(lines[i+i_start])
        return out
    
    def _get_colon_var_t1(self, linetext: str, lkey: str) -> float | None:
        '''
        Reads a float from an elec minimization line assuming value appears as
        "... lkey value ..."
        '''
        colon_var = None
        if lkey in linetext:
            colon_var = float(linetext.split(lkey)[1].strip().split(" ")[0])
        return colon_var

        



