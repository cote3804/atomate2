from pymatgen.core.structure import Structure, Site, Molecule
from pymatgen.analysis.local_env import CrystalNN
from scipy.spatial.transform import Rotation as R
import numpy as np
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.core.operations import SymmOp
from itertools import product
from jobflow import job

@job
def generate_adsorbed_structures(structure, adsorbate, supercell):
    undercoord_sites = find_undercoordinated_Ir_sites(structure)
    surface_sites = get_surface_sites(undercoord_sites)
    adsorption_sites = get_non_equivalent_sites(surface_sites, structure, supercell)    
    surface_sites_numpy = np.array([site.coords for site in adsorption_sites])
    for site_coord in surface_sites_numpy:
        slab = place_adsorbate(structure, adsorbate, site_coord)
        
    return slab

def find_undercoordinated_Ir_sites(
        structure: Structure,
        bulk_coord_num: int = 6, #this can be from the bulk structure
        tolerance: float = 0.1,
) -> list[Site]:
    
    nn_analyzer = CrystalNN(search_cutoff=5.0, distance_cutoffs=(0.5, tolerance))
    ir_sites = [site for site in structure.sites if site.species_string == "Ir"]
    undercoord_sites = []

    for site in ir_sites:
        nn_info = nn_analyzer.get_nn_info(structure, structure.sites.index(site))
        o_neighbors = sum(1 for nn in nn_info if nn["site"].species_string == "O")

        if o_neighbors < bulk_coord_num:
            undercoord_sites.append(site)

    return undercoord_sites

def get_surface_sites(
        sites: list[Site],
        height: float = 0.5,
) -> list[Site]:
    
    max_z = max(site.coords[2] for site in sites)

    surface_sites = [
        site for site in sites if max_z - site.coords[2] < height
    ]

    return surface_sites

def get_non_equivalent_sites(
    sites: list[Site],
    structure: Structure,
    supercell_dims: list[int],
    tolerance: float = 0.1
) -> list[Site]:
    lattice = structure.lattice
    frac_tolerance = tolerance / np.linalg.norm(lattice.matrix[0])  # Approximate conversion to fractional
    sym_analyzer = SpacegroupAnalyzer(structure, symprec=tolerance)
    sym_ops = sym_analyzer.get_symmetry_operations()

    non_equiv_sites = []
    processed_frac_coords = []
    
    for site in sites:
        frac_coords = site.frac_coords
        is_equivalent = False
        
        for ref_coords in processed_frac_coords:
            # Check all possible translations within the supercell
            for shifts in product(*[range(dim) for dim in supercell_dims]):
                # Apply translation in fractional coordinates
                shift_vector = [shift/dim for shift, dim in zip(shifts, supercell_dims)]

                for op in sym_ops:
                    transformed = op.operate(ref_coords)
                    translated = np.array([
                        (transformed[i] + shift_vector[i]) % 1.0 
                        for i in range(3)
                    ])
                    
                    # Calculate periodic distance in fractional coordinates
                    diff = np.array([
                        min((frac_coords[i] - translated[i]) % 1.0, 
                            (translated[i] - frac_coords[i]) % 1.0)
                        for i in range(3)
                    ])
                    
                    # Convert to cartesian distance
                    dist = np.linalg.norm(np.dot(diff, lattice.matrix))
                    
                    if dist < tolerance:
                        is_equivalent = True
                        break
                
                if is_equivalent:
                    break
            
            if is_equivalent:
                break
        
        if not is_equivalent:
            non_equiv_sites.append(site)
            processed_frac_coords.append(frac_coords)
    
    return non_equiv_sites

def place_adsorbate(slab: Structure, adsorbate: Molecule, site_coords, height=2.0):

    binding_index = get_binding_atom_index(adsorbate)
    ads_mol = center_on_atom(adsorbate, binding_index)

        # Only perform alignment and orientation if we have enough atoms
    num_atoms = len(ads_mol)
    
    if num_atoms >= 3:
        # For triatomic or larger molecules, full alignment and orientation
        ads_mol = align_adsorbate(ads_mol, slab)
        ads_mol = orient_adsorbate(ads_mol, binding_index)
    elif num_atoms == 2:
        # For diatomic, perform basic orientation to point away from surface
        ads_mol = orient_diatomic(ads_mol, binding_index)
    # For monoatomic, no alignment/orientation needed as it's just a point
    
    # Calculate the position to place the adsorbate
    placement_site = site_coords.copy()
    placement_site[2] += height  # Add height to z coordinate
    
    # Add the adsorbate atoms to the slab
    for site in ads_mol.sites:
        new_coord = site.coords + placement_site
        
        # Get all properties from the original adsorbate site
        # This preserves things like 'magmom', 'charges', etc.
        site_props = {}
        if hasattr(site, 'properties') and site.properties:
            site_props = site.properties.copy()
        print(site_props)
        # Add any required properties from slab that might be missing in the adsorbate
        if "surface_properties" in slab.site_properties and "surface_properties" not in site_props:
            site_props["surface_properties"] = "adsorbate"
            
        if "selective_dynamics" in slab.site_properties and "selective_dynamics" not in site_props:
            site_props["selective_dynamics"] = [True, True, True]
            
        # Note: We don't override 'charges' if it already exists in site_props
        if "charges" in slab.site_properties and "charges" not in site_props:
            site_props["charges"] = 0.0

        if "magmom" in slab.site_properties and "magmom" not in site_props:
            site_props["magmom"] = 0.0
        
        # Append the site with all its properties preserved
        slab.append(
            species=site.specie, 
            coords=new_coord, 
            coords_are_cartesian=True, 
            properties=site_props
        )
    
    return slab

def get_binding_atom_index(molecule):
    """
    Determine which atom in the adsorbate should bind to the surface
    """
    formula = molecule.composition.reduced_formula
    
    # Define which atom should bind based on adsorbate type
    if formula in ['H2O', 'OH', 'O']:
        for i, site in enumerate(molecule):
            if site.specie.symbol == 'O':
                return i
    elif formula == 'OOH':
        # For OOH, we want the O not bonded to H to bind
        o_indices = [i for i, site in enumerate(molecule) if site.specie.symbol == 'O']
        
        # Find which oxygen has H bonded to it
        for i in o_indices:
            has_h_neighbor = False
            for j, site in enumerate(molecule):
                if site.specie.symbol == 'H':
                    # Calculate distance to check bonding
                    dist = np.linalg.norm(molecule[i].coords - site.coords)
                    if dist < 1.1:  # Typical O-H bond length is ~0.96Å
                        has_h_neighbor = True
        
            if not has_h_neighbor:
                return i
            
        # For diatomic molecules, try to identify a binding atom based on common conventions
    if len(molecule) == 2:
        symbols = [site.specie.symbol for site in molecule]
        # Common diatomics: CO, NO, O2, etc. First atom usually binds
        if 'C' in symbols or 'N' in symbols:
            for i, symbol in enumerate(symbols):
                if symbol in ['C', 'N']:
                    return i
        # For O2, either atom can bind, so just return the first
        elif symbols.count('O') == 2:
            return 0
    return 0

def center_on_atom(molecule, atom_index):
    binding_coords = molecule[atom_index].coords
    translation = -binding_coords
    op = SymmOp.from_rotation_and_translation(translation_vec=translation)
    
    return molecule.apply_operation(op)
    
def align_adsorbate(molecule: Molecule, slab: Structure) -> Molecule:
    """
    Aligns the adsorbate molecule so that its plane (defined by three atoms) is perpendicular to the slab surface.

    Returns:
    - Molecule: The rotated adsorbate molecule.
    """
    if len(molecule) < 3:
        return molecule  
    
    # Step 1: Compute the slab normal (normal to the a-b plane)
    a, b, _ = slab.lattice.matrix  # Extract lattice vectors
    slab_normal = np.cross(a, b)
    slab_normal /= np.linalg.norm(slab_normal)  # Normalize
    # Step 2: Compute the adsorbate plane normal using three non-collinear atoms
    coords = np.array(molecule.cart_coords)
    r1, r2, r3 = coords[:3]  # Take the first three atoms
    # Check if the atoms are collinear
    v1 = r2 - r1
    v2 = r3 - r1
    cross_prod = np.cross(v1, v2)
    if np.linalg.norm(cross_prod) < 1e-6:  # If cross product is nearly zero, points are collinear
        # Try a different combination if possible
        if len(coords) > 3:
            r3 = coords[3]
            v2 = r3 - r1
            cross_prod = np.cross(v1, v2)
        
        # If still collinear or we don't have a 4th atom, return the original molecule
        if np.linalg.norm(cross_prod) < 1e-6:
            return molecule
    
    ads_normal = cross_prod
    ads_normal /= np.linalg.norm(ads_normal)  # Normalize
    
    # Step 3: Compute rotation to make adsorbate normal perpendicular to slab normal
    # Find a vector in the slab plane (e.g., a lattice vector 'a')
    target_vector = a / np.linalg.norm(a)  # Normalize to use as reference
    # Find rotation axis and angle to align adsorbate normal with target vector
    axis = np.cross(ads_normal, target_vector)
    if np.linalg.norm(axis) < 1e-6:
        return molecule  # Already aligned
    axis /= np.linalg.norm(axis)  # Normalize rotation axis
    angle = np.arccos(np.clip(np.dot(ads_normal, target_vector), -1.0, 1.0))  # Compute rotation angle
    
    # Step 4: Apply rotation to the molecule
    rotation = R.from_rotvec(angle * axis)  # Rodrigues rotation
    rotation_matrix = rotation.as_matrix()
    symm_op = SymmOp.from_rotation_and_translation(rotation_matrix=rotation_matrix, translation_vec=[0, 0, 0])

    return molecule.apply_operation(symm_op)

def orient_diatomic(molecule, binding_index):
    """
    Orient a diatomic molecule so that the non-binding atom points away from the surface (in +z direction).
    
    Args:
        molecule: The diatomic molecule
        binding_index: Index of the atom binding to the surface
        
    Returns:
        Oriented molecule
    """
    if len(molecule) != 2:
        return molecule  # Only works for diatomics
        
    # Get the index of the non-binding atom
    other_index = 1 if binding_index == 0 else 0
    
    # Get the vector from binding atom to the other atom
    binding_coords = molecule[binding_index].coords
    other_coords = molecule[other_index].coords
    vec = other_coords - binding_coords
    
    # Normalize the vector
    vec_norm = vec / np.linalg.norm(vec)
    
    # Target direction: +z axis
    z_axis = np.array([0, 0, 1])
    
    # Compute the rotation axis and angle
    rotation_axis = np.cross(vec_norm, z_axis)
    
    # Check if vectors are parallel or anti-parallel
    dot_product = np.dot(vec_norm, z_axis)
    
    if np.allclose(rotation_axis, 0):
        # Vectors are parallel or anti-parallel
        if dot_product > 0:
            # Already pointing in +z, no rotation needed
            return molecule
        else:
            # Pointing in -z, rotate 180 degrees around any perpendicular axis
            rotation_axis = np.array([1, 0, 0])  # Use x-axis
            rotation_angle = np.pi
    else:
        # Normalize rotation axis
        rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
        # Compute rotation angle
        rotation_angle = np.arccos(np.clip(dot_product, -1.0, 1.0))
    
    # Create rotation object
    rot = R.from_rotvec(rotation_angle * rotation_axis)
    
    # Apply rotation to all coordinates
    coords = molecule.cart_coords
    rotated_coords = rot.apply(coords)
    
    # Create new molecule with rotated coordinates
    rotated_molecule = Molecule(molecule.species, rotated_coords, 
                               site_properties=molecule.site_properties)
    
    return rotated_molecule

def orient_adsorbate(molecule, binding_index):
    # If it's a monoatomic adsorbate, no orientation needed
    if len(molecule) == 1:
        return molecule
        
    # If it's a diatomic, use the specialized function
    if len(molecule) == 2:
        return orient_diatomic(molecule, binding_index)
    # Get coordinates
    coords = molecule.cart_coords
    binding_coords = coords[binding_index]
    
    # Find the center of mass excluding the binding atom
    other_indices = [i for i in range(len(coords)) if i != binding_index]
    if not other_indices:
        return molecule  # Only one atom, no orientation needed
    
    # Calculate center of mass of non-binding atoms
    com = np.mean([coords[i] for i in other_indices], axis=0)
    
    # Vector from binding atom to center of mass
    vec = com - binding_coords
    
    # We want to rotate the molecule so that this vector points in the +z direction
    # This will ensure the binding atom has the lowest z-coordinate
    
    # Current direction of the vector
    vec_norm = vec / np.linalg.norm(vec)
    
    # Target direction: +z axis
    z_axis = np.array([0, 0, 1])
    
    # Compute the rotation axis and angle
    rotation_axis = np.cross(vec_norm, z_axis)
    
    # Check if vectors are parallel or anti-parallel
    dot_product = np.dot(vec_norm, z_axis)
    
    if np.allclose(rotation_axis, 0):
        # Vectors are parallel or anti-parallel
        if dot_product > 0:
            # Already pointing in +z, no rotation needed
            return molecule
        else:
            # Pointing in -z, rotate 180 degrees around any perpendicular axis
            rotation_axis = np.array([1, 0, 0])  # Use x-axis
            rotation_angle = np.pi
    else:
        # Normalize rotation axis
        rotation_axis = rotation_axis / np.linalg.norm(rotation_axis)
        # Compute rotation angle
        rotation_angle = np.arccos(np.clip(dot_product, -1.0, 1.0))
    
    # Create rotation object
    rot = R.from_rotvec(rotation_angle * rotation_axis)
    
    # Apply rotation to all coordinates
    rotated_coords = rot.apply(coords)
    
    # Create new molecule with rotated coordinates
    rotated_molecule = Molecule(molecule.species, rotated_coords, 
                               site_properties=molecule.site_properties)
    
    return rotated_molecule