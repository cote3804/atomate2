from pymatgen.core.structure import Structure, Site
from pymatgen.analysis.local_env import CrystalNN

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