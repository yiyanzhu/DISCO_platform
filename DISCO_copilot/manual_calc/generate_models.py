import os
from ase.build import fcc111, add_adsorbate
from ase.io import write

OUTPUT_DIR = "structures_to_upload"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

def build_models():
    print("Building Pt(111) + O models...")
    
    sites = ["top", "bridge", "fcc", "hcp"]
    
    for site in sites:
        slab = fcc111('Pt', size=(3, 3, 4), vacuum=10.0)
        
        ase_site = "ontop"
        if site == "bridge": ase_site = "bridge"
        elif site == "fcc": ase_site = "fcc"
        elif site == "hcp": ase_site = "hcp"
        
        add_adsorbate(slab, 'O', height=1.8, position=ase_site)
        
        filename = f"{OUTPUT_DIR}/Pt111_O_{site}.xyz"
        write(filename, slab)
        print(f"Generated: {filename}")

if __name__ == "__main__":
    build_models()
