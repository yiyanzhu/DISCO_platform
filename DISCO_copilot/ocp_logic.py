# 文件名: ocp_logic.py
import time
import numpy as np
from pymatgen.core import Structure
from pymatgen.ext.matproj import MPRester
from pymatgen.core.surface import SlabGenerator
from pymatgen.analysis.adsorption import AdsorbateSiteFinder
from pymatgen.io.ase import AseAtomsAdaptor
from ase.build import molecule

# ==========================================
# 🔑 这里填 Materials Project 的 Key
# (不是 DeepSeek 的 Key！)
# ==========================================
MP_API_KEY = "31HfDNN66lqSNhq4YH6zCxTQ2Re9t6cD"

class OCPLogic:
    """
    真实科学计算逻辑库 (基于 Pymatgen & Materials Project)
    """

    @staticmethod
    def smart_search_mp(api_key, query_str, limit=1):
        """连接 MP 数据库下载真实晶体结构"""
        # 如果调用时没传 key，就用上面定义的默认 key
        key = api_key if api_key else MP_API_KEY
        
        if key == "在这里填你的_Materials_Project_API_Key":
            return {}, "❌ 错误：请先在 ocp_logic.py 中填入 Materials Project API Key"

        print(f"   [底层库] 正在连接 Materials Project 下载: {query_str} ...")
        
        try:
            with MPRester(key) as mpr:
                # 搜索材料 (只找稳定的)
                docs = mpr.materials.summary.search(
                    formula=query_str, 
                    is_stable=True
                )
                
                if not docs:
                    return {}, f"未找到 {query_str} 的稳定结构"
                
                # 取第一个结果
                doc = docs[0]
                
                if isinstance(doc, dict):
                    struct = doc.get("structure")
                    mat_id = str(doc.get("material_id"))
                    formula = doc.get("formula_pretty")
                else:
                    struct = doc.structure
                    mat_id = str(doc.material_id)
                    formula = doc.formula_pretty
                
                return {
                    mat_id: {
                        "struct": struct,
                        "meta": {"formula": formula}
                    }
                }, f"成功下载 {formula} (ID: {mat_id})"
                
        except Exception as e:
            return {}, f"MP API 连接失败: {str(e)}"

    @staticmethod
    def generate_substitutions(parent_id, struct_obj, rules):
        """执行真实的原子替换，rules 格式: '0:Ni'"""
        print(f"   [底层库] 正在执行掺杂: {rules}")
        try:
            new_s = struct_obj.copy()
            rule_list = rules.split(";")
            info_list = []
            
            for r in rule_list:
                if ":" not in r: continue
                idx_str, species = r.split(":")
                idx = int(idx_str.strip())
                species = species.strip()
                
                if idx < len(new_s):
                    original_spec = new_s[idx].specie.symbol
                    new_s.replace(idx, species)
                    info_list.append(f"{original_spec}{idx}->{species}")
                
            new_id = f"{parent_id}_sub"
            return {
                new_id: {
                    "struct": new_s,
                    "meta": {"info": ", ".join(info_list)}
                }
            }, f"替换完成: {', '.join(info_list)}"
        except Exception as e:
            return {}, f"掺杂失败: {str(e)}"

    @staticmethod
    def generate_slabs(parent_id, struct_obj, h, k, l, min_size=10.0):
        """切真实的晶面"""
        print(f"   [底层库] 正在切面 ({h} {k} {l})...")
        try:
            slab_gen = SlabGenerator(
                struct_obj, 
                miller_index=(int(h), int(k), int(l)), 
                min_slab_size=min_size, 
                min_vacuum_size=15.0, 
                center_slab=True
            )
            slabs = slab_gen.get_slabs()
            
            if not slabs: return {}, "未生成有效的 Slab"
            
            slab = slabs[0] # 取第一个终端
            new_id = f"{parent_id}_slab_{h}{k}{l}"
            return {
                new_id: {
                    "struct": slab,
                    "meta": {"miller": [h, k, l]}
                }
            }, "切面生成成功 (Vacuum=15A)"
        except Exception as e:
            return {}, f"切面报错: {str(e)}"

    @staticmethod
    def generate_reaction_intermediates(parent_id, slab_obj, rxn_type, site_idx):
        """生成反应路径中间体"""
        print(f"   [底层库] 正在生成 {rxn_type} 吸附结构 (Site {site_idx})...")
        results = {}
        pathways = {
            "N2RR": ["N2", "N"], 
            "CO2RR": ["CO2", "CO"],
            "ORR": ["O2", "OH", "O"]
        }
        mols_to_add = pathways.get(rxn_type, ["CO"])
        
        try:
            target_atom = slab_obj[int(site_idx)]
            # 简单的 Ontop 判断
            z_coords = slab_obj.cart_coords[:, 2]
            z_center = (np.min(z_coords) + np.max(z_coords)) / 2
            direction = 1.0 if target_atom.coords[2] > z_center else -1.0
            ads_coords = target_atom.coords + np.array([0, 0, 2.0 * direction])
            
            for mol_name in mols_to_add:
                ase_mol = molecule(mol_name)
                pmg_mol = AseAtomsAdaptor.get_molecule(ase_mol)
                
                new_slab = slab_obj.copy()
                # 简单的平移吸附
                center = pmg_mol.center_of_mass
                pmg_mol.translate_sites(range(len(pmg_mol)), -center)
                pmg_mol.translate_sites(range(len(pmg_mol)), ads_coords)
                
                for site in pmg_mol:
                    new_slab.append(site.specie, site.coords, coords_are_cartesian=True)
                
                res_id = f"{parent_id}_{rxn_type}_{mol_name}"
                results[res_id] = {
                    "struct": new_slab,
                    "meta": {"mol": mol_name}
                }
            return results, f"成功生成 {len(results)} 个吸附结构"
        except Exception as e:
            return {}, f"吸附生成失败: {str(e)}"