import h5py
import os
import numpy as np
from functools import partial

# ---- Helper function for visititems ----
def collect_target_datasets(name, obj, targets, found):
    """
    Check if a dataset name contains one of the target substrings.
    Update the found dictionary in place.
    """
    for key, target in targets.items():
        if found[key] is None and target in name:
            found[key] = name

# ---- Main functions ----
def find_matching_keys(h5_file):
    """
    Traverse the HDF5 file and return the paths corresponding to target datasets.
    """
    targets = {
        "eiger_image": "scan_data/eiger_image",
        "position_x": "i11-c-c08__ex__tab-mt_tx.4/position",
        "position_z": "i11-c-c08__ex__tab-mt_tz.4/position",
        "basler_image": "i11-c-c08__dt__basler_analyzer/image"
    }
    found = {key: None for key in targets}

    # Use functools.partial to pass extra arguments without defining a nested function
    h5_file.visititems(partial(collect_target_datasets, targets=targets, found=found))

    return found["eiger_image"], found["position_x"], found["position_z"], found["basler_image"]


def extract_images_from_nxs(nxs_path):
    """
    Extract Eiger, Basler images and X/Z positions from a .nxs file.
    """
    with h5py.File(nxs_path, 'r') as f:
        k_eiger, k_x, k_z, k_basler = find_matching_keys(f)

        if k_eiger is None:
            raise KeyError(f"Eiger dataset not found in {nxs_path}")

        eiger_image = f[k_eiger][()]
        pos_x = f[k_x][()] if k_x else None
        pos_z = f[k_z][()] if k_z else None
        basler_image = f[k_basler][()] if k_basler else None

    position = np.stack((pos_x, pos_z), axis=-1) if pos_x is not None and pos_z is not None else None

    if eiger_image.ndim == 4:
        eiger_image = np.transpose(eiger_image, (0, 2, 3, 1))

    return eiger_image, position, basler_image


def extract_nxs_folder(folder_path, save_file):
    """
    Traverse a folder, read all .nxs files, return concatenated datasets,
    and save the Eiger data in the current working directory.
    """
    nxs_files = sorted(
        os.path.join(folder_path, f) 
        for f in os.listdir(folder_path) 
        if f.lower().endswith(".nxs")
    )

    eiger_data, position_data, basler_data = [], [], []

    for i, nxs_file in enumerate(nxs_files, start=1):
        eiger, pos, basler = extract_images_from_nxs(nxs_file)
        eiger_data.append(eiger)
        position_data.append(pos)
        basler_data.append(basler)
        print(f"{i}/{len(nxs_files)} files processed")

    eiger_data = np.array(eiger_data)
    position_data = np.array(position_data)
    basler_data = np.array(basler_data)

    
    # np.save("data_eiger.npy", eiger_data)
    np.save(save_file, eiger_data)
    
    print(f"Eiger data saved as {save_file} in {os.getcwd()}")

    return eiger_data, position_data, basler_data


#####################################
# def find_matching_keys(h5_file):
#     """
#     Traverse the HDF5 file and return the paths corresponding to target datasets.
#     """
#     # Target dataset keys we want to find
#     targets = {
#         "eiger_image": "scan_data/eiger_image",
#         "position_x": "i11-c-c08__ex__tab-mt_tx.4/position",
#         "position_z": "i11-c-c08__ex__tab-mt_tz.4/position",
#         "basler_image": "i11-c-c08__dt__basler_analyzer/image"
#     }

#     # Dictionary to store the found dataset paths
#     found = {key: None for key in targets}

#     def visitor(name, obj):
#         # For each dataset, check if its path contains one of the targets
#         for key, target in targets.items():
#             if found[key] is None and target in name:
#                 found[key] = name

#     h5_file.visititems(visitor)
#     return found["eiger_image"], found["position_x"], found["position_z"], found["basler_image"]


# def extract_images_from_nxs(nxs_path):
#     """
#     Extract Eiger, Basler images and X/Z positions from a .nxs file.
#     """
#     with h5py.File(nxs_path, 'r') as f:
#         k_eiger, k_x, k_z, k_basler = find_matching_keys(f)

#         if k_eiger is None:
#             raise KeyError(f"Eiger dataset not found in {nxs_path}")

#         eiger_image = f[k_eiger][()]
#         pos_x = f[k_x][()] if k_x else None
#         pos_z = f[k_z][()] if k_z else None
#         basler_image = f[k_basler][()] if k_basler else None

#     # Stack X/Z positions if available
#     position = np.stack((pos_x, pos_z), axis=-1) if pos_x is not None and pos_z is not None else None

#     # Permute dimensions for CNN if Eiger is 4D
#     if eiger_image.ndim == 4:
#         eiger_image = np.transpose(eiger_image, (0, 2, 3, 1))

#     return eiger_image, position, basler_image


# def extract_nxs_folder(folder_path):
#     """
#     Traverse a folder, read all .nxs files, and return concatenated datasets.
#     """
#     nxs_files = sorted(
#         os.path.join(folder_path, f) 
#         for f in os.listdir(folder_path) 
#         if f.lower().endswith(".nxs")
#     )

#     eiger_data, position_data, basler_data = [], [], []

#     for i, nxs_file in enumerate(nxs_files, start=1):
#         eiger, pos, basler = extract_images_from_nxs(nxs_file)
#         eiger_data.append(eiger)
#         position_data.append(pos)
#         basler_data.append(basler)
#         print(f"{i}/{len(nxs_files)} files processed")

#     # Convert lists to numpy arrays
#     eiger_data = np.array(eiger_data)
#     position_data = np.array(position_data)
#     basler_data = np.array(basler_data)

#     # Save Eiger data in the current working directory
#     np.save("data_eiger.npy", eiger_data)
#     print(f"Eiger data saved as 'data_eiger.npy' in {os.getcwd()}")
#     return np.array(eiger_data), np.array(position_data), np.array(basler_data)

############################################################################################

# import h5py
# import os
# import numpy as np

# def find_matching_keys(h5_file):
#     """
#     Parcourt le fichier HDF5 et retourne les chemins correspondant aux datasets cibles.
#     """
#     # Les clés cibles que l'on veut retrouver
#     targets = {
#         "eiger_image": "scan_data/eiger_image",
#         "position_x": "i11-c-c08__ex__tab-mt_tx.4/position",
#         "position_z": "i11-c-c08__ex__tab-mt_tz.4/position",
#         "basler_image": "i11-c-c08__dt__basler_analyzer/image"
#     }

#     # Dictionnaire pour stocker les chemins trouvés
#     found = {key: None for key in targets}

#     def visitor(name, obj):
#         # Pour chaque dataset, on regarde si son chemin contient une des cibles
#         for key, target in targets.items():
#             if found[key] is None and target in name:
#                 found[key] = name

#     h5_file.visititems(visitor)
#     return found["eiger_image"], found["position_x"], found["position_z"], found["basler_image"]


# def extraire_images_nxs(chemin_nxs):
#     """
#     Extrait les images Eiger, Basler et les positions X/Z depuis un fichier .nxs.
#     """
#     with h5py.File(chemin_nxs, 'r') as f:
#         k_eiger, k_x, k_z, k_basler = find_matching_keys(f)

#         if k_eiger is None:
#             raise KeyError(f"Pas trouvé le dataset Eiger dans {chemin_nxs}")

#         eiger_image = f[k_eiger][()]
#         pos_x = f[k_x][()] if k_x else None
#         pos_z = f[k_z][()] if k_z else None
#         basler_image = f[k_basler][()] if k_basler else None

#     # Concaténer X/Z si disponible
#     position = np.stack((pos_x, pos_z), axis=-1) if pos_x is not None and pos_z is not None else None

#     # Permuter pour CNN si Eiger est 4D
#     if eiger_image.ndim == 4:
#         eiger_image = np.transpose(eiger_image, (0, 2, 3, 1))

#     return eiger_image, position, basler_image


# def extraire_dossier_nxs(dossier_nxs):
#     """
#     Parcourt un dossier, lit tous les fichiers .nxs et retourne les données concaténées.
#     """
#     fichiers_nxs = sorted(
#         os.path.join(dossier_nxs, f) 
#         for f in os.listdir(dossier_nxs) 
#         if f.lower().endswith(".nxs")
#     )

#     data_eiger, data_pos, data_basler = [], [], []

#     for i, f_nxs in enumerate(fichiers_nxs, start=1):
#         eiger, pos, basler = extraire_images_nxs(f_nxs)
#         data_eiger.append(eiger)
#         data_pos.append(pos)
#         data_basler.append(basler)
#         print(f"{i}/{len(fichiers_nxs)} fichiers traités")

#     return np.array(data_eiger), np.array(data_pos), np.array(data_basler)
