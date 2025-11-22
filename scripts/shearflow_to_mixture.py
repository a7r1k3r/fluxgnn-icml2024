#!/usr/bin/env python3
"""
Convert multiple Shear Flow HDF5 files (with multiple Re/Sc combinatorics) to a single eager Mixture Solver format .pth file.

- All cases are processed and merged into a single python list of (mesh_data, dict_conditions, dict_answers)
- mesh_data (nodes, elements, x_coords, y_coords) is the same object (dict) for each sample (uniform grid, global mesh)
- No objects/class instances stored, only pure tensors/dicts for full compatibility
- The output file can be directly loaded by TorchFVDataset (non-lazy loader) for eagerloading training

Usage:
    python scripts/shearflow_to_mixture_eager.py \
        --input-dir /path/to/shear_h5/ \
        --output-file /output_path/mixture_eager_dataset.pth \
        [--Re 1e4 5e4] [--Sc 0.1 0.5] [--max-cases 10]
"""

import argparse
import pathlib
import re

import h5py
import numpy as np
import torch
import femio

def extract_Re_Sc_from_filename(filename, fallback_Re=1e4, fallback_Sc=1.0):
    match = re.search(r'Reynolds_(\d+\.?\d*(?:e[+-]?\d+)?)_Schmidt_(-?\d+\.?\d*(?:e[+-]?\d+)?)', filename)
    if match:
        Re = float(match.group(1))
        Sc = float(match.group(2))
        return Re, Sc
    print(f"[WARNING] Could not extract Re/Sc from filename {filename}, using fallback ({fallback_Re=}, {fallback_Sc=})")
    return fallback_Re, fallback_Sc

def generate_cartesian_fem_mesh(x_coords, y_coords):
    Nx, Ny = len(x_coords), len(y_coords)
    X, Y = np.meshgrid(x_coords, y_coords, indexing='ij')
    nodes = np.zeros((Nx*Ny, 3))
    nodes[:, 0] = X.ravel()
    nodes[:, 1] = Y.ravel()
    cells = []
    for i in range(Nx-1):
        for j in range(Ny-1):
            n0 = i*Ny + j
            n1 = (i+1)*Ny + j
            n2 = (i+1)*Ny + (j+1)
            n3 = i*Ny + (j+1)
            cells.append([n0, n1, n2, n3])
    cells = np.asarray(cells, dtype=np.int32)
    return {
        "nodes": nodes,         # [Nnode, 3]
        "elements": cells,      # [Ncell, 4]
        "x_coords": x_coords,
        "y_coords": y_coords,
    }

def node_to_cell(node_data: np.ndarray, Nx: int, Ny: int) -> np.ndarray:
    cell_data = 0.25 * (
        node_data[:-1, :-1] + node_data[1:, :-1] +
        node_data[1:, 1:] + node_data[:-1, 1:]
    )
    return cell_data.ravel()  # shape [Nc]

def process_one_h5(
    h5_file: str,
    mesh_dict: dict,
    fallback_Re: float = 1e4,
    fallback_Sc: float = 1.0,
    max_cases: int = None
):
    samples = []
    with h5py.File(h5_file, 'r') as f:
        tracer = f['t0_fields/tracer'][:]     # (n_cases, n_time, Nx, Ny)
        pressure = f['t0_fields/pressure'][:]
        velocity = f['t1_fields/velocity'][:]
        x_periodic_mask = f['boundary_conditions/x_periodic/mask'][:]
        y_periodic_mask = f['boundary_conditions/y_periodic/mask'][:]
        Nx, Ny = len(mesh_dict['x_coords']), len(mesh_dict['y_coords'])
        n_cases, n_time = tracer.shape[:2]

    Re, Sc = extract_Re_Sc_from_filename(str(h5_file), fallback_Re, fallback_Sc)
    nu = 1.0 / Re
    D = nu / Sc

    for icase in range(n_cases if max_cases is None else min(n_cases, max_cases)):
        Nc = (Nx-1) * (Ny-1)
        u_cells, p_cells, alpha_cells = [], [], []
        for t in range(n_time):
            u_x = node_to_cell(velocity[icase, t, ..., 0], Nx, Ny)
            u_y = node_to_cell(velocity[icase, t, ..., 1], Nx, Ny)
            p_t = node_to_cell(pressure[icase, t, ...], Nx, Ny)
            a_t = node_to_cell(tracer[icase, t, ...], Nx, Ny)
            u_cells.append(np.stack([u_x, u_y, np.zeros_like(u_x)], axis=1))  # (Nc,3)
            p_cells.append(p_t[:, None])
            alpha_cells.append(a_t[:, None])
        u_cells = torch.from_numpy(np.stack(u_cells)).float()
        p_cells = torch.from_numpy(np.stack(p_cells)).float()
        alpha_cells = torch.from_numpy(np.stack(alpha_cells)).float()

        initial = {
            'u': u_cells[0, ...].unsqueeze(-1),      # (Nc, 3, 1)
            'p': p_cells[0, ...].unsqueeze(-1),      # (Nc, 1, 1)
            'alpha': alpha_cells[0, ...].unsqueeze(-1)
        }
        prop = {
            'nu_solvent': torch.tensor([[nu]], dtype=torch.float32),
            'nu_solute': torch.tensor([[nu]], dtype=torch.float32),
            'rho_solvent': torch.tensor([[1.0]], dtype=torch.float32),
            'rho_solute': torch.tensor([[1.0]], dtype=torch.float32),
            'diffusion_alpha': torch.tensor([[D]], dtype=torch.float32),
            'gravity': torch.zeros(1, 3, dtype=torch.float32),
            'Reynolds': torch.tensor([[Re]], dtype=torch.float32),
            'Schmidt': torch.tensor([[Sc]], dtype=torch.float32),
        }
        answers = {
            'u': u_cells.unsqueeze(-1),        # [n_time, Nc, 3, 1]
            'p': p_cells.unsqueeze(-1),        # [n_time, Nc, 1, 1]
            'alpha': alpha_cells.unsqueeze(-1) # [n_time, Nc, 1, 1]
        }
        conditions = {
            'initial': initial,
            'property': prop,
            "periodic_mask_x": torch.from_numpy(x_periodic_mask),
            "periodic_mask_y": torch.from_numpy(y_periodic_mask),
        }
        metadata = {
            "icase": icase,
            "input_h5": str(h5_file),
            "nx": Nx, "ny": Ny,
            "n_cells": Nc,
            "Re": Re, "Sc": Sc,
            "nu": nu, "D": D,
        }
        # (mesh_data, dict_conditions, dict_answers)
        samples.append((mesh_dict, conditions, answers))
    return samples

def main():
    parser = argparse.ArgumentParser(
        description='Convert multiple Shear Flow HDF5 files to a single eager MixtureSolver .pth file (for eager loading)'
    )
    parser.add_argument('--input-dir', type=str, required=True, help='Input dir with .hdf5 files')
    parser.add_argument('--output-file', type=str, required=True, help='Output .pth file for all samples')
    parser.add_argument('--Re', type=float, nargs='*', default=None, help='Reynolds number(s) to filter (default: all)')
    parser.add_argument('--Sc', type=float, nargs='*', default=None, help='Schmidt number(s) to filter (default: all)')
    parser.add_argument('--max-cases', type=int, default=None, help='Maximum cases per file')
    args = parser.parse_args()

    input_dir = pathlib.Path(args.input_dir)
    h5_files = sorted(input_dir.glob('*.hdf5'))
    if args.Re is not None and len(args.Re)==0:
        args.Re = None
    if args.Sc is not None and len(args.Sc)==0:
        args.Sc = None

    samples = []
    mesh_built = False
    mesh_dict = None

    for h5_file in h5_files:
        # 过滤Re/Sc
        file_Re, file_Sc = extract_Re_Sc_from_filename(h5_file.name)
        if (args.Re and file_Re not in args.Re) or (args.Sc and file_Sc not in args.Sc):
            continue
        print(f"[{h5_file.name}] Re={file_Re}, Sc={file_Sc}")
        with h5py.File(h5_file, 'r') as f:
            x_coords, y_coords = f['dimensions/x'][:], f['dimensions/y'][:]
            if not mesh_built: 
                mesh_dict = generate_cartesian_fem_mesh(x_coords, y_coords)
                mesh_built = True
        new_samples = process_one_h5(
            str(h5_file),
            mesh_dict=mesh_dict,
            fallback_Re=file_Re, fallback_Sc=file_Sc,
            max_cases=args.max_cases
        )
        samples.extend(new_samples)

    if not samples:
        print("[ERROR] No samples matched filter or h5 files not found!")
        return
    print(f"[INFO] Total {len(samples)} samples processed (from {len(h5_files)} files)")
    print(f"[INFO] Saving all samples to {args.output_file}")
    torch.save(samples, args.output_file)
    print(f"Done.")

if __name__ == '__main__':
    main()