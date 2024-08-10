##############################################################################
# MDTraj: A Python Library for Loading, Saving, and Manipulating
#         Molecular Dynamics Trajectories.
# Copyright 2012-2024 Stanford University and the Authors
#
# Authors: Stefan Hervø-Hansen
# Contributors:
#
# MDTraj is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as
# published by the Free Software Foundation, either version 2.1
# of the License, or (at your option) any later version.
#
# This library is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public
# License along with MDTraj. If not, see <http://www.gnu.org/licenses/>.
##############################################################################

import numpy as np
from scipy.spatial import cKDTree

from mdtraj.geometry.distance import compute_distances
from mdtraj.utils import ensure_type

__all__ = ["compute_local_density"]

def compute_local_density(traj, solute_indices, grid_spacing=0.1, radius=1.0):
    """
    Compute the local density of solvent particles around the solute using a 3D grid.
    
    Parameters:
    traj : md.Trajectory
        The trajectory to analyze.
    solute_indices : array-like
        Indices of the solute atoms.
    grid_spacing : float, optional
        Spacing of the grid points in nanometers (default is 0.1 nm).
    radius : float, optional
        Radius of the spherical region around the solute to consider (default is 1.0 nm).
    
    Returns:
    local_density : ndarray
        Local density of solvent particles around the solute.
    """
    # Calculate the average number of solvent particles per frame
    n_frames = traj.n_frames
    n_solvent = len(set(range(traj.n_atoms)) - set(solute_indices))
    total_solvent_particles = np.zeros((n_frames,))
    
    # Create a grid covering the space around the solute
    grid_range = np.arange(-radius, radius + grid_spacing, grid_spacing)
    grid_points = np.array(np.meshgrid(grid_range, grid_range, grid_range)).T.reshape(-1, 3)
    
    # Initialize an array to store local density values
    local_density = np.zeros((len(grid_points),))

    for frame_idx in range(n_frames):
        # Get the positions of the solute and solvent particles
        solute_positions = traj[frame_idx].xyz[0][solute_indices]
        solvent_positions = traj[frame_idx].xyz[0][
            set(range(traj.n_atoms)) - set(solute_indices)
        ]
        
        # Build a KDTree for fast neighbor search
        tree = cKDTree(solvent_positions)
        
        # Compute local density for each grid point
        for i, grid_point in enumerate(grid_points):
            # Transform grid point to solute-centered coordinate system
            transformed_points = grid_point - solute_positions.mean(axis=0)
            
            # Find the number of solvent particles within the radius
            count = tree.query_ball_point(transformed_points, radius)
            local_density[i] = len(count) / (4/3 * np.pi * radius**3)
        
        # Update total solvent particles
        total_solvent_particles[frame_idx] = len(solvent_positions)

    # Average local density over all frames
    mean_local_density = np.mean(local_density, axis=0)
    
    # Compute average solvent density
    avg_density = np.sum(total_solvent_particles) / (4/3 * np.pi * radius**3 * n_frames)
    
    return mean_local_density / avg_density