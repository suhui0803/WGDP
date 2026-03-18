import torch.nn as nn
import torch
import torch.nn.functional as F
from torch import Tensor

import inspect
from collections import OrderedDict
from typing import Optional, List, Callable

class GaussRBF(nn.Module):
    def __init__(self, label_features: int, output_channel_features: int):
        """
        Args:
            label_features: Tensor containing the number of type element pairs (scalar tensor).
            output_channel_features: Number of output features (Gaussian centers).
        """
        super().__init__()
        self.label_features = label_features
        self.output_channel_features = output_channel_features

        # Gaussian kernel parameters
        self.nuww = nn.Parameter(torch.Tensor(self.label_features))
        self.sigmas = nn.Parameter(torch.Tensor(self.label_features))
        self.centres = nn.Parameter(torch.Tensor(self.label_features, self.output_channel_features))

        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.constant_(self.nuww, -0.1)
        nn.init.constant_(self.sigmas, 4.0)
        nn.init.xavier_normal_(self.centres, gain=1.0)

    def forward(self, zij_label: torch.Tensor, rij: torch.Tensor):
        """
        Args:
            zij_label: (M,) integer labels, values in [0, self.label_features-1].
            rij: (M, input_channel_features) input features (e.g., distances).
        Returns:
            phi: (M, output_channel_features) Gaussian kernel output.
        """
        # Select the corresponding parameter type based on the label.
        ww = self.nuww[zij_label].unsqueeze(-1)   # (M, 1)
        sgm = self.sigmas[zij_label].unsqueeze(-1) # (M, 1)
        cc = self.centres[zij_label]   # (M, output_channel_features)

        # Sum along channel dimension to get scalar distance per sample
        dist = rij.sum(dim=1, keepdim=True)   # (M, 1)

        # Calculate Gaussian Kernel
        alpha = (dist - cc) * sgm   # (M, output_channel_features)
        phi = ww * torch.exp(-1*alpha.pow(2))    # (M, output_channel_features)

        return phi

    def __repr__(self):
        return f"GaussRBF(label_features={self.label_features}, output_channel_features={self.output_channel_features})"

class BesselRBF(nn.Module):
    """
    Bessel Radial Basis Function Embedding Module (using l=0 Bessel functions + optional Envelope function)
    Input:
        zij_label: (M,) label indices
        rij: (M, input_channel_features) distance features
    Output:
        phi: (M, output_channel_features)
    """
    def __init__(self, label_features: int, output_channel_features: int, cutoff: float):
        super().__init__()
        self.label_features = label_features
        self.output_channel_features = output_channel_features
        self.cutoff = cutoff

        # wave number k_n = nπ / cutoff
        k = torch.arange(1, output_channel_features + 1, dtype=torch.float32) * torch.pi / cutoff
        self.register_buffer("k", k)

        # Weights associated with labels
        self.nuww = nn.Parameter(torch.Tensor(label_features))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.constant_(self.nuww, -0.1)

    def forward(self, zij_label: torch.Tensor, rij: torch.Tensor):
         # 1. Sum along channel dimension to get scalar distance per sample
        dist = rij.sum(dim=1, keepdim=True)   # (M, 1)

        # 2. Calculation k_n * d
        kd = dist * self.k   # (M, output_channel_features)

        # 3. Spherical Bézier function j0(x) = sin(x)/x
        rbf = torch.sin(kd) / (kd + 1e-8)       # (M, output_channel_features)

        # 4. Label weighting + Envelope
        ww = self.nuww[zij_label].unsqueeze(-1)           # (M,1)
        phi = ww * rbf   # (M, output_channel_features)

        return phi

    def __repr__(self):
        return f"BesselRBF(label_features={self.label_features}, output_channel_features={self.output_channel_features})"

class LinearRBF(nn.Module):
    def __init__(self, label_features: int, input_channel_features: int, output_channel_features: int):
        """
        Args:
            label_features: Number of type element pairs.
            input_channel_features: Number of input channels for rij.
            output_channel_features: Number of output features.
        """
        super().__init__()
        self.label_features = label_features
        self.input_channel_features = input_channel_features
        self.output_channel_features = output_channel_features

        # Type-specific linear transformation parameters
        self.W = nn.Parameter(torch.Tensor(self.label_features, self.input_channel_features, self.output_channel_features))
        self.b = nn.Parameter(torch.Tensor(self.label_features, self.output_channel_features))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.W)
        nn.init.zeros_(self.b)

    def forward(self, zij_label: torch.Tensor, rij: torch.Tensor):
        """
        Args:
            zij_label: (M,) integer labels, values in [0, self.label_features-1].
            rij: (M, input_channel_features) input features.
        Returns:
            phi: (M, output_channel_features) linearly transformed features.
        """
        # Select type-specific weights and biases
        W = self.W[zij_label]          # (M, input_channel_features, output_channel_features)
        b = self.b[zij_label]          # (M, output_channel_features)

        # Linear transformation
        rij_trans = torch.einsum('mi,mio->mo', rij, W) + b  # (M, output_channel_features)

        # Optional activation (can be modified or removed)
        phi = torch.tanh(rij_trans)
        return phi

    def __repr__(self):
        return f"LinearRBF(label_features={self.label_features}, input_channel_features={self.input_channel_features}, output_channel_features={self.output_channel_features})"
    
def find_neighbors(coords: torch.Tensor, lattice: torch.Tensor, rc: torch.Tensor):
    """
    Find neighbors within a cutoff radius considering periodic boundary conditions.
    Handles non-orthogonal lattices and small unit cells (lattice vectors < rc).

    Args:
        coords (torch.Tensor): (N, 3) tensor of atomic coordinates.
        lattice (torch.Tensor): (3, 3) tensor with lattice vectors as rows.
        rc (torch.Tensor): Cutoff radius.

    Returns:
        pairs (torch.LongTensor): (M, 2) atom pair index [[i], [j]].
        shifts (torch.LongTensor): (M, 3) displacement vector [n1, n2, n3]
                                   such that r_j' = r_j + n1*a1 + n2*a2 + n3*a3.
    """
    lattice = lattice.detach()
    coords = coords.detach()
    lattice.requires_grad = False
    coords.requires_grad = False

    device = coords.device
    dtype = coords.dtype
    num_atoms = coords.shape[0]

    # Calculate the required expansion factor for each direction.
    recip_lattice = torch.linalg.inv(lattice).T
    recip_norm = torch.norm(recip_lattice, dim=1)
    rep = torch.ceil(rc * recip_norm).int()
    rep = torch.clamp(rep, min=1)

    # Generate integer indices for all displacements (using meshgrid)
    k_range = torch.arange(-rep[0], rep[0] + 1, device=device)
    l_range = torch.arange(-rep[1], rep[1] + 1, device=device)
    m_range = torch.arange(-rep[2], rep[2] + 1, device=device)
    kk, ll, mm = torch.meshgrid(k_range, l_range, m_range, indexing='ij')
    shifts_all = torch.stack([kk.reshape(-1), ll.reshape(-1), mm.reshape(-1)], dim=1)  # (P, 3)

    num_shifts = shifts_all.shape[0]

    # Convert integer displacement to a Cartesian coordinate displacement vector
    shift_vecs = torch.matmul(shifts_all.to(dtype), lattice)   # (num_shifts, 3)

    # Construct the coordinates of all mirrored atoms
    # image_coords: (num_shifts * num_atoms, 3)
    image_coords = coords.unsqueeze(0) + shift_vecs.unsqueeze(1)   # (num_shifts, num_atoms, 3)
    image_coords = image_coords.view(-1, 3)

    # Generate the corresponding index mapping
    j_indices = torch.arange(num_atoms, device=device).repeat(num_shifts)
    shift_values = shifts_all.repeat_interleave(num_atoms, dim=0)

    # Calculate the distance between the original atom and all mirror atoms.
    # dists: (num_atoms, num_shifts * num_atoms)
    dists = torch.cdist(coords, image_coords)

    # Pair selection with distances less than the cutoff radius
    mask = dists <= rc
    src_idx, flat_dst_idx = torch.nonzero(mask, as_tuple=True)

    # Decoding Target Index and Offset
    dst_idx = j_indices[flat_dst_idx]
    final_shifts = shift_values[flat_dst_idx]

    # Eliminate self-interactions (i == j and shift == (0,0,0))
    is_self = (src_idx == dst_idx) & (torch.all(final_shifts == 0, dim=1))
    valid_mask = ~is_self

    pairs = torch.stack([src_idx[valid_mask], dst_idx[valid_mask]], dim=1)
    shifts = final_shifts[valid_mask]

    return pairs, shifts


def find_neighbors_non_periodic(coords: torch.Tensor, rc: torch.Tensor):
    """
    Find neighbors for non-periodic configurations (no PBC) using GPU acceleration.
    Args:
        coords (torch.Tensor): (N, 3) tensor of atomic coordinates.
        rc (torch.Tensor): Cutoff radius.

    Returns:
        pairs (torch.LongTensor): (M, 2) tensor of atom pair indices [[i, j], ...]
    """
    coords = coords.detach()
    coords.requires_grad = False
    num_atoms = coords.shape[0]

    # Calculate the distance matrix
    dists = torch.cdist(coords, coords)  # (N, N)

    # Filter distances less than rc and exclude self-interactions
    mask = (dists <= rc) & (~torch.eye(num_atoms, dtype=torch.bool, device=coords.device))

    # Obtain valid neighbor pairs
    src_idx, dst_idx = torch.nonzero(mask, as_tuple=True)
    pairs = torch.stack([src_idx, dst_idx], dim=1)

    return pairs


def cutoff_cosine(distances: torch.Tensor, cutoff: torch.Tensor):
    # assuming all elements in distances are smaller than cutoff
    return 0.5 * torch.cos(distances * (torch.pi / cutoff)) + 0.5


class EmbedSequential(nn.Module):
    """
    A module that sequentially executes multiple accept operations (zij_label, rij) and returns rij.
    """
    def __init__(self, *layers):
        super().__init__()
        self.layers = nn.ModuleList(layers)

    def forward(self, zij_label, rij):
        for layer in self.layers:
            rij = layer(zij_label, rij)
        return rij

def make_embed_layers(label_features: int, out_channels_list: list,
                      module_name: str = 'GaussRBF', **kwargs):
    """
    Construct a sequence of multi-layer RBF units of the specified type,
    inserting type-specific LayerNorm after each layer (handled by EmbedSequential).

    Args:
        label_features: Total number of feature types (scalar tensor).
        out_channels_list: List of output channels per layer.
        module_name: Name of the RBF module to use ('GaussRBF', 'BesselRBF', 'LinearRBF').
        **kwargs: Additional keyword arguments required by the specific module.
                  The three core arguments (label_features, input_channel_features,
                  output_channel_features) are automatically injected per layer.

    Returns:
        EmbedSequential container.
    """
    # Map module names to their constructors
    module_map = {
        'GaussRBF': GaussRBF,
        'BesselRBF': BesselRBF,
        'LinearRBF': LinearRBF,
    }

    if module_name not in module_map:
        raise ValueError(f"Unknown module name '{module_name}'. Available: {list(module_map.keys())}")

    module_class = module_map[module_name]

    layers = []
    current_in = 1

    for out_channels in out_channels_list:
        # Prepare arguments for the current layer
        # The three fixed arguments
        layer_kwargs = {
            'label_features': label_features,
            'input_channel_features': current_in,
            'output_channel_features': out_channels,
        }
        # Merge with user-provided kwargs (user args override if conflict, but label_features etc. are fixed)
        layer_kwargs.update(kwargs)

        # Get the signature of the module's __init__ (excluding self)
        sig = inspect.signature(module_class.__init__)
        # Filter layer_kwargs to only include parameters that the constructor accepts
        filtered_kwargs = {}
        for param_name, param in sig.parameters.items():
            if param_name == 'self':
                continue
            if param_name in layer_kwargs:
                filtered_kwargs[param_name] = layer_kwargs[param_name]
            elif param.default is inspect.Parameter.empty:
                # Required parameter missing
                raise TypeError(f"Missing required argument '{param_name}' for {module_name}")

        # Create layer
        layer = module_class(**filtered_kwargs)
        layers.append(layer)
        current_in = out_channels

    return EmbedSequential(*layers)

class Descriptor(nn.Module):
    def __init__(self, symbol_features: int, embedding_layers_list: list, Rcut: float):
        """
        symbol_features: Number of elements
        Rcut: cutoff radius
        """
        super().__init__()
        self.register_buffer('symbol_features', torch.tensor(symbol_features))
        label_features = symbol_features * symbol_features
        self.register_buffer('label_features', torch.tensor(label_features))
        self.register_buffer('Rcut', torch.tensor(Rcut))
        ZIJ_Label = torch.arange(0, label_features).reshape((symbol_features, symbol_features))
        self.register_buffer('ZIJ_Label', torch.as_tensor(ZIJ_Label).clone().detach())
        self.embedding_layers = make_embed_layers(label_features=label_features,out_channels_list=embedding_layers_list,module_name='GaussRBF',cutoff=self.Rcut)
        
    def forward(self, boxs: torch.Tensor, numbers: torch.Tensor, coords: torch.Tensor):
        # boxs [n_frames,9]
        # numbers [n_frames,n_atoms]
        # coord [n_frames,n_atoms*3]
        n_frames = coords.shape[0]
        n_atoms  = numbers.shape[1]
        boxs     = boxs.view(n_frames, 3, 3)
        coords   = coords.view(n_frames, n_atoms, 3)

        batch_dij = []  # Store the atomic dij matrix for each frame
        for frame_idx in range(n_frames):
            # Neighbor Search
            if torch.isnan(boxs[frame_idx]).any():
                all_pairs = find_neighbors_non_periodic(coords[frame_idx], self.Rcut)
                # Calculate relative vectors
                ri = coords[frame_idx][all_pairs[:, 0]]
                rj = coords[frame_idx][all_pairs[:, 1]]
                pair_vec_distances = rj - ri
            else:
                all_pairs, all_shifts = find_neighbors(coords[frame_idx], boxs[frame_idx], self.Rcut) 
                # Calculate relative vectors
                ri = coords[frame_idx][all_pairs[:, 0]]
                rj = coords[frame_idx][all_pairs[:, 1]]
                shift_cart = torch.matmul(all_shifts.to(boxs.dtype), boxs[frame_idx])
                pair_vec_distances = rj + shift_cart - ri

            # Atomic-pair type coding
            type_pairs = numbers[frame_idx][all_pairs]
            pair_zij_label = self.ZIJ_Label[type_pairs[:, 0], type_pairs[:, 1]]

            # Distance, Unit Vector, and Cutoff
            pair_rij_abs = torch.norm(pair_vec_distances, dim=1, keepdim=True)  # (M,1)
            pair_rij_unit_vec = pair_vec_distances / (pair_rij_abs + 1e-8)  # (M,3)
            pair_rij_cutoff = cutoff_cosine(pair_rij_abs, self.Rcut) # (M,1)

            # Construct input features (distance + unit vector) and multiply by the cutoff value.
            pair_rij_features = torch.cat([pair_rij_cutoff, pair_rij_unit_vec * pair_rij_cutoff], dim=-1)  # (M, 4)

            # Obtain the vector for each pair of neighbors through the embedding layer.
            pair_gij = self.embedding_layers(pair_zij_label, pair_rij_cutoff)  # (M, D)

            D = pair_gij.size(1)

            # Initialize the accumulator for each atom
            left_sum = torch.zeros(n_atoms, D, 4, device=pair_gij.device, dtype=pair_gij.dtype)   # (n_atoms, D, 4)
            right_sum = torch.zeros(n_atoms, 4, D, device=pair_gij.device, dtype=pair_gij.dtype)  # (n_atoms, 4, D)

            # Compute the outer product contribution for each neighbor
            left_contrib = pair_gij.unsqueeze(2) * pair_rij_features.unsqueeze(1)  # (M, D, 4)
            right_contrib = pair_rij_features.unsqueeze(2) * pair_gij.unsqueeze(1)  # (M, 4, D)

            # Accumulate by atomic index (this is equivalent to selecting the corresponding neighbor contribution for each atom)
            left_sum.scatter_add_(0, all_pairs[:, 0].unsqueeze(-1).unsqueeze(-1).expand(-1, D, 4), left_contrib)
            right_sum.scatter_add_(0, all_pairs[:, 0].unsqueeze(-1).unsqueeze(-1).expand(-1, 4, D), right_contrib)

            # Calculate dij for each atom dij = left_sum @ right_sum
            dij_atoms = torch.matmul(left_sum, right_sum)  # (n_atoms, D, D)

            # Frobenius Norm normalization (avoiding division by zero)
            norm = torch.norm(dij_atoms, dim=(1,2), p='fro')  # (n_atoms,)
            norm = torch.where(norm > 0, norm, torch.ones_like(norm))
            dij_atoms = dij_atoms / norm.unsqueeze(-1).unsqueeze(-1) # (n_atoms, D, D)
            dij_atoms = dij_atoms.reshape(n_atoms,-1)

            batch_dij.append(dij_atoms)  #(n_atoms, D*D)

        # Stack the results of all frames
        return torch.stack(batch_dij, dim=0)  # (n_frames, n_atoms, D*D)


class NormLayer(nn.Module):
    def __init__(self, norm_type, num_features, eps=1e-6, momentum=0.1, affine=True,
                 track_running_stats=True, elementwise_affine=True):
        super().__init__()
        self.norm_type = norm_type.lower()
        if self.norm_type == 'batch':
            self.norm = nn.BatchNorm1d(num_features=num_features, eps=eps, momentum=momentum,
                                       affine=affine, track_running_stats=track_running_stats)
        elif self.norm_type == 'layer':
            self.norm = nn.LayerNorm(normalized_shape=num_features, eps=eps,
                                     elementwise_affine=elementwise_affine)
        else:
            raise ValueError(
                f"Unsupported normalization type: {self.norm_type}")

    def forward(self, x: torch.Tensor):
        return self.norm(x)


class ResidualFCBlock(nn.Module):
    def __init__(self, input_size:int, hidden_size:int, lnorm: bool, bias: bool, norm_type='layer'):
        super().__init__()
        if lnorm == True:
            self.base_layer1 = nn.Sequential(nn.Linear(in_features=input_size, out_features=hidden_size, bias=bias),
                                             NormLayer(norm_type=norm_type, num_features=hidden_size))
            self.base_layer2 = nn.Sequential(nn.Linear(in_features=hidden_size, out_features=hidden_size, bias=bias),
                                             NormLayer(norm_type=norm_type, num_features=hidden_size))
        else:
            self.base_layer1 = nn.Sequential(
                nn.Linear(in_features=input_size, out_features=hidden_size, bias=bias))
            self.base_layer2 = nn.Sequential(
                nn.Linear(in_features=hidden_size, out_features=hidden_size, bias=bias))
        self.activation = nn.Tanh()
        if input_size != hidden_size:
            if lnorm == True:
                self.shortcut = nn.Sequential(nn.Linear(in_features=input_size, out_features=hidden_size, bias=bias),
                                              NormLayer(norm_type=norm_type, num_features=hidden_size))
            else:
                self.shortcut = nn.Sequential(
                    nn.Linear(in_features=input_size, out_features=hidden_size, bias=bias))
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor):
        residual = x
        x = self.base_layer1(x)
        x = self.activation(x)
        x = self.base_layer2(x)

        x += self.shortcut(residual)
        x = self.activation(x)
        return x


def Make_ResidualFCBlock(cfg: list, input_size: int, ouput_size: int, lnorm: bool, bias: bool, norm_type='layer'):
    layers = []
    input_features = input_size
    if lnorm == True:
        layers.append(nn.Sequential(nn.Linear(in_features=input_features, out_features=cfg[0], bias=bias),
                                    NormLayer(norm_type=norm_type,
                                              num_features=cfg[0]),
                                    nn.Tanh()))
        input_features = cfg[0]
    else:
        layers.append(nn.Sequential(nn.Linear(in_features=input_features, out_features=cfg[0], bias=bias),
                                    nn.Tanh()))
        input_features = cfg[0]
    for v in cfg:
        layers.append(ResidualFCBlock(input_size=input_features,
                      hidden_size=v, lnorm=lnorm, bias=bias, norm_type=norm_type))
        input_features = v
    layers += [nn.Linear(cfg[-1], ouput_size, bias=bias)]
    return nn.Sequential(*layers)

class DPNET(nn.Module):
    def __init__(self, symbol_features: int, embedding_layers: list, Rcut: float, fitting_layers: list, lnorm: bool, norm_type: str, bias: bool, initialize_weights=True):
        super().__init__()
        self.symbol_features = symbol_features
        self.embedding_layers_list = embedding_layers
        self.Rcut = Rcut
        self.fitting_layers_list = fitting_layers
        self.lnorm = lnorm
        self.norm_type = norm_type
        self.bias = bias
        self.descriptor = Descriptor(symbol_features=self.symbol_features,embedding_layers_list=self.embedding_layers_list, Rcut=self.Rcut)
        self.fc_input_features = self.embedding_layers_list[-1] * self.embedding_layers_list[-1]
        self.fitting_layer = Make_ResidualFCBlock(self.fitting_layers_list, self.fc_input_features, 1, lnorm=self.lnorm, bias=self.bias, norm_type=self.norm_type)
        if initialize_weights == True:
            self._initialize_weights()

    def forward(self, boxs: torch.Tensor, numbers: torch.Tensor, coords: torch.Tensor):
        # boxs [n_frames,3*3]
        # numbers [n_frames,n_atoms]
        # coords[n_frames,n_atomsx3]
        x = self.descriptor(boxs, numbers, coords)
        bs, natoms, mm = x.shape
        x = x.reshape(bs*natoms, mm)
        # N = bs*natoms

        atom_energy = self.fitting_layer(x)  # [N]
        frame_energy = atom_energy.reshape(bs, -1)  # [bs,natoms]
        energy = torch.sum(frame_energy, dim=-1, keepdim=True)  # [bs,1]
        return energy

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0.00, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
