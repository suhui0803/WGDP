import torch.nn as nn
import torch
import torch.nn.functional as F
from torch import Tensor

from collections import OrderedDict


class RConv(nn.Module):
    def __init__(self, label_features: Tensor, output_channel_features: int):
        """
        Gaussian kernel function
        label_features: zij Indicates the number of type element pairs
        centres_features: number of centers of a Gaussian function for every zij
        """
        super(RConv, self).__init__()
        self.label_features = label_features
        self.output_channel_features = output_channel_features
        self.nuww = nn.Parameter(torch.Tensor(self.label_features.item()))
        self.sigmas = nn.Parameter(torch.Tensor(self.label_features.item()))
        self.centres = nn.Parameter(torch.Tensor(
            self.label_features.item(), self.output_channel_features))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.constant_(self.nuww, -0.05)
        # nn.init.uniform_(self.nuww,a=-1.0,b=-0.1)
        nn.init.constant_(self.sigmas, 4.0)
        nn.init.xavier_normal_(self.centres, gain=1.0)

    def forward(self, zij_label: Tensor, rij: Tensor):
        """
        zij_label [length]
        rij [length,channels]
        out [length,output_channel_features]
        """
        rij = rij.sum(dim=1, keepdim=True)
        ww = self.nuww.index_select(dim=0, index=zij_label).unsqueeze(-1)
        sgm = self.sigmas.index_select(dim=0, index=zij_label).unsqueeze(-1)
        cc = self.centres.index_select(dim=0, index=zij_label)

        alpha = (rij - cc) * sgm
        phi = ww * torch.exp(-1*alpha.pow(2))
        return phi

    def __repr__(self):
        str = "RConv(label_features={}, output_channel_features={})".format(
            self.label_features, self.output_channel_features)
        return str


def find_neighbors(coords, lattice, rc, batch_size=32):
    """
    Args:
        coords (torch.Tensor): (N, 3) tensor of atomic coordinates.
        lattice (torch.Tensor): (3, 3) tensor with lattice vectors as rows.
        rc (float): Cutoff radius.
        batch_size (int): Number of shifts processed in each batch to balance speed and memory.

    Returns:
        (pairs, shifts): 
            pairs (torch.LongTensor): (M, 2) atom pair index [[i],[j]]
            shifts (torch.LongTensor): (M, 3) displacement vector [k,l,m]
    """
    lattice = lattice.detach()
    coords = coords.detach()
    lattice.requires_grad = False
    coords.requires_grad = False
    device = coords.device
    N = coords.shape[0]

    # Calculate basis vector lengths
    a, b, c = lattice[0], lattice[1], lattice[2]
    a_length = torch.norm(a)
    b_length = torch.norm(b)
    c_length = torch.norm(c)

    # Compute all possible diagonal vectors of the unit cell
    combinations = torch.tensor([
        [1, 1, 1], [1, 1, -1], [1, -1, 1], [-1, 1, 1],
        [1, -1, -1], [-1, 1, -1], [-1, -1, 1], [-1, -1, -1]
    ], dtype=lattice.dtype, device=device)
    diag_vectors = torch.matmul(combinations, lattice)
    diag_norms = torch.norm(diag_vectors, dim=1)
    unit_cell_diag = torch.max(diag_norms)

   # Calculate maximum shifts with safety margin
    k_max = torch.ceil((rc + unit_cell_diag) / a_length).int().item()
    l_max = torch.ceil((rc + unit_cell_diag) / b_length).int().item()
    m_max = torch.ceil((rc + unit_cell_diag) / c_length).int().item()

    # Generate all shift combinations
    k = torch.arange(-k_max, k_max + 1, device=device)
    l = torch.arange(-l_max, l_max + 1, device=device)
    m = torch.arange(-m_max, m_max + 1, device=device)
    shifts = torch.cartesian_prod(k, l, m)  # (M,3)

    # Filter shifts within rc + unit cell diagonal
    shift_norms = torch.norm(shifts.float() @ lattice, dim=1)
    valid_shifts = shifts[shift_norms <= (rc + unit_cell_diag)]
    valid_shifts = torch.unique(valid_shifts, dim=0)

    # Process shifts in batches
    pair_list = []
    shift_list = []
    num_shifts = valid_shifts.shape[0]
    for i in range(0, num_shifts, batch_size):
        batch_shifts = valid_shifts[i:i + batch_size]
        current_batch_size = batch_shifts.shape[0]

        # Compute shifted coordinates for the batch
        shifted_coords = coords.unsqueeze(
            0) + (batch_shifts.float() @ lattice).unsqueeze(1)  # (B, N, 3)

        # Compute distances between all atoms and shifted coordinates
        dists = torch.cdist(coords.unsqueeze(0).expand(
            current_batch_size, -1, -1), shifted_coords)  # (B, N, N)

        # Process each shift in the current batch
        for b in range(current_batch_size):
            shift = batch_shifts[b]
            dist_matrix = dists[b]

            # Apply cutoff and exclude self pairs for zero shift
            mask = dist_matrix <= rc
            if torch.all(shift == 0):
                mask &= ~torch.eye(N, dtype=torch.bool, device=device)

            # Collect indices and shifts
            rows, cols = torch.where(mask)
            if rows.numel() > 0:
                pair_list.append(torch.stack([rows, cols], dim=1))
                shift_list.append(batch_shifts[b].expand(rows.size(0), -1))

    # combined result
    if not pair_list:
        return torch.empty((0, 2), dtype=torch.long, device=device), \
            torch.empty((0, 3), dtype=torch.long, device=device)

    pairs_comb = torch.cat(pair_list, dim=0)
    shifts_comb = torch.cat(shift_list, dim=0)

    # Remove duplicate pairs elements
    combined = torch.cat([pairs_comb, shifts_comb], dim=1)
    unique_combined = torch.unique(combined, dim=0)

    pairs = unique_combined[:, :2].long().T
    shifts = unique_combined[:, 2:].long()

    return pairs, shifts


def find_neighbors_non_periodic(coords, rc):
    """
    Args:
        coords (torch.Tensor): (N, 3) tensor of atomic coordinates.
        rc (float): Cutoff radius.
    Returns:
        pairs (torch.LongTensor): (M, 2) atom pair index [[i],[j]]
    """
    coords = coords.detach()
    coords.requires_grad = False
    device = coords.device
    N = coords.shape[0]

    # calculate the full distance matrix
    dists = torch.cdist(coords, coords)  # (N, N)

    # generate off-diagonal masks (excluding the case of i=j)
    non_diag_mask = ~torch.eye(N, dtype=torch.bool, device=device)

    # combined screening criteria
    mask = (dists <= rc) & non_diag_mask

    # gets a valid atom pair
    rows, cols = torch.where(mask)
    pairs = torch.stack([rows, cols], dim=1)
    return pairs.T


def cutoff_cosine(distances: Tensor, cutoff: float):
    # assuming all elements in distances are smaller than cutoff
    return 0.5 * torch.cos(distances * (torch.pi / cutoff)) + 0.5


class EmbedSequential(nn.Module):
    def __init__(self, *args):
        super(EmbedSequential, self).__init__()
        if len(args) == 1 and isinstance(args[0], OrderedDict):
            for key, module in args[0].items():
                self.add_module(key, module)
        else:
            for idx, module in enumerate(args):
                self.add_module(str(idx), module)

    def forward(self, zij_label: Tensor, rij: Tensor):
        for module in self._modules.values():
            rij = module(zij_label, rij)
        return rij


def Make_EmbedLayers(embedding_layers_list: list, label_features: Tensor):
    layers = []
    for v in embedding_layers_list:
        rconv = RConv(label_features=label_features, output_channel_features=v)
        layers += [rconv]
    return EmbedSequential(*layers)


class Descriptor(nn.Module):
    def __init__(self, symbol_features: Tensor, embedding_layers: list, Rcut: Tensor,  device: None):
        """
        symbol_features: Number of elements
        Rcut: cutoff radius
        """
        super(Descriptor, self).__init__()
        self.symbol_features = symbol_features
        self.label_features = self.symbol_features * self.symbol_features
        self.embedding_layers_list = embedding_layers
        self.Rcut = Rcut
        self.ZIJ_Label = torch.arange(0, self.label_features, device=device).reshape(
            (self.symbol_features, self.symbol_features))
        self.embedding_layers = Make_EmbedLayers(
            embedding_layers_list=self.embedding_layers_list, label_features=self.label_features)

    def forward(self, boxs: Tensor, numbers: Tensor, coords: Tensor):
        # boxs [n_frames,9]
        # numbers [n_frames,n_atoms]
        # coord [n_frames,n_atoms*3]
        batch_dij = []
        for i_cr in range(coords.shape[0]):
            cell = boxs[i_cr, :].unsqueeze(0).reshape(3, 3)
            number = numbers[i_cr, :]
            coordinates = coords[i_cr, :].reshape(-1, 3)
            if torch.isnan(cell).any():
                all_pairs = find_neighbors_non_periodic(
                    coords=coordinates, rc=self.Rcut)
                pair_vec_distances = (coordinates.index_select(
                    dim=0, index=all_pairs[1]) - coordinates.index_select(dim=0, index=all_pairs[0]))
            else:
                all_pairs, all_shifts = find_neighbors(
                    coords=coordinates, lattice=cell, rc=self.Rcut)
                shift_values = all_shifts.to(cell.dtype) @ cell
                pair_vec_distances = (coordinates.index_select(dim=0, index=all_pairs[1]) + shift_values -
                                      coordinates.index_select(dim=0, index=all_pairs[0]))
            all_numbers = number.index_select(
                dim=0, index=all_pairs.reshape(-1)).reshape(all_pairs.size())
            all_zij_label = self.ZIJ_Label[all_numbers[0], all_numbers[1]]
            all_dij = []
            for i in range(coordinates.shape[0]):
                mask_label = (all_pairs[0] == i).unsqueeze(-1)
                rij = torch.masked_select(
                    pair_vec_distances, mask_label).view(-1, 3)
                zij_label = torch.masked_select(
                    all_zij_label.unsqueeze(-1), mask_label)
                rij_abs = rij.norm(2, -1)
                rij_angle = rij / rij_abs.unsqueeze(-1)

                rij_cos = cutoff_cosine(rij_abs, self.Rcut)
                rij_cos = rij_cos.unsqueeze(-1)  # [length,1]
                rij_cos_copy = rij_cos

                aij_cos = rij_angle * rij_cos  # [length,3]

                raij = torch.cat((rij_cos, aij_cos), dim=1)  # [length,4]

                # [length,output_channel_features]
                phi = self.embedding_layers(zij_label, rij_cos_copy)
                phi = phi.squeeze()  # [length,channels]

                raij_out_left = torch.matmul(phi.T, raij)
                raij_out_right = torch.matmul(raij.T, phi)
                raij_out = torch.matmul(
                    raij_out_left, raij_out_right)  # [M1,M2]
                raij_out = raij_out / torch.norm(raij_out, p='fro')

                dij = raij_out.reshape(shape=(1, -1))
                all_dij.append(dij.squeeze())

            all_Dij = torch.stack(all_dij, dim=0)
            batch_dij.append(all_Dij)
        Dij = torch.stack(batch_dij, dim=0)
        return Dij


class NormLayer(nn.Module):
    def __init__(self, norm_type, num_features, eps=1e-6, momentum=0.1, affine=True,
                 track_running_stats=True, elementwise_affine=True):
        super(NormLayer, self).__init__()
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

    def forward(self, x):
        return self.norm(x)


class ResidualFCBlock(nn.Module):
    def __init__(self, input_size, hidden_size, lnorm: bool, bias: bool, norm_type='layer'):
        super(ResidualFCBlock, self).__init__()
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

    def forward(self, x):
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

# model


class DPNET(nn.Module):
    def __init__(self, symbol_features: int, embedding_layers: list, Rcut: float, fitting_layers: list, lnorm: bool, norm_type: str, bias: bool, device: None, initialize_weights=True):
        super(DPNET, self).__init__()
        self.symbol_features = torch.tensor(symbol_features, device=device)
        self.embedding_layers_list = embedding_layers
        self.Rcut = torch.tensor(Rcut, device=device)
        self.fitting_layers_list = fitting_layers
        self.lnorm = lnorm
        self.norm_type = norm_type
        self.bias = bias
        self.descriptor = Descriptor(symbol_features=self.symbol_features,
                                     embedding_layers=self.embedding_layers_list, Rcut=self.Rcut, device=device)
        self.fc_input_features = self.embedding_layers_list[-1] * \
            self.embedding_layers_list[-1]
        self.fitting_layers = Make_ResidualFCBlock(
            self.fitting_layers_list, self.fc_input_features, 1, lnorm=self.lnorm, bias=self.bias, norm_type=self.norm_type)
        if initialize_weights == True:
            self._initialize_weights()

    def forward(self, boxs: Tensor, numbers: Tensor, coords: Tensor):
        # [n_frames,n_atomsx3]
        # [n_frames,n_atoms,mm_features]
        x = self.descriptor(boxs, numbers, coords)
        bs, natoms, mm = x.shape
        x = x.reshape(bs*natoms, mm)
        # N = bs*natoms

        atom_energy = self.fitting_layers(x)  # [N]
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
