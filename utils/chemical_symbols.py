
chemical_symbols = [
# 0
'X',
# 1
'H', 'He',
# 2
'Li', 'Be', 'B', 'C', 'N', 'O', 'F', 'Ne',
# 3
'Na', 'Mg', 'Al', 'Si', 'P', 'S', 'Cl', 'Ar',
# 4
'K', 'Ca', 'Sc', 'Ti', 'V', 'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
'Ga', 'Ge', 'As', 'Se', 'Br', 'Kr',
# 5
'Rb', 'Sr', 'Y', 'Zr', 'Nb', 'Mo', 'Tc', 'Ru', 'Rh', 'Pd', 'Ag', 'Cd',
'In', 'Sn', 'Sb', 'Te', 'I', 'Xe',
# 6
'Cs', 'Ba', 'La', 'Ce', 'Pr', 'Nd', 'Pm', 'Sm', 'Eu', 'Gd', 'Tb', 'Dy',
'Ho', 'Er', 'Tm', 'Yb', 'Lu',
'Hf', 'Ta', 'W', 'Re', 'Os', 'Ir', 'Pt', 'Au', 'Hg', 'Tl', 'Pb', 'Bi',
'Po', 'At', 'Rn',
# 7
'Fr', 'Ra', 'Ac', 'Th', 'Pa', 'U', 'Np', 'Pu', 'Am', 'Cm', 'Bk',
'Cf', 'Es', 'Fm', 'Md', 'No', 'Lr',
'Rf', 'Db', 'Sg', 'Bh', 'Hs', 'Mt', 'Ds', 'Rg', 'Cn', 'Nh', 'Fl', 'Mc',
'Lv', 'Ts', 'Og']

def to_atomic_numbers(symbols):
    atomic_numbers = {}
    for Z, symbol in enumerate(chemical_symbols):
        atomic_numbers[symbol] = Z
    if isinstance(symbols,list):
        numbers = []
        for symbol in symbols:
            numbers.append(atomic_numbers[symbol])
        return numbers
    else:
        return atomic_numbers[symbols]

def to_atomic_symbols(numbers):
    if isinstance(numbers,list):
        symbols = []
        for i in numbers:
            symbols.append(chemical_symbols[i])
        return symbols
    else:
        symbol = chemical_symbols[numbers]
    return symbol