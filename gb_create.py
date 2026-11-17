#!/usr/bin/env python
"""
Interactive Crystal & Grain Boundary Builder
A unified, interactive script to generate oriented Single Crystals or Grain Boundary structures.
Supports target physical dimensions (Angstroms), rigid body translations, overlap removal, and outputs to VASP, LAMMPS, or Extended XYZ.
"""

import sys
import numpy as np
from math import degrees, atan, sqrt, pi, ceil, cos, sin, gcd
from numpy import dot, cross
from numpy.linalg import det, norm, inv

# ==========================================
# Core Crystallography & Math Functions
# ==========================================

def get_cubic_sigma(uvw, m, n=1):
    u, v, w = uvw
    sqsum = u*u + v*v + w*w
    sigma = m*m + n*n * sqsum
    while sigma != 0 and sigma % 2 == 0:
        sigma /= 2
    return sigma if sigma > 1 else None

def get_cubic_theta(uvw, m, n=1):
    u, v, w = uvw
    sqsum = u*u + v*v + w*w
    if m > 0:
        return 2 * atan(sqrt(sqsum) * n / m)
    else:
        return pi

def get_theta_m_n_list(uvw, sigma):
    if sigma == 1:
        return [(0., 0., 0.)]
    thetas = []
    max_m = int(ceil(sqrt(4*sigma)))
    for m in range(1, max_m):
        for n in range(1, max_m):
            if gcd(m, n) == 1:
                s = get_cubic_sigma(uvw, m, n)
            if s == sigma:
                theta = (get_cubic_theta(uvw, m, n))
                thetas.append((theta, m, n))
                thetas.sort(key=lambda x: x[0])
    return thetas

def rot(a, Theta):
    c = cos(Theta)
    s = sin(Theta)
    a = a / norm(a)
    ax, ay, az = a
    return np.array([[c + ax * ax * (1 - c), ax * ay * (1 - c) - az * s, ax * az * (1 - c) + ay * s],
                     [ay * ax * (1 - c) + az * s, c + ay * ay * (1 - c), ay * az * (1 - c) - ax * s],
                     [az * ax * (1 - c) - ay * s, az * ay * (1 - c) + ax * s, c + az * az * (1 - c)]])

def integer_array(A, tol=1e-7):
    return np.all(abs(np.round(A) - A) < tol)

def ang(a, b):
    val = np.round(dot(a, b)/norm(a)/norm(b), 7)
    return abs(val)

def CommonDivisor(a):
    a = np.array(a)
    if np.all(a == 0):
        return (a.astype(int), 1)
    
    CommFac = []
    for i in range(2, 100):
        while (a[0] % i == 0 and a[1] % i == 0 and a[2] % i == 0):
            a = a / i
            CommFac.append(i)
            
    prod_val = abs(np.prod(CommFac)) if CommFac else 1
    return (a.astype(int), int(prod_val))

def SymmEquivalent(arr):
    Sym = np.zeros([24, 3, 3])
    Sym[0, :] = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]; Sym[1, :] = [[1, 0, 0], [0, -1, 0], [0, 0, -1]]
    Sym[2, :] = [[-1, 0, 0], [0, 1, 0], [0, 0, -1]]; Sym[3, :] = [[-1, 0, 0], [0, -1, 0], [0, 0, 1]]
    Sym[4, :] = [[0, -1, 0], [-1, 0, 0], [0, 0, -1]]; Sym[5, :] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
    Sym[6, :] = [[0, 1, 0], [-1, 0, 0], [0, 0, 1]]; Sym[7, :] = [[0, 1, 0], [1, 0, 0], [0, 0, -1]]
    Sym[8, :] = [[-1, 0, 0], [0, 0, -1], [0, -1, 0]]; Sym[9, :] = [[-1, 0, 0], [0, 0, 1], [0, 1, 0]]
    Sym[10, :] = [[1, 0, 0], [0, 0, -1], [0, 1, 0]]; Sym[11, :] = [[1, 0, 0], [0, 0, 1], [0, -1, 0]]
    Sym[12, :] = [[0, 1, 0], [0, 0, 1], [1, 0, 0]]; Sym[13, :] = [[0, 1, 0], [0, 0, -1], [-1, 0, 0]]
    Sym[14, :] = [[0, -1, 0], [0, 0, 1], [-1, 0, 0]]; Sym[15, :] = [[0, -1, 0], [0, 0, -1], [1, 0, 0]]
    Sym[16, :] = [[0, 0, 1], [1, 0, 0], [0, 1, 0]]; Sym[17, :] = [[0, 0, 1], [-1, 0, 0], [0, -1, 0]]
    Sym[18, :] = [[0, 0, -1], [1, 0, 0], [0, -1, 0]]; Sym[19, :] = [[0, 0, -1], [-1, 0, 0], [0, 1, 0]]
    Sym[20, :] = [[0, 0, -1], [0, -1, 0], [-1, 0, 0]]; Sym[21, :] = [[0, 0, -1], [0, 1, 0], [1, 0, 0]]
    Sym[22, :] = [[0, 0, 1], [0, -1, 0], [1, 0, 0]]; Sym[23, :] = [[0, 0, 1], [0, 1, 0], [-1, 0, 0]]
    arr = np.atleast_2d(arr)
    Result = []
    for i in range(len(Sym)):
        for j in range(len(arr)): Result.append(dot(Sym[i, :], arr[j]))
    return np.unique(np.array(Result), axis=0)

def Create_minimal_cell_Method_1(sigma, uvw, R):
    uvw = np.array(uvw)
    MiniCell_1 = np.zeros([3, 3])
    MiniCell_1[:, 2] = uvw
    lim = 20
    x = np.arange(-lim, lim + 1, 1)
    y, z = x, x
    indice = (np.stack(np.meshgrid(x, y, z)).T).reshape(len(x) ** 3, 3)
    indice_0 = indice[np.where(np.sum(abs(indice), axis=1) != 0)]
    condition1 = ((abs(dot(indice_0, uvw) / norm(indice_0, axis=1) / norm(uvw))).round(7))
    indice_0 = indice_0[np.where(condition1 != 1)]
    tol = 0.001
    newindices = dot(R, indice_0.T).T
    nn = indice_0[np.all(abs(np.round(newindices) - newindices) < 1e-6, axis=1)]
    TestVecs = nn[np.argsort(norm(nn, axis=1))]
    Found = False
    count = 0
    while (not Found) and count < len(TestVecs) - 1:
        if 1 - ang(TestVecs[count], MiniCell_1[:, 2]) > tol:
            MiniCell_1[:, 1] = (TestVecs[count])
            count += 1
            for j in range(len(TestVecs)):
                if (1 - ang(TestVecs[j], MiniCell_1[:, 2]) > tol) and (1 - ang(TestVecs[j], MiniCell_1[:, 1]) > tol):
                    if (ang(TestVecs[j], cross(MiniCell_1[:, 2], MiniCell_1[:, 1])) > tol):
                        MiniCell_1[:, 0] = (TestVecs[j]).astype(int)
                        Det1 = abs(round(det(MiniCell_1), 5))
                        MiniCell_1 = (MiniCell_1).astype(int)
                        MiniCell_2 = ((np.round(dot(R, MiniCell_1), 7)).astype(int))
                        Det2 = abs(round(det(MiniCell_2), 5))
                        if ((abs(Det1 - sigma)) < tol and (abs(Det2 - sigma)) < tol):
                            Found = True
                            break
    return (MiniCell_1, MiniCell_2) if Found else (None, None)

def Create_Possible_GB_Plane_List(uvw, m, n, lim=3):
    uvw = np.array(uvw)
    Theta = get_cubic_theta(uvw, m, n)
    Sigma = get_cubic_sigma(uvw, m, n)
    R1 = rot(uvw, Theta)
    x = np.arange(-lim, lim + 1, 1)
    y, z = x, x
    indice = (np.stack(np.meshgrid(x, y, z)).T).reshape(len(x) ** 3, 3)
    indice_0 = indice[np.where(np.sum(abs(indice), axis=1) != 0)]
    indice_0 = indice_0[np.argsort(norm(indice_0, axis=1))]
    Min_1, Min_2 = Create_minimal_cell_Method_1(Sigma, uvw, R1)
    if Min_1 is None: return [], [], [], []
    V1 = np.zeros([len(indice_0), 3])
    V2 = np.zeros([len(indice_0), 3])
    GBtype = []
    tol = 0.001
    MP = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [0, 1, 1], [1, 0, 1]], dtype='float')
    for i in range(len(indice_0)):
        if CommonDivisor(indice_0[i])[1] <= 1:
            V1[i, :] = indice_0[i, 0]*Min_1[:, 0] + indice_0[i, 1]*Min_1[:, 1] + indice_0[i, 2]*Min_1[:, 2]
            V2[i, :] = indice_0[i, 0]*Min_2[:, 0] + indice_0[i, 1]*Min_2[:, 1] + indice_0[i, 2]*Min_2[:, 2]
    V1 = (V1[~np.all(V1 == 0, axis=1)]).astype(int)
    V2 = (V2[~np.all(V2 == 0, axis=1)]).astype(int)
    MeanPlanes = (V1 + V2) / 2
    for i in range(len(V1)):
        if ang(V1[i], uvw) < tol:
            for j in range(len(SymmEquivalent(MP))):
                if 1 - ang(MeanPlanes[i], SymmEquivalent(MP)[j]) < tol:
                    GBtype.append('Symmetric Tilt'); break
            else:
                GBtype.append('Tilt')
        elif 1 - ang(V1[i], uvw) < tol:
            GBtype.append('Twist')
        else:
            GBtype.append('Mixed')
    return V1, V2, MeanPlanes, GBtype

def Basis(basis):
    if str(basis) == 'fcc':
        return np.array([[0, 0, 0], [0.5, 0, 0.5], [0.5, 0.5, 0], [0, 0.5, 0.5]], dtype=float)
    elif str(basis) == 'bcc':
        return np.array([[0, 0, 0], [0.5, 0.5, 0.5]], dtype=float)
    elif str(basis) == 'sc':
        return np.eye(3)
    elif str(basis) == 'diamond':
        return np.array([[0, 0, 0], [0.5, 0, 0.5], [0.5, 0.5, 0], [0, 0.5, 0.5],
                         [0.25, 0.25, 0.25], [0.75, 0.25, 0.75], [0.75, 0.75, 0.25], [0.25, 0.75, 0.75]], dtype=float)
    return np.eye(3)

def integer_vector(vec, max_mult=2000, tol=1e-6):
    """将浮点方向向量转换为最简整数方向向量"""
    vec = np.array(vec, dtype=float)
    for j in range(1, max_mult):
        testV = j * vec
        if np.all(abs(np.round(testV) - testV) < tol):
            int_vec = np.round(testV).astype(int)
            return CommonDivisor(int_vec)[0]
    return np.round(vec).astype(int)


def Find_Orthogonal_cell(basis, uvw, m, n, GB1):
    lim = 15
    uvw = np.array(uvw)
    Theta = get_cubic_theta(uvw, m, n)
    Sigma = get_cubic_sigma(uvw, m, n)
    R = rot(uvw, Theta)
    GB2 = np.round((dot(R, GB1)), 6)
    x = np.arange(-lim, lim + 1, 1)
    y, z = x, x
    indice = (np.stack(np.meshgrid(x, y, z)).T).reshape(len(x) ** 3, 3)
    indice_0 = indice[np.where(np.sum(abs(indice), axis=1) != 0)]
    indice_0 = indice_0[np.argsort(norm(indice_0, axis=1))]
    OrthoCell_1 = np.zeros([3, 3])
    OrthoCell_1[:, 0] = np.array(GB1)
    OrthoCell_2 = np.zeros([3, 3])
    OrthoCell_2[:, 0] = np.array(GB2)
    Min_1, Min_2 = Create_minimal_cell_Method_1(Sigma, uvw, R)
    if Min_1 is None: return None
    tol = 0.001
    if ang(OrthoCell_1[:, 0], uvw) < tol:
        OrthoCell_1[:, 1] = uvw; OrthoCell_2[:, 1] = uvw
    else:
        for i in range(len(indice_0)):
            v1 = indice_0[i, 0]*Min_1[:, 0] + indice_0[i, 1]*Min_1[:, 1] + indice_0[i, 2]*Min_1[:, 2]
            v2 = indice_0[i, 0]*Min_2[:, 0] + indice_0[i, 1]*Min_2[:, 1] + indice_0[i, 2]*Min_2[:, 2]
            if ang(v1, OrthoCell_1[:, 0]) < tol:
                OrthoCell_1[:, 1] = v1; OrthoCell_2[:, 1] = v2; break
    OrthoCell_1[:, 2] = np.cross(OrthoCell_1[:, 0], OrthoCell_1[:, 1])
    OrthoCell_2[:, 2] = np.cross(OrthoCell_2[:, 0], OrthoCell_2[:, 1])
    if CommonDivisor(OrthoCell_1[:, 2])[1] == CommonDivisor(OrthoCell_2[:, 2])[1]:
        OrthoCell_1[:, 2] = CommonDivisor(OrthoCell_1[:, 2])[0]
        OrthoCell_2[:, 2] = CommonDivisor(OrthoCell_2[:, 2])[0]
    
    Volume_1 = det(OrthoCell_1)
    Volume_2 = det(OrthoCell_2)
    Num = int(round(Volume_1 * len(Basis(basis)) * 2))
    
    if np.isclose(Volume_1, Volume_2, atol=1e-4):
        if basis in ['sc', 'diamond']:
            return OrthoCell_1.astype(float), OrthoCell_2.astype(float), Num
        elif basis in ['fcc', 'bcc']:
            ortho1, ortho2 = np.zeros((3, 3)), np.zeros((3, 3))
            base = np.delete(Basis('fcc').T, 0, 1) if basis == 'fcc' else ((np.array([[0.5, 0.5, 0.5], [0.5, 0.5, -0.5], [-0.5, 0.5, 0.5]])).T)
            for i in range(3):
                Min_d = min(CommonDivisor(dot(OrthoCell_1[:, i], inv(base)))[1], CommonDivisor(dot(OrthoCell_2[:, i], inv(base)))[1])
                ortho1[:, i] = OrthoCell_1[:, i] / Min_d
                ortho2[:, i] = OrthoCell_2[:, i] / Min_d
            return ortho1, ortho2, Num
    return None

def integerMatrix(a):
    b = np.zeros((3, 3))
    a = np.array(a)
    for i in range(3):
        for j in range(1, 2000):
            testV = j * a[i]
            if integer_array(testV):
                b[i] = testV
                break
    return b

def odd_even(M1):
    d_e = np.array([['a','a','a'],['a','a','a'],['a','a','a']], dtype=str)
    for i in range(3):
        for j in range(3):
            d_e[i][j] = 'e' if abs(M1[i][j]) % 2 == 0 else 'd'
    return d_e

def self_test_b(a):
    z_b = np.eye(3, 3); M = a.copy()
    for i in range(3):
        if np.all(odd_even(M)[:, i] == ['d', 'd', 'd']):
            z_b[i][i] = 0.5; break
    return z_b

def binary_test_b(a):
    count = 0; z_b = np.eye(3, 3)
    for i in [0, 1]:
        for j in [1, 2]:
            if i != j and count < 1:
                M = a.copy(); M[:, j] = M[:, i] + M[:, j]
                if np.all(odd_even(M)[:, j] == ['d', 'd', 'd']):
                    count += 1; z_b[i][j] = 0.5; z_b[j][j] = 0.5
    return z_b

def tertiary_test_b(a):
    z_b = np.eye(3, 3); M = a.copy(); M[:, 2] = M[:, 0] + M[:, 1] + M[:, 2]
    for k in range(3):
        if np.all(odd_even(M)[:, k] == ['d', 'd', 'd']):
            z_b[0][k] = 0.5; z_b[1][k] = 0.5; z_b[2][k] = 0.5; break
    return z_b

def body_centering(b):
    z_b = np.eye(3, 3)
    while det(z_b) != 0.5:
        if det(self_test_b(b)) == 0.5: z_b = self_test_b(b); break
        if det(binary_test_b(b)) == 0.5: z_b = binary_test_b(b); break
        if det(tertiary_test_b(b)) == 0.5: z_b = tertiary_test_b(b); break
    return z_b

def face_centering(a):
    z_f = np.eye(3, 3); M = a.copy(); count = 0
    for i in range(3):
        if np.all(odd_even(M)[:, i] == ['d', 'd', 'e']) or np.all(odd_even(M)[:, i] == ['e', 'd', 'd']) or np.all(odd_even(M)[:, i] == ['d', 'e', 'd']):
            count += 1; z_f[i][i] = 0.5
    if det(z_f) == 0.25: return z_f
    for i in [0, 1]:
        for j in [1, 2]:
            if i != j and count < 2:
                M = a.copy(); M[:, j] = M[:, i] + M[:, j]
                if np.all(odd_even(M)[:, j] == ['d', 'd', 'e']) or np.all(odd_even(M)[:, j] == ['e', 'd', 'd']) or np.all(odd_even(M)[:, j] == ['d', 'e', 'd']):
                    count += 1; z_f[i][j] = 0.5; z_f[j][j] = 0.5
    return z_f if det(z_f) == 0.25 else None

def DSC_vec(basis, sigma, minicell):
    D_sc = np.round(sigma * (inv(minicell).T), 6).astype(int)
    if basis == 'sc': D = D_sc.copy()
    if basis == 'bcc': D = dot(D_sc, body_centering(D_sc))
    if basis in ['fcc', 'diamond']: D = dot(D_sc, face_centering(D_sc))
    return D

def DSC_on_plane(D, Pnormal):
    D_proj = np.zeros((3, 3)); Pnormal = np.array(Pnormal)
    for i in range(3):
        D_proj[:, i] = D[:, i] - (dot(D[:, i], Pnormal) / norm(Pnormal)) * Pnormal/norm(Pnormal)
    return D_proj

def build_oriented_cell(ortho_mat, basis):
    Orint = integerMatrix(ortho_mat.T)
    CubeCoords = np.array([[1,0,0], [0,1,0], [0,0,1], [1,1,0], [0,1,1], [1,0,1], [1,1,1], [0,0,0]], dtype=float)
    transformed = np.array([np.dot(Orint.T, c) for c in CubeCoords])
    x = np.arange(min(transformed[:,0])-1, max(transformed[:,0])+2, 1)
    y = np.arange(min(transformed[:,1])-1, max(transformed[:,1])+2, 1)
    z = np.arange(min(transformed[:,2])-1, max(transformed[:,2])+2, 1)
    indice = (np.stack(np.meshgrid(x, y, z)).T).reshape(len(x)*len(y)*len(z), 3)
    Base = Basis(basis)
    Atoms = []
    for i in range(len(indice)):
        for j in range(len(Base)): Atoms.append(indice[i, 0:3] + Base[j, 0:3])
    Atoms = np.array(Atoms)
    Con1 = dot(Atoms, ortho_mat.T[0]) / norm(ortho_mat.T[0]) + 0.001
    Con2 = dot(Atoms, ortho_mat.T[1]) / norm(ortho_mat.T[1]) + 0.001
    Con3 = dot(Atoms, ortho_mat.T[2]) / norm(ortho_mat.T[2]) + 0.001
    return Atoms[(Con1 >= 0) & (Con1 <= norm(ortho_mat.T[0])) & (Con2 >= 0) & (Con2 <= norm(ortho_mat.T[1])) & (Con3 >= 0) & (Con3 <= norm(ortho_mat.T[2]))]

# ==========================================
# Generation Classes
# ==========================================

class SingleCrystal_character:
    def __init__(self):
        self.element = 'Cr'
        self.File = 'LAMMPS'
        self.dim = np.array([1, 1, 1])
        
    def generate(self, basis, lat_p, orient_x, orient_y, target_dims):
        self.basis = basis
        self.LatP = lat_p
        
        vx = CommonDivisor(np.array(orient_x))[0]
        vy = CommonDivisor(np.array(orient_y))[0]
        vz = np.cross(vx, vy)
        vz = CommonDivisor(vz)[0]
        print(f"--> Calculated Z-axis orientation: {vz}")
        
        self.ortho_mat = np.column_stack((vx, vy, vz)).astype(float)
        atoms = build_oriented_cell(self.ortho_mat, self.basis)
        
        rot_mat = np.array([
            vx / norm(vx),
            vy / norm(vy),
            vz / norm(vz)
        ])
        atoms = dot(rot_mat, atoms.T).T
        
        Lx_base = norm(vx) * self.LatP
        Ly_base = norm(vy) * self.LatP
        Lz_base = norm(vz) * self.LatP
        
        target_X, target_Y, target_Z = target_dims
        self.dim[0] = max(1, int(round(target_X / Lx_base)))
        self.dim[1] = max(1, int(round(target_Y / Ly_base)))
        self.dim[2] = max(1, int(round(target_Z / Lz_base)))
        
        dimX, dimY, dimZ = self.dim
        print(f"\n--> Calculated base block repetitions [Nx Ny Nz]: [{dimX} {dimY} {dimZ}]")
        print(f"--> Actual Single Crystal size: ~ {dimX*Lx_base:.2f} x {dimY*Ly_base:.2f} x {dimZ*Lz_base:.2f} Angstroms")
        
        a = norm(vx)
        b = norm(vy)
        c = norm(vz)
        X_new = []
        for i in range(dimX):
            for j in range(dimY):
                for k in range(dimZ):
                    p = [i*a, j*b, k*c]
                    for atom in atoms:
                        X_new.append(p + atom)
        self.atoms = np.array(X_new)
        
    def write_output(self):
        name = "SingleCrystal"
        X = self.atoms * self.LatP
        
        a = norm(self.ortho_mat[:, 0])
        b = norm(self.ortho_mat[:, 1])
        c = norm(self.ortho_mat[:, 2])
        
        xlo = 0.0
        xhi = np.round(a * self.dim[0] * self.LatP, 8)
        ylo = 0.0
        yhi = np.round(b * self.dim[1] * self.LatP, 8)
        zlo = 0.0
        zhi = np.round(c * self.dim[2] * self.LatP, 8)
        
        Lx = xhi - xlo
        Ly = yhi - ylo
        Lz = zhi - zlo

        if self.File == "XYZ":
            with open(name + '.xyz', 'w') as f:
                f.write(f"{len(X)}\n")
                f.write(f'Lattice="{Lx} 0.0 0.0 0.0 {Ly} 0.0 0.0 0.0 {Lz}" Properties=species:S:1:pos:R:3 pbc="T T T"\n')
                for atom in X: f.write(f"{self.element} {atom[0]:.8f} {atom[1]:.8f} {atom[2]:.8f}\n")
            print(f"Saved: {name}.xyz")

        elif self.File == "VASP":
            with open('POSCAR_' + name, 'w') as f:
                f.write(f"{name}\n1.0\n")
                f.write(f"{Lx:.8f} 0.0 0.0\n0.0 {Ly:.8f} 0.0\n0.0 0.0 {Lz:.8f}\n")
                f.write(f"{self.element}\n{len(X)}\nCartesian\n")
                np.savetxt(f, X, fmt='%.8f %.8f %.8f')
            print(f"Saved: POSCAR_{name}")

        elif self.File == "LAMMPS":
            NumberAt = len(X)
            Type_all = np.ones(NumberAt, int).reshape(1, -1)
            Counter = np.arange(1, NumberAt + 1).reshape(1, -1)
            FinalMat = np.concatenate((Counter.T, Type_all.T, X), axis=1)

            with open('data.' + name, 'w') as f:
                f.write(f"# LAMMPS Data File for {name}\n\n{NumberAt} atoms\n1 atom types\n\n")
                f.write(f"{xlo:.8f} {xhi:.8f} xlo xhi\n{ylo:.8f} {yhi:.8f} ylo yhi\n{zlo:.8f} {zhi:.8f} zlo zhi\n\nAtoms\n\n")
                np.savetxt(f, FinalMat, fmt='%i %i %.8f %.8f %.8f')
            print(f"Saved: data.{name}")

class GB_character:
    def __init__(self):
        self.element = 'Cr'
        self.File = 'LAMMPS'
        self.overD = 0.4
        self.whichG = 'g1'
        self.dim = np.array([1, 1, 1])

    def setup(self, axis, basis, LatP, m, n, gb):
        self.axis = np.array(axis)
        self.basis = str(basis)
        self.LatP = float(LatP)
        self.m = int(m)
        self.n = int(n)
        self.gbplane = np.array(gb)
        self.sigma = get_cubic_sigma(self.axis, self.m, self.n)
        self.theta = get_cubic_theta(self.axis, self.m, self.n)
        self.R = rot(self.axis, self.theta)

        res = Find_Orthogonal_cell(self.basis, self.axis, self.m, self.n, self.gbplane)
        if res:
            self.ortho1, self.ortho2, self.Num = res
        else:
            print("Error: Could not find orthogonal cells. Is it a valid CSL plane?")
            sys.exit()

    def generate_atoms(self):
        Or_1, Or_2 = self.ortho1.T, self.ortho2.T
        
        self.rot1 = np.array([
            Or_1[1, :]/norm(Or_1[1, :]),
            Or_1[2, :]/norm(Or_1[2, :]),
            Or_1[0, :]/norm(Or_1[0, :])
        ])
        self.rot2 = np.array([
            Or_2[1, :]/norm(Or_2[1, :]),
            Or_2[2, :]/norm(Or_2[2, :]),
            Or_2[0, :]/norm(Or_2[0, :])
        ])

        self.atoms1 = build_oriented_cell(self.ortho1, self.basis)
        self.atoms2 = build_oriented_cell(self.ortho2, self.basis)
        
        self.atoms1 = dot(self.rot1, self.atoms1.T).T
        self.atoms2 = dot(self.rot2, self.atoms2.T).T
        
        self.atoms2[:, 2] -= norm(Or_2[0, :])

    def expand(self, target_dims):
        a_len = norm(self.ortho1[:, 1]) 
        b_len = norm(self.ortho1[:, 2]) 
        c_len = norm(self.ortho1[:, 0]) 
        
        Lx_base = a_len * self.LatP
        Ly_base = b_len * self.LatP
        Lz_base = c_len * self.LatP
        
        target_X, target_Y, target_Z = target_dims
        self.dim = np.array([1, 1, 1])
        self.dim[0] = max(1, int(round(target_X / Lx_base)))
        self.dim[1] = max(1, int(round(target_Y / Ly_base)))
        self.dim[2] = max(1, int(round((target_Z / 2.0) / Lz_base))) # 双晶模型Z轴对称，单边只占一半
        
        dimX, dimY, dimZ = self.dim
        print(f"\n--> Calculated base block repetitions [Nx Ny Nz]: [{dimX} {dimY} {dimZ}]")
        print(f"--> Actual GB supercell size: ~ {dimX*Lx_base:.2f} x {dimY*Ly_base:.2f} x {dimZ*Lz_base*2:.2f} Angstroms")
        
        X, Y = self.atoms1.copy(), self.atoms2.copy()
        X_new, Y_new = [], []
        for i in range(dimX):
            for j in range(dimY):
                for k in range(dimZ):
                    p1 = [i*a_len, j*b_len,  k*c_len]
                    p2 = [i*a_len, j*b_len, -k*c_len]
                    for l in range(len(X)): X_new.append(p1 + X[l])
                    for m in range(len(Y)): Y_new.append(p2 + Y[m])
        self.atoms1, self.atoms2 = np.array(X_new), np.array(Y_new)

    def remove_overlap(self, verbose=True):
        periodic_length = norm(self.ortho1[:, 0]) * self.dim[2]
        periodic_image = self.atoms2 + [0, 0, periodic_length * 2]
        
        # 修复2：动态搜索半径，防止漏判
        search_radius = max(1.0, self.overD + 0.5)
        
        IndX = np.where((self.atoms1[:, 2] < search_radius) | (self.atoms1[:, 2] > (periodic_length - search_radius)))[0]
        IndY = np.where(self.atoms2[:, 2] > -search_radius)[0]
        IndY_image = np.where(periodic_image[:, 2] < (periodic_length + search_radius))[0]
        
        X_new = self.atoms1[IndX]
        Y_new = np.concatenate((self.atoms2[IndY], periodic_image[IndY_image]))
        IndY_new = np.concatenate((IndY, IndY_image))

        Lx = norm(self.ortho1[:, 1]) * self.dim[0]
        Ly = norm(self.ortho1[:, 2]) * self.dim[1]

        unique_x_set = set()
        unique_y_set = set()
        
        overD_sq = self.overD ** 2
        
        # 修复3：通过循环遍历取代 meshgrid 笛卡尔积，彻底消除 OOM (Out Of Memory)
        for i, x_atom in enumerate(X_new):
            dx = x_atom[0] - Y_new[:, 0]
            dy = x_atom[1] - Y_new[:, 1]
            dz = x_atom[2] - Y_new[:, 2]
            
            dx = dx - Lx * np.round(dx / Lx)
            dy = dy - Ly * np.round(dy / Ly)
            
            norms_sq = dx**2 + dy**2 + dz**2
            
            overlaps = np.where(norms_sq < overD_sq)[0]
            if len(overlaps) > 0:
                unique_x_set.add(i)
                unique_y_set.update(overlaps)
                
        unique_x = np.array(list(unique_x_set), dtype=int)
        unique_y = np.array(list(unique_y_set), dtype=int)

        if verbose:
            print(f"--> Removed {len(unique_x)} overlapping atoms.")
            
        if self.whichG.lower() == "g1" and len(unique_x) > 0:
            self.atoms1 = np.delete(self.atoms1, IndX[unique_x], axis=0)
        elif self.whichG.lower() != "g1" and len(unique_y) > 0:
            self.atoms2 = np.delete(self.atoms2, IndY_new[unique_y], axis=0)


    def get_single_crystal_orientation(self):
        """
        根据双晶 GB1 的取向，生成建议的单晶取向参数。
        单晶的 X/Y/Z 轴将与双晶输出坐标系中的 GB1 晶粒对齐。
        """
        # ortho1 的列向量含义：
        #   [:, 0] = GB 平面法向（双晶输出坐标系的 Z 轴）
        #   [:, 1] = GB 平面内方向（双晶输出坐标系的 X 轴）
        #   [:, 2] = GB 平面内方向（双晶输出坐标系的 Y 轴）

        orient_x = integer_vector(self.ortho1[:, 1])
        orient_y = integer_vector(self.ortho1[:, 2])
        orient_z = integer_vector(self.ortho1[:, 0])

        # 计算各方向在晶体学坐标系中的归一化向量
        x_dir = self.ortho1[:, 1] / norm(self.ortho1[:, 1])
        y_dir = self.ortho1[:, 2] / norm(self.ortho1[:, 2])
        z_dir = self.ortho1[:, 0] / norm(self.ortho1[:, 0])

        return {
            'orient_x': orient_x.tolist(),
            'orient_y': orient_y.tolist(),
            'orient_z': orient_z.tolist(),
            'x_dir': x_dir.tolist(),
            'y_dir': y_dir.tolist(),
            'z_dir': z_dir.tolist(),
        }

    def save_info_txt(self, filename="GB_info.txt"):
        """
        将双晶的完整信息和建议的单晶取向保存为 txt 文件。
        """
        sc_orient = self.get_single_crystal_orientation()

        # 计算旋转角度
        theta_deg = degrees(self.theta)

        # GB2 的取向信息
        orient_x2 = integer_vector(self.ortho2[:, 1])
        orient_y2 = integer_vector(self.ortho2[:, 2])
        orient_z2 = integer_vector(self.ortho2[:, 0])

        # 构建输出内容
        lines = []
        lines.append("=" * 70)
        lines.append(" Grain Boundary (GB) Structure Information Summary")
        lines.append("=" * 70)
        lines.append("")
        lines.append("--- Basic Parameters ---")
        lines.append(f"Lattice Basis      : {self.basis}")
        lines.append(f"Lattice Parameter  : {self.LatP:.4f} Angstrom")
        lines.append(f"Element            : {self.element}")
        lines.append("")
        lines.append("--- Rotation Axis & Sigma ---")
        lines.append(f"Rotation Axis [uvw]: {self.axis.tolist()}")
        lines.append(f"Sigma (Σ)          : {int(self.sigma)}")
        lines.append(f"Rotation Angle     : {theta_deg:.4f}°")
        lines.append(f"m, n (CSL params)  : {self.m}, {self.n}")
        lines.append("")
        lines.append("--- Grain Boundary Plane ---")
        lines.append(f"GB1 Plane Normal   : {self.gbplane.tolist()}")
        lines.append(f"GB2 Plane Normal   : {np.round(dot(self.R, self.gbplane), 6).tolist()}")
        lines.append("")
        lines.append("--- GB1 Grain Orientation (Crystal Directions in Output Cartesian Frame) ---")
        lines.append(f"Output X-axis ↔ Crystal Direction : {sc_orient['orient_x']}")
        lines.append(f"Output Y-axis ↔ Crystal Direction : {sc_orient['orient_y']}")
        lines.append(f"Output Z-axis ↔ Crystal Direction : {sc_orient['orient_z']}")
        lines.append("")
        lines.append("--- GB2 Grain Orientation (Crystal Directions in Output Cartesian Frame) ---")
        lines.append(f"Output X-axis ↔ Crystal Direction : {orient_x2.tolist()}")
        lines.append(f"Output Y-axis ↔ Crystal Direction : {orient_y2.tolist()}")
        lines.append(f"Output Z-axis ↔ Crystal Direction : {orient_z2.tolist()}")
        lines.append("")
        lines.append("--- Recommended Single Crystal Orientation (to match GB1) ---")
        lines.append("To generate a single crystal with the SAME orientation as GB1,")
        lines.append("use the following parameters in Single Crystal mode:")
        lines.append("")
        lines.append(f"  X-axis orientation [u v w] : {' '.join(map(str, sc_orient['orient_x']))}")
        lines.append(f"  Y-axis orientation [u v w] : {' '.join(map(str, sc_orient['orient_y']))}")
        lines.append(f"  (Z-axis will be auto-calculated as cross(X, Y))")
        lines.append("")
        lines.append("  Verification:")
        lines.append(f"    dot(X, Y) = {dot(sc_orient['orient_x'], sc_orient['orient_y']):.6f} (should be 0)")
        lines.append(f"    Z (auto)  = {np.cross(sc_orient['orient_x'], sc_orient['orient_y']).tolist()}")
        lines.append(f"    matches GB1 Z [{sc_orient['orient_z']}] ? {np.allclose(np.cross(sc_orient['orient_x'], sc_orient['orient_y']), sc_orient['orient_z'])}")
        lines.append("")
        lines.append("--- Supercell Dimensions ---")
        lines.append(f"Repetitions [Nx Ny Nz] : {self.dim.tolist()}")
        a = norm(self.ortho1[:, 1]) * self.dim[0] * self.LatP
        b = norm(self.ortho1[:, 2]) * self.dim[1] * self.LatP
        c = norm(self.ortho1[:, 0]) * self.dim[2] * self.LatP * 2  # 双晶总厚度
        lines.append(f"Actual GB supercell    : ~ {a:.2f} x {b:.2f} x {c:.2f} Angstroms")
        lines.append("")
        lines.append("=" * 70)
        lines.append(" End of Summary")
        lines.append("=" * 70)

        content = "\n".join(lines)

        with open(filename, 'w') as f:
            f.write(content)

        print(f"\n📄 GB structure info saved to: {filename}")
        print("\n" + "-" * 60)
        print("💡 Recommended Single Crystal Orientation (to match GB1):")
        print(f"   X-axis: {sc_orient['orient_x']}")
        print(f"   Y-axis: {sc_orient['orient_y']}")
        print(f"   Z-axis: {sc_orient['orient_z']} (auto-calculated)")
        print("-" * 60)

        return content

    def write_output(self, suffix=""):
        name = f"GB_{self.gbplane[0]}{self.gbplane[1]}{self.gbplane[2]}{suffix}"
        
        X, Y = self.atoms1 * self.LatP, self.atoms2 * self.LatP
        dimx, dimy, dimz = self.dim
        
        a = norm(self.ortho1[:, 1])
        b = norm(self.ortho1[:, 2])
        c = norm(self.ortho1[:, 0])

        xlo = 0.0
        xhi = np.round(a * dimx * self.LatP, 8)
        ylo = 0.0
        yhi = np.round(b * dimy * self.LatP, 8)
        zlo = -1 * np.round(c * dimz * self.LatP, 8)
        zhi = np.round(c * dimz * self.LatP, 8)
        
        Lx = xhi - xlo
        Ly = yhi - ylo
        Lz = zhi - zlo

        shift_vec = np.array([xlo, ylo, zlo])
        X_shifted = X - shift_vec
        Y_shifted = Y - shift_vec

        if self.File == "XYZ":
            with open(name + '.xyz', 'w') as f:
                f.write(f"{len(X_shifted) + len(Y_shifted)}\n")
                f.write(f'Lattice="{Lx} 0.0 0.0 0.0 {Ly} 0.0 0.0 0.0 {Lz}" Properties=species:S:1:pos:R:3 pbc="T T T"\n')
                for atom in X_shifted: f.write(f"{self.element} {atom[0]:.8f} {atom[1]:.8f} {atom[2]:.8f}\n")
                for atom in Y_shifted: f.write(f"{self.element} {atom[0]:.8f} {atom[1]:.8f} {atom[2]:.8f}\n")
            print(f"Saved: {name}.xyz")

        elif self.File == "VASP":
            with open('POSCAR_' + name, 'w') as f:
                f.write(f"GB_{name}\n1.0\n")
                f.write(f"{Lx:.8f} 0.0 0.0\n0.0 {Ly:.8f} 0.0\n0.0 0.0 {Lz:.8f}\n")
                f.write(f"{self.element}\n{len(X_shifted) + len(Y_shifted)}\nCartesian\n")
                np.savetxt(f, np.concatenate((X_shifted, Y_shifted)), fmt='%.8f %.8f %.8f')
            print(f"Saved: POSCAR_{name}")

        elif self.File == "LAMMPS":
            NumberAt = len(X) + len(Y)
            Type_all = np.ones(NumberAt, int).reshape(1, -1)
            Counter = np.arange(1, NumberAt + 1).reshape(1, -1)
            Wf = np.concatenate((X, Y), axis=0)
            FinalMat = np.concatenate((Counter.T, Type_all.T, Wf), axis=1)

            with open('data.' + name, 'w') as f:
                f.write(f"# LAMMPS Data File for {name}\n\n{NumberAt} atoms\n1 atom types\n\n")
                f.write(f"{xlo:.8f} {xhi:.8f} xlo xhi\n{ylo:.8f} {yhi:.8f} ylo yhi\n{zlo:.8f} {zhi:.8f} zlo zhi\n\nAtoms\n\n")
                np.savetxt(f, FinalMat, fmt='%i %i %.8f %.8f %.8f')
            print(f"Saved: data.{name}")
            
    def translate_and_generate(self, grid_a, grid_b):
        tol = 0.001
        if (1 - ang(self.gbplane, self.axis) < tol):
            M1, _ = Create_minimal_cell_Method_1(self.sigma, self.axis, self.R)
            D = (1 / self.sigma * DSC_vec(self.basis, self.sigma, M1))
            Dvecs = DSC_on_plane(D, self.gbplane)
            TransDvecs = np.round(dot(self.rot1, Dvecs), 7)
            shift1 = TransDvecs[:, 0] / 2
            shift2 = TransDvecs[:, 1] / 2
        else:
            if norm(self.ortho1[:, 1]) > norm(self.ortho1[:, 2]):
                shift1 = (1 / grid_a) * (norm(self.ortho1[:, 1]) * np.array([1, 0, 0]))
                shift2 = (1 / grid_b) * (norm(self.ortho1[:, 2]) * np.array([0, 1, 0]))
            else:
                shift1 = (1 / grid_a) * (norm(self.ortho1[:, 2]) * np.array([0, 1, 0]))
                shift2 = (1 / grid_b) * (norm(self.ortho1[:, 1]) * np.array([1, 0, 0]))

        print(f"\n<<------ Generating {grid_a * grid_b} translated structures ------>>")
        original_atoms1 = self.atoms1.copy()
        original_atoms2 = self.atoms2.copy()
        
        Lx = norm(self.ortho1[:, 1]) * self.dim[0]
        Ly = norm(self.ortho1[:, 2]) * self.dim[1]
        
        for i in range(grid_a):
            for j in range(grid_b):
                self.atoms1 = original_atoms1.copy()
                self.atoms2 = original_atoms2.copy()
                
                # 施加平移
                current_shift = i * shift1 + j * shift2
                self.atoms1 += current_shift
                
                # 将平移后的原子折叠回边界内
                self.atoms1[:, 0] = self.atoms1[:, 0] % Lx
                self.atoms1[:, 1] = self.atoms1[:, 1] % Ly
                
                if self.overD > 0:
                    self.remove_overlap(verbose=False)
                self.write_output(suffix=f"_trans_{i}_{j}")

# ==========================================
# Interactive Wizard
# ==========================================

def get_user_input(prompt, default, cast_type):
    val = input(f"{prompt} [Default: {default}]: ").strip()
    if not val: return default
    try: return cast_type(val)
    except:
        print("Invalid format. Using default.")
        return default

def get_list_input(prompt, default_list, cast_type):
    default_str = " ".join(map(str, default_list))
    val = input(f"{prompt} (space separated) [Default: {default_str}]: ").strip()
    if not val: return default_list
    try: return [cast_type(x) for x in val.split()]
    except:
        print("Invalid format. Using default.")
        return default_list

def main():
    print("=" * 70)
    print(" 🌟 Interactive Crystal & Grain Boundary Builder (Z-Axis Normal) 🌟 ")
    print("=" * 70)

    # 1. 询问构建类型
    build_type = input("Do you want to build a Single Crystal (S) or a Bicrystal (B)? (S/B) [Default: B]: ").strip().upper() or 'B'

    if build_type == 'S':
        print("\n--- [Mode: Single Crystal] ---")
        basis = input("Lattice basis (fcc/bcc/sc/diamond) [Default: bcc]: ").strip().lower() or 'bcc'
        lat_p = get_user_input("Lattice parameter (Angstrom)", 2.88, float)
        element = input("Element symbol (e.g. Cr) [Default: Cr]: ").strip() or 'Cr'
        file_format = input("Output format (LAMMPS/VASP/XYZ) [Default: LAMMPS]: ").strip().upper() or 'LAMMPS'
        if file_format not in ["XYZ", "VASP", "LAMMPS"]: file_format = "LAMMPS"
        
        target_dims = get_list_input("Target cell physical dimensions in Angstroms [Lx Ly Lz]", [50.0, 50.0, 50.0], float)
        
        print("\nDefine the crystal orientation (must be orthogonal):")
        orient_x = get_list_input("X-axis orientation [u v w]", [1, 0, 0], int)
        orient_y = get_list_input("Y-axis orientation [u v w]", [0, 1, 0], int)
        
        if dot(orient_x, orient_y) != 0:
            print("❌ Error: X and Y orientations are not orthogonal! Please try again.")
            return

        print("\nGenerating Single Crystal... Please wait.")
        sc = SingleCrystal_character()
        sc.element = element
        sc.File = file_format
        sc.generate(basis, lat_p, orient_x, orient_y, target_dims)
        sc.write_output()

    else:
        print("\n--- [Mode: Bicrystal (Grain Boundary)] ---")
        basis = input("Lattice basis (fcc/bcc/sc/diamond) [Default: bcc]: ").strip().lower() or 'bcc'
        lat_p = get_user_input("Lattice parameter (Angstrom)", 2.88, float)
        axis_in = get_list_input("Rotation axis [u v w]", [1, 1, 0], int)
        axis = CommonDivisor(np.array(axis_in))[0]

        limit = 50
        print(f"\n--- Calculating possible Sigma values for axis {axis} up to {limit} ---")
        sigmas_found = []
        for i in range(1, limit):
            tt = get_theta_m_n_list(axis, i)
            if len(tt) > 0:
                sigmas_found.append(i)
                print(f"Sigma: {i:3d}  |  Theta: {degrees(tt[0][0]):5.2f}°")

        if not sigmas_found:
            print("No Sigmas found in this range. Exiting.")
            return
        sigma = get_user_input("\nSelect a Sigma from the list above", sigmas_found[1] if len(sigmas_found)>1 else sigmas_found[0], int)
        
        try: _, m, n = get_theta_m_n_list(axis, sigma)[0]
        except: print("Invalid Sigma selected. Exiting."); return

        print(f"\n--- Calculating valid GB Planes for Sigma {sigma} (m={m}, n={n}) ---")
        V1, V2, _, GB_type = Create_Possible_GB_Plane_List(axis, m, n, lim=3)
        valid_planes = []
        
        print("Index | GB1 Plane          | GB2 Plane          | Type")
        print("-" * 60)
        idx = 0
        for i in range(len(V1)):
            if Find_Orthogonal_cell(basis, axis, m, n, V1[i]) is not None:
                valid_planes.append(V1[i])
                print(f" [{idx:2d}] | {str(V1[i]):<18s} | {str(V2[i]):<18s} | {GB_type[i]}")
                idx += 1

        if not valid_planes:
            print("No compatible orthogonal cells found for this basis. Try another Sigma.")
            return

        plane_idx = get_user_input(f"\nSelect Plane Index (0 to {len(valid_planes)-1})", 0, int)
        selected_plane = valid_planes[plane_idx]

        print("\n--- Modeling Parameters ---")
        target_dims = get_list_input("Target supercell dimensions in Angstroms [Lx Ly Lz] (Z is GB normal)", [50.0, 50.0, 50.0], float)
        
        overlap = get_user_input("Overlap removal distance (Fraction of lattice parameter, 0 to disable)", 0.4, float)
        
        which_g = 'g1'
        if overlap > 0:
            which_g = input("Remove overlap from which grain? (g1/g2) [Default: g1]: ").strip().lower() or 'g1'

        element = input("Element symbol for the grain boundary (e.g. Cr) [Default: Cr]: ").strip() or 'Cr'
        
        file_format = input("Output format (LAMMPS/VASP/XYZ) [Default: LAMMPS]: ").strip().upper() or 'LAMMPS'
        if file_format not in ["XYZ", "VASP", "LAMMPS"]: file_format = "LAMMPS"

        do_trans = input("\nDo you want to perform Rigid Body Translations? (yes/no) [Default: no]: ").strip().lower() == 'yes'
        grid_a, grid_b = 1, 1
        if do_trans:
            grid_a = get_user_input("Grid divisions for shift in-plane vector 1 (X direction)", 10, int)
            grid_b = get_user_input("Grid divisions for shift in-plane vector 2 (Y direction)", 10, int)

        print("\nGenerating structure(s)... Please wait.")
        gb = GB_character()
        gb.overD = overlap
        gb.whichG = which_g
        gb.element = element
        gb.File = file_format
        
        gb.setup(axis, basis, lat_p, m, n, selected_plane)
        gb.generate_atoms()
        
        gb.expand(target_dims)
        
        if do_trans:
            gb.translate_and_generate(grid_a, grid_b)
            gb.save_info_txt(filename=f"GB_info_{gb.gbplane[0]}{gb.gbplane[1]}{gb.gbplane[2]}.txt")
        else:
            if gb.overD > 0:
                gb.remove_overlap(verbose=True)
            gb.write_output()
            gb.save_info_txt(filename=f"GB_info_{gb.gbplane[0]}{gb.gbplane[1]}{gb.gbplane[2]}.txt")
            
    print("=" * 60)
    print("Done! 🎉")

if __name__ == '__main__':
    main()
