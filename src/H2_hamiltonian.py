#Importing libraries
import numpy as np
import scipy.special as special
import itertools as it
from openfermionpsi4 import run_psi4
from openfermion.transforms import get_fermion_operator, jordan_wigner, bravyi_kitaev
from openfermion.linalg import get_ground_state, get_sparse_operator
from openfermion.chem import MolecularData
import psi4


def h2_hamiltonian(bondlength, JW=True):
    """
    Function returning the Jordan Wigner transformed H2 Hamiltonian,
    based on the following link: https://blog.artwolf.in/a?ID=49e3eada-cdca-4f83-b338-4cd442aae73a

    Args:
            Bondlenght(float):  Bondlength between the atoms in units of Angstrom

    Returns:
            List of Jordan-Wigner transformed Hamiltonian
    """
    #Defines geometry and basis and details
    geometry = [["H", [0,0,0]], ["H", [0,0,bondlength]]]
    basis = "sto-3g"
    multiplicity = 1
    charge = 0
    description = "test" #str()
    
    #Extracting the molecule information
    molecule = MolecularData(geometry, basis, multiplicity, charge, description)
    molecule = run_psi4(molecule)

    #JW and Bravyi-Kitaev transformation as an openfermion object
    if JW is True:
        hamiltonian = jordan_wigner(get_fermion_operator(molecule.get_molecular_hamiltonian()))
    else:
        hamiltonian = bravyi_kitaev(get_fermion_operator(molecule.get_molecular_hamiltonian()))

    sparse_operator = get_sparse_operator(hamiltonian)
    gs=get_ground_state(sparse_operator)[0]

    psi4.set_memory('1000 MB')

    h2 = psi4.geometry("""
    H 
    H 1 0.7408481486
    """)

    psi4.set_options({'basis': 'sto-3g',
                    #'scf_type': 'pk',
                    'e_convergence': 1e-8,
                    'd_convergence': 1e-8})

    gs=psi4.energy('fci')

    print(gs)
    return hamiltonian, gs


h2_hamiltonian(1.4)