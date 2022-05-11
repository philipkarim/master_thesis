import psi4
psi4.set_memory('1000 MB')

h2 = psi4.geometry("""
H 
H 1 0.75
""")

psi4.set_options({'basis': 'sto-3g',
                  #'scf_type': 'pk',
                  'e_convergence': 1e-8,
                  'd_convergence': 1e-8})

psi4.energy('fci')
