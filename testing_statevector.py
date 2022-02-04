from qiskit import QuantumCircuit, transpile, Aer
from qiskit.providers.aer import AerSimulator, backends
from qiskit.tools.visualization import plot_histogram

import random

import matplotlib.pyplot as plt
import qiskit as qk

"""
circ = QuantumCircuit(40, 40)

# Initialize with a Hadamard layer
circ.h(range(40))
# Apply some random CNOT and T gates
qubit_indices = [i for i in range(40)]
for i in range(10):
    control, target, t = random.sample(qubit_indices, 3)
    circ.cx(control, target)
    circ.t(t)
circ.measure(range(40), range(40))

# Create statevector method simulator
statevector_simulator = AerSimulator(method='statevector')

# Transpile circuit for backend
tcirc = transpile(circ, statevector_simulator)
"""
# Try and run circuit
#statevector_result =  statevector_simulator.run(tcirc, shots=1).result()
#print('This succeeded?: {}'.format(statevector_result.success))
#print('Why not? {}'.format(statevector_result.status))


"""
# Create extended stabilizer method simulator

# Transpile circuit for backend
tcirc = transpile(circ, extended_stabilizer_simulator)

extended_stabilizer_result = extended_stabilizer_simulator.run(tcirc, shots=1).result()
print('This succeeded?: {}'.format(extended_stabilizer_result.success))

circ.t(qr[qubit])
circ.tdg(qr[qubit])
circ.ccx(qr[control_1], qr[control_2], qr[target])
circ.u1(rotation_angle, qr[qubit])
"""
extended_stabilizer_simulator = Aer.get_backend('statevector_simulator')


small_circ = QuantumCircuit(4, 4)
small_circ.h(0)
small_circ.cx(0, 1)
small_circ.ry(1.51, 1)
small_circ.rx(1.1, 0)
small_circ.rx(1.1, 1)
small_circ.rx(1.1, 3)
small_circ.rx(1.1, 3)
small_circ.cx(0, 3)

small_circ.t(0)
#small_circ.measure([0, 1], [0, 1])
# This circuit should give 00 or 11 with equal probability...
expected_results ={'00': 50, '11': 50}

#tsmall_circ = transpile(small_circ, extended_stabilizer_simulator)
#result = extended_stabilizer_simulator.run(small_circ, shots=100).result()
#counts = result.get_counts(0)

backendtest = Aer.get_backend('statevector_simulator')


result = qk.execute(small_circ,
            backend=backendtest,
            shots=0
            )
result = result.result()
counts = result.get_counts(small_circ)

#measure_key = len(str(next(iter(counts.items()))))*'1'

for key,value in counts.items():
    print(key, value)
    if key == '0':
        prediction = value

print(value)


print('100 shots in {}s'.format(result.time_taken))

plot_histogram([expected_results, counts],
               legend=['Expected', 'Extended Stabilizer'])
plt.show()

#print(Aer.backends())