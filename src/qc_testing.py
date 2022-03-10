import qiskit as qk
import time
#from numba import jit
import concurrent.futures

def run(circ_list):
    backend=qk.Aer.get_backend('statevector_simulator')
    backend.set_options(max_parallel_experiments=1)
    #backend.set_options(statevector_parallel_threshold=1)

    #backendtest.set_options(max_parallel_experiments=0)
    #backendtest.set_options(statevector_parallel_threshold=1)
            #backendtest.set_options(device='CPU')

    #job = qk.execute(circ_list,
    #            backend=backend,
    #            shots=0,
    #            optimization_level=0)
    #results = job.result()

    results = backend.run(circ_list).result()


    #test_list=[]

    #for i in range(len(circ_list)):
    #    test_list.append(results.get_statevector(i).probabilities([0])[1])

    #for i in range(len(circ_list)):
    #    test_list.append(results.get_statevector(i).probabilities([0])[1])
    #print(results)
    #print(probs_qubit_0[1])
    psi0=results.get_statevector(0).probabilities([0])[1]
    psi1=results.get_statevector(1).probabilities([0])[1]
    psi2=results.get_statevector(2).probabilities([0])[1]

    #psi=results.get_statevector(0)
    #probs_qubit_0 = results.get_statevector(1).probabilities([0])
    #print(probs_qubit_0[1])
    
    #probs_qubit_0 = results.get_statevector(2).probabilities([0])
    print(psi0, psi1, psi2)

    
    return psi0, psi1, psi2

#@jit(nopython=True) # Set "nopython" mode for best performance, equivalent to @njit


def run_single(circ_list):
    
    #job = qk.execute(circ_list,
    #            backend=qk.Aer.get_backend('statevector_simulator'),
    #            shots=0,
    #            optimization_level=0)
    #results = job.result()

    simulator2 = qk.Aer.get_backend('statevector_simulator')
    results = simulator2.run(circ_list).result()

    return results.get_statevector(0).probabilities([0])[1]



qc1=qk.QuantumCircuit(2)
qc1.h(0)
qc1.cx(0,1)
qc1.h(0)
qc1.cy(0,1)
qc1.h(0)
qc1.cx(1,0)
qc1.h(1)
qc1.cx(0,1)


qc2=qk.QuantumCircuit(3)
qc2.h(0)
qc2.cx(0,1)
qc2.cx(1,2)
qc2.h(1)
qc2.cx(1,0)
qc2.cx(1,0)

qc3=qk.QuantumCircuit(4)
qc3.h(0)
qc3.cx(0,1)
qc3.cx(1,2)
qc3.cx(2,3)
qc3.h(0)
qc3.cx(0,1)
qc3.cx(0,2)
qc3.cx(0,3)
#qc3.h(0)

#import multiprocessing as mp
#print("Number of processors: ", mp.cpu_count())





#run(qc3)

qc_l=[qc1, qc2, qc3, qc1, qc2, qc3, qc1, qc2, qc3]


run(qc_l)
#run(qc_l)


l_time=time.time()
for i in range(10):
    run(qc_l)
print(f'Time: {time.time()-l_time}')



single=time.time()
for i in range(10):
    run_single(qc1)
    run_single(qc2)
    run_single(qc3)
    run_single(qc1)
    run_single(qc2)
    run_single(qc3)    
    run_single(qc1)
    run_single(qc2)
    run_single(qc3)
    run_single(qc1)
    run_single(qc2)
    run_single(qc3)
    run_single(qc1)
    run_single(qc2)
    run_single(qc3)    
    run_single(qc1)
    run_single(qc2)
    run_single(qc3)
print(f'Single circ time: {time.time()-single}')



def parallel2(circ2):
    
    simulator2 = qk.Aer.get_backend('aer_simulator')
    result2 = simulator2.run(circ2).result()
    #print(result2)


#exit()
"""
This paralellization part works, but doesnt seem to be much faster
"""
paralell_time=time.time()
for i in range(100):
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as p:
        p.map(parallel2,qc_l)
print(f'Time for paralell run: {time.time()-paralell_time}')

paralell_time2=time.time()
for i in range(10):
    for j in range(len(qc_l)):
        parallel2(qc_l[j])    
        

print(f'Time for paralell_single run: {time.time()-paralell_time2}')


#if __name__ == '__main__':

#futures.ThreadPoolExecutor(max_workers=5)

#pool = mp.Pool(mp.cpu_count())
#results = pool.map(run_single, qc_l)
#pool.close()

#print(results)