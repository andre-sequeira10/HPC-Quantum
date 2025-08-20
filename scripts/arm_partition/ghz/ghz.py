from qiskit import QuantumCircuit
#from qiskit.providers.basic_provider import BasicSimulator
#from qiskit.compiler import transpile, assemble
from qiskit_aer import AerSimulator
from mpi4py import MPI

mpicomm = MPI.COMM_WORLD
mpirank = mpicomm.Get_rank()
mpisize = mpicomm.Get_size()

def create_ghz_circuit(n_qubits):
    circuit = QuantumCircuit(n_qubits)
    circuit.h(0)
    for qubit in range(n_qubits - 1):
        circuit.cx(qubit, qubit + 1)
        circuit.cy(qubit, qubit +1)
        circuit.cz(qubit, qubit+1)
    for qubit in range(1,n_qubits-1):
        circuit.cx(qubit,qubit-1)
        circuit.cy(qubit,qubit-1)
        circuit.cz(qubit,qubit-1)
    return circuit

shots = 10
#depth=1000
qubits = 31
block_bits = 21
#backend = BasicSimulator()



backend = AerSimulator(method = 'statevector')#, max_parallel_threads=48)
#backend = QasmSimulator(method="statevector",max_parallel_threads=48)
#circuit = transpile(create_ghz_circuit(qubits),backend = backend)
circuit=create_ghz_circuit(qubits)
circuit.measure_all()

#if mpirank == 0:
#print("before")
#circuit.save_statevector()
#print("after")

backend.set_options(
    max_parallel_threads = 1,
    max_parallel_experiments = 0,
    max_parallel_shots = 1,
    statevector_parallel_threshold = 16,
    blocking_enable=True,
    blocking_qubits=block_bits
)
#if mpirank == 0:
    #print("resultou")

result=backend.run(circuit,shots=shots).result()
#if mpirank == 0:
    #print("resultou 2")

#result = backend.run(circuit, shots=shots, seed_simulator=10).result()
#if mpirank == 0:
print(result)

#print("state vector ",result.get_statevector())
print("state vector ",result.get_counts())
adict = result.to_dict()
meta = adict['metadata']
print("meta: ", meta)
myrank = meta['mpi_rank']
print("myrank: ",myrank)
