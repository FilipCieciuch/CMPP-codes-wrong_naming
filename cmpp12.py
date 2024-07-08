# -*- coding: utf-8 -*-
"""
Created on Wed May 31 21:01:10 2023

@author: filip
"""


from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit
from qiskit import Aer, execute

q = QuantumRegister(2, 'qubit')
c = ClassicalRegister(2, 'bit')

circuit = QuantumCircuit(q, c)

circuit.h(q[0])
circuit.measure(q, c)
circuit.draw()



#%%

backend = Aer.get_backend('statevector_simulator')
job = execute(circuit, backend)
res = job.result()

