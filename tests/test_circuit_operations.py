import pytest
import math
from qpiai_quantum.circuit import Circuit
from qpiai_quantum.icr.circuitoperation import (
    HGate,
    XGate,
    YGate,
    ZGate,
    IDGate,
    SXGate,
    SGate,
    SDGGate,
    TGate,
    TDGGate,
    RXGate,
    RYGate,
    RZGate,
    PGate,
    CPGate,
    RZZGate,
    CXGate,
    CYGate,
    CZGate,
    SwapGate,
    ISwapGate,
    CCXGate,
    CSwapGate,
    BarrierOperation,
    MeasureOperation,
    MCXGate,
)

# ==========================================
# GATE ADDITION TESTS
# ==========================================


def test_single_qubit_standard_gates():
    circuit = Circuit(1)
    circuit.h(0)
    circuit.x(0)
    circuit.y(0)
    circuit.z(0)
    circuit.id(0)
    circuit.s(0)
    circuit.sdg(0)
    circuit.t(0)
    circuit.tdg(0)
    circuit.sx(0)

    ops = list(circuit.icr.evolve)
    assert len(ops) == 10
    assert isinstance(ops[0], HGate)
    assert isinstance(ops[1], XGate)
    assert isinstance(ops[2], YGate)
    assert isinstance(ops[3], ZGate)
    assert isinstance(ops[4], IDGate)
    assert isinstance(ops[5], SGate)
    assert isinstance(ops[6], SDGGate)
    assert isinstance(ops[7], TGate)
    assert isinstance(ops[8], TDGGate)
    assert isinstance(ops[9], SXGate)

    for op in ops:
        assert op.qubits == [0]


def test_single_qubit_parametric_gates():
    circuit = Circuit(1)
    theta = math.pi / 4
    circuit.rx(0, theta)
    circuit.ry(0, theta)
    circuit.rz(0, theta)
    circuit.p(0, theta)

    ops = list(circuit.icr.evolve)
    assert len(ops) == 4
    assert isinstance(ops[0], RXGate)
    assert isinstance(ops[1], RYGate)
    assert isinstance(ops[2], RZGate)
    assert isinstance(ops[3], PGate)

    for op in ops:
        assert op.qubits == [0]
        assert op.params == [theta]


def test_two_and_three_qubit_gates():
    circ = Circuit(3)
    circ.cx(0, 1)
    circ.cy(0, 1)
    circ.cz(0, 1)
    circ.swap(0, 1)
    circ.iswap(0, 1)
    circ.cp(0, 1, 0.5)
    circ.rzz(0, 1, 0.5)
    circ.ccx(0, 1, 2)
    circ.cswap(0, 1, 2)

    ops = list(circ.icr.evolve)
    assert len(ops) == 9
    assert isinstance(ops[0], CXGate)
    assert isinstance(ops[1], CYGate)
    assert isinstance(ops[2], CZGate)
    assert isinstance(ops[3], SwapGate)
    assert isinstance(ops[4], ISwapGate)
    assert isinstance(ops[5], CPGate)
    assert isinstance(ops[6], RZZGate)
    assert isinstance(ops[7], CCXGate)
    assert isinstance(ops[8], CSwapGate)


def test_barrier_and_measure():
    circ = Circuit(2, 2)
    circ.barrier(0, 1)
    circ.measure(0, 0)
    circ.measure_all()

    ops = list(circ.icr.evolve)
    assert len(ops) == 4  # 1 barrier + 1 explicit measure + 2 measure_all
    assert isinstance(ops[0], BarrierOperation)
    assert isinstance(ops[1], MeasureOperation)
    assert isinstance(ops[2], MeasureOperation)


# ==========================================
# CIRCUIT NATIVE OPERATIONS (COMPOSE / INVERSE)
# ==========================================


def test_circuit_compose_basic():
    circuit1 = Circuit(2)
    circuit1.h(0)

    circuit2 = Circuit(2)
    circuit2.x(1)

    circuit1.compose(circuit2)

    ops = list(circuit1.icr.evolve)
    assert len(ops) == 2
    assert ops[0].gate_name == "H"
    assert ops[0].qubits == [0]
    assert ops[1].gate_name == "X"
    assert ops[1].qubits == [1]


def test_circuit_compose_with_qubit_map():
    base = Circuit(3)
    base.h(0)

    append_c = Circuit(2)
    append_c.cx(0, 1)

    # Map append_c's qubit 0 to base's qubit 2
    # Map append_c's qubit 1 to base's qubit 1
    base.compose(append_c, qubits=[2, 1])

    ops = list(base.icr.evolve)
    assert len(ops) == 2
    assert ops[1].gate_name == "CX"
    assert ops[1].qubits == [2, 1]


def test_circuit_inverse():
    circuit = Circuit(2)
    circuit.h(0)
    circuit.rx(1, 0.5)
    circuit.cx(0, 1)

    inv = circuit.inverse()

    ops = list(inv.icr.evolve)
    assert len(ops) == 3
    # Reverse order: CX, RX(-0.5), H
    assert isinstance(ops[0], CXGate)

    assert isinstance(ops[1], RXGate)
    assert ops[1].params[0] == -0.5

    assert isinstance(ops[2], HGate)


# ==========================================
# CIRCUIT UTILITY / PROPERTY TESTS
# ==========================================


def test_circuit_properties_and_size():
    circ = Circuit(3, 2)
    assert circ.num_qubits == 3
    assert circ.num_clbits == 2
    assert circ.size() == 0

    circ.h(0)
    circ.cx(0, 1)
    assert circ.size() == 2


def test_circuit_depth():
    circ = Circuit(3)
    # Layer 1
    circ.h(0)
    circ.h(1)
    assert circ.depth() == 1

    # Layer 2
    circ.cx(0, 1)
    assert circ.depth() == 2

    # Layer 3
    circ.cx(1, 2)
    assert circ.depth() == 3


def test_circuit_to_json():
    circ = Circuit(2)
    circ.h(0)
    circ.rx(1, 1.5)

    json_data = circ.to_json()
    assert isinstance(json_data, dict)
    assert "name" in json_data
    assert "num_qubits" in json_data
    assert "evolve" in json_data
    assert len(json_data["evolve"]) == 2


def test_circuit_str_representation():
    circ = Circuit(1)
    circ.x(0)
    s = str(circ)
    assert isinstance(s, str)
    assert len(s) > 0


def test_circuit_list_gates():
    circ = Circuit(2)
    circ.h(0)
    circ.cx(0, 1)
    circ.rx(0, 0.5)
    circ.measure(0, 0)
    circ.barrier(0, 1)

    stats = circ.list_gates()
    assert stats["total_operations"] == 5
    assert stats["measurements"] == 1
    assert stats["barriers"] == 1
    assert stats["total_gates"] == 3
    assert stats["single_qubit_gates"] == 2
    assert stats["two_qubit_gates"] == 1

    counts = stats["gate_counts"]
    assert counts.get("H") == 1
    assert counts.get("CX") == 1
    assert counts.get("RX") == 1
    assert counts.get("MEASURE") == 1
    assert counts.get("BARRIER") == 1


def test_circuit_to_circuit_operation():
    circ = Circuit(2)
    circ.h(0)
    circ.rx(0, 1.2)
    circ.cx(0, 1)

    op = circ.to_circuit_operation(name="CustomGate")
    assert op.gate_name == "CustomGate"
    assert set(op.qubits) == {0, 1}
    assert op.params == [1.2]
    assert len(op.order) == 3


def test_circuit_to_circuit_operation_fails_with_measure():
    circ = Circuit(1, 1)
    circ.measure(0, 0)

    with pytest.raises(Exception):
        circ.to_circuit_operation()


def test_circuit_to_qasm():
    circ = Circuit(1)
    circ.h(0)

    # Check that exporting to QASM format doesn't fail
    qasm_str = circ.to_qasm()
    assert isinstance(qasm_str, str)
    assert "qreg" in qasm_str or "OPENQASM" in qasm_str


def test_mcx_gate():
    circ = Circuit(4)
    gate = MCXGate([0, 1, 2], 3)
    circ.add_operation(gate)

    ops = list(circ.icr.evolve)
    assert len(ops) == 1
    assert isinstance(ops[0], MCXGate)
    assert ops[0].qubits == [0, 1, 2, 3]


def test_local_simulator_iswap():
    circ = Circuit(2)
    circ.iswap(0, 1)
    result = circ.run(device_name="QpiAI-QSV-Local", shots=100)
    assert result is not None


def test_local_simulator_composite_operation():
    sub_circ = Circuit(2)
    sub_circ.h(0)
    sub_circ.cx(0, 1)

    composite_op = sub_circ.to_circuit_operation(name="Entangler")

    main_circ = Circuit(2)
    main_circ.add_operation(composite_op)

    result = main_circ.run(device_name="QpiAI-QSV-Local", shots=100)
    assert result is not None

