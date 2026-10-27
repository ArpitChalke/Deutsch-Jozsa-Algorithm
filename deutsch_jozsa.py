"""
Deutsch-Jozsa Algorithm — Qiskit Fall Fest 2026, Problem Statement 3

Install:
    python -m pip install qiskit qiskit-aer matplotlib

Run:
    python deutsch_jozsa.py
"""
from qiskit import QuantumCircuit, transpile
from qiskit_aer import AerSimulator
import matplotlib.pyplot as plt

N_INPUTS = 3
SHOTS = 1024

ORACLES = {
    "constant_0": "Constant oracle: f(x) = 0",
    "constant_1": "Constant oracle: f(x) = 1",
    "balanced_x0": "Balanced oracle: f(x) = x[0]",
    "balanced_x0_xor_x1": "Balanced oracle: f(x) = x[0] XOR x[1]",
}


def apply_oracle(qc, n, oracle_name):
    """Apply one of the four promised constant/balanced oracle functions."""
    ancilla = n

    if oracle_name == "constant_0":
        pass  # f(x) = 0
    elif oracle_name == "constant_1":
        qc.x(ancilla)  # f(x) = 1
    elif oracle_name == "balanced_x0":
        qc.cx(0, ancilla)  # f(x) = x[0]
    elif oracle_name == "balanced_x0_xor_x1":
        if n < 2:
            raise ValueError("This oracle requires at least two input qubits.")
        qc.cx(0, ancilla)
        qc.cx(1, ancilla)  # f(x) = x[0] XOR x[1]
    else:
        raise ValueError(f"Unknown oracle: {oracle_name}")


def deutsch_jozsa_circuit(n, oracle_name):
    """Create a Deutsch-Jozsa circuit with n input qubits and one ancilla."""
    if n < 2:
        raise ValueError("Set N_INPUTS to 2 or greater for all four example oracles.")

    qc = QuantumCircuit(n + 1, n)
    ancilla = n

    # Initialize ancilla to |1>; input qubits start in |0...0>.
    qc.x(ancilla)

    # Hadamard gates create input superposition and ancilla state |->.
    for qubit in range(n + 1):
        qc.h(qubit)

    # One oracle query.
    apply_oracle(qc, n, oracle_name)

    # Interference step.
    for qubit in range(n):
        qc.h(qubit)

    # Measure input register only.
    for qubit in range(n):
        qc.measure(qubit, qubit)

    return qc


def expected_classification(oracle_name):
    return "constant" if oracle_name.startswith("constant") else "balanced"


def classify_counts(counts, n):
    """All-zero result means constant; a nonzero result means balanced."""
    zero_string = "0" * n
    return "constant" if set(counts.keys()) == {zero_string} else "balanced"


def run_oracle(simulator, n, oracle_name, shots):
    circuit = deutsch_jozsa_circuit(n, oracle_name)
    compiled = transpile(circuit, simulator)
    result = simulator.run(compiled, shots=shots).result()
    counts = result.get_counts()

    expected = expected_classification(oracle_name)
    observed = classify_counts(counts, n)

    if expected == "constant":
        verified = all(bits == "0" * n for bits in counts)
    else:
        verified = all(bits != "0" * n for bits in counts)

    return circuit, counts, expected, observed, verified


def main():
    if N_INPUTS < 2:
        raise ValueError("Set N_INPUTS to 2 or greater.")

    simulator = AerSimulator()
    results = []

    print("=" * 70)
    print("DEUTSCH-JOZSA ALGORITHM — PROBLEM STATEMENT 3")
    print("=" * 70)
    print(f"Input qubits: {N_INPUTS}")
    print(f"Shots per oracle: {SHOTS}")

    for oracle_name, description in ORACLES.items():
        circuit, counts, expected, observed, verified = run_oracle(
            simulator, N_INPUTS, oracle_name, SHOTS
        )
        results.append({
            "name": oracle_name,
            "description": description,
            "circuit": circuit,
            "counts": counts,
            "expected": expected,
            "observed": observed,
            "verified": verified,
        })

        print("\n" + "-" * 70)
        print(description)
        print("Measurement counts:", counts)
        print("Expected:", expected)
        print("Observed:", observed)
        print("Verification passed:", verified)

        # Save a circuit diagram. Text output is printed if Matplotlib drawing fails.
        try:
            figure = circuit.draw(output="mpl", fold=-1)
            filename = f"circuit_{oracle_name}.png"
            figure.savefig(filename, bbox_inches="tight", dpi=160)
            plt.close(figure)
            print("Circuit diagram saved:", filename)
        except Exception as error:
            print("Could not save circuit image:", error)
            print(circuit.draw(output="text"))

    if not all(item["verified"] for item in results):
        raise RuntimeError("At least one oracle verification failed.")

    print("\nAll four oracle tests passed in the ideal simulator.")

    # Plot measurement counts for all four oracle functions.
    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    for ax, item in zip(axes.flatten(), results):
        counts = item["counts"]
        bitstrings = list(counts.keys())
        frequencies = [counts[key] for key in bitstrings]
        ax.bar(bitstrings, frequencies)
        ax.set_title(item["name"])
        ax.set_xlabel("Measured input bitstring")
        ax.set_ylabel("Counts")
        ax.tick_params(axis="x", rotation=45)

    fig.suptitle("Deutsch-Jozsa measurement results")
    fig.tight_layout()
    fig.savefig("deutsch_jozsa_measurements.png", dpi=160, bbox_inches="tight")
    print("Measurement plot saved: deutsch_jozsa_measurements.png")
    plt.show()

    # Query-complexity comparison.
    quantum_queries = 1
    classical_queries = 2 ** (N_INPUTS - 1) + 1

    print("\n" + "=" * 70)
    print("ORACLE-QUERY COMPLEXITY")
    print("=" * 70)
    print(f"Input bits (n): {N_INPUTS}")
    print(f"Quantum oracle queries: {quantum_queries}")
    print(f"Classical deterministic worst-case queries: {classical_queries}")
    print(f"Classical-to-quantum query ratio: {classical_queries}:1")

    n_values = list(range(1, 11))
    classical_values = [2 ** (n - 1) + 1 for n in n_values]
    quantum_values = [1] * len(n_values)

    plt.figure(figsize=(8, 5))
    plt.plot(n_values, classical_values, marker="o",
             label="Classical deterministic worst case")
    plt.plot(n_values, quantum_values, marker="s",
             label="Deutsch-Jozsa quantum queries")
    plt.yscale("log", base=2)
    plt.xlabel("Number of input bits (n)")
    plt.ylabel("Oracle queries (log scale)")
    plt.title("Classical vs quantum oracle-query complexity")
    plt.grid(True, which="both", linestyle=":")
    plt.legend()
    plt.tight_layout()
    plt.savefig("query_complexity.png", dpi=160, bbox_inches="tight")
    print("Query-complexity graph saved: query_complexity.png")
    plt.show()


if __name__ == "__main__":
    main()
