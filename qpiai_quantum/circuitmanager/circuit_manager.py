import random


class CircuitManager:
    @staticmethod
    def generate_name() -> str:
        first_names = [
            "Floating",
            "Circling",
            "Levitating",
            "Flying",
            "Diving",
            "Soaring",
            "Hovering",
            "Gliding",
            "Swooping",
            "Swirling",
        ]
        last_names = [
            "Circuit",
            "Qubits",
            "GateMappings",
            "FuzzyOperations",
            "Unitaries",
            "States",
            "Entanglements",
            "Superpositions",
            "Quantumness",
        ]
        random_integer = random.randint(1111, 9999999)
        return (
            f"{random.choice(first_names)}_{random.choice(last_names)}_{random_integer}"
        )

    @staticmethod
    def save(name: str = None):
        if name is None:
            name = CircuitManager.generate_name()

        def decorator(func):
            def wrapper(*args, **kwargs):
                return func(*args, **kwargs)

            return wrapper

        return decorator
