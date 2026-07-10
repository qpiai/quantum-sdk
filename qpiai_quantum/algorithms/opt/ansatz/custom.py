"""Custom ansatz template."""


def custom_ansatz(*args, **kwargs):
    """Construct a custom user-defined ansatz.

    This function is reserved for future implementation. Use one of the
    built-in ansatz builders (e.g. ``hardware_efficient_ansatz``,
    ``standard_vqe_ansatz``) or pass a :class:`Circuit` instance directly.

    Raises:
        NotImplementedError: Always raised until a concrete implementation
            is provided.
    """
    raise NotImplementedError(
        "custom_ansatz is not implemented yet. Pass a Circuit instance directly "
        "or use one of the built-in ansatz builders."
    )
