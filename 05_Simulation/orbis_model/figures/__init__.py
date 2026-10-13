"""Figure-generation helpers for steady-state outputs."""

__all__ = ["generate_simulation_figures"]


def __getattr__(name: str):
    if name == "generate_simulation_figures":
        from .simulation_figures import generate_simulation_figures

        return generate_simulation_figures
    raise AttributeError(name)
