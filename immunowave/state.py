r"""Field states."""

import jax
from numpy.typing import NDArray
import equinox as eqx
import abc
from typing import Any
import matplotlib.pyplot as plt
import dataclasses

from immunowave import spatial

jax.config.update("jax_enable_x64", True)


class State(eqx.Module, abc.ABC):
    r"""Field variables. Subclass this to implement your state, adding fields for
    the different components of the system.
    """

    def __check_init__(self):
        named_scalar_fields = [
            (field.name, getattr(self, field.name))
            for field in dataclasses.fields(self)
        ]
        for i, (name, scalar_field) in enumerate(named_scalar_fields):
            if not isinstance(scalar_field, spatial.ScalarField):
                raise TypeError(
                    f"Field {name} must be a ScalarField, got {type(scalar_field)}"
                )
            if i > 0:
                named_scalar_fields[0][1].check_aligned(scalar_field)

    def plot(
        self,
        time_idx: int | None = None,
        axes: NDArray[plt.Axes] | None = None,
        **kwargs: Any,
    ) -> NDArray[plt.Axes]:
        r"""Plot the state.

        Args:
            time_idx: If ``self`` is the ``ys`` attribute of a :py:class:`diffrax.Solution` with a series of
                      time points, this index specifies which time index to plot
            axes: Axes on which to plot. If ``None``, a new figure is created.
            kwargs: Keyword arguments passed to :py:meth:`immunowave.spatial.ScalarField.plot`.
        """
        named_scalar_fields = [
            (field.name, getattr(self, field.name))
            for field in dataclasses.fields(self)
        ]
        if axes is None:
            fig, axes = plt.subplots(
                len(named_scalar_fields), 1, figsize=(6, 2 * len(named_scalar_fields))
            )
        for axis, (name, scalar_field) in zip(axes, named_scalar_fields):
            plt.sca(axis)
            scalar_field.plot(time_idx=time_idx, **kwargs)
            axis.set_title(name)
        plt.tight_layout()
        return axes
