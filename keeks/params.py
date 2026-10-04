"""
sklearn-style parameter introspection for keeks strategies and allocators.

:class:`ParameterMixin` gives every keeks strategy and allocator the
:func:`~sklearn.base.BaseEstimator.get_params` / ``set_params`` pair on the
sklearn convention: the introspected parameters are exactly the constructor's
named parameters, each stored verbatim as a same-named public attribute, and
``get_params(deep=True)`` recurses into constructor parameters that are
themselves introspectable estimators (e.g. :class:`RiskAversionScaling`'s
``inner``) using sklearn's double-underscore paths (``inner__factor``).

The mixin reads the constructor signature dynamically, so a subclass only
needs to follow the storage convention - the parameter names are not
duplicated in a per-class list.
"""

import inspect

__author__ = "willmcginnis"


class ParameterMixin:
    """
    Mixin adding sklearn-style ``get_params`` / ``set_params`` introspection.

    The convention (sklearn's ``BaseEstimator``, minus fitting, which keeks
    strategies do not have): every named ``__init__`` parameter is stored
    verbatim as a same-named public attribute, ``get_params()`` reads those
    attributes back, and ``set_params(**params)`` writes them and returns
    ``self``. ``**kwargs``-style constructors are not introspectable and
    reject the mixin.

    Note for bound allocators: descriptive inputs bind at construction, and
    an allocator reprices by fresh construction. ``set_params`` updates the
    stored descriptors without re-running a solver, so for the solver-backed
    families it exists for API compatibility (hyperparameter-search
    wrappers) rather than as a reprice operation - fresh construction is
    the supported reprice path. For the online family the updates take
    effect at the next settlement.
    """

    def get_params(self, deep: bool = True) -> dict[str, object]:
        """
        Read the constructor parameters back from their public attributes.

        Parameters
        ----------
        deep : bool, default=True
            When True, constructor parameters whose values are themselves
            introspectable (expose ``get_params``) are expanded into
            ``name__subparam`` entries, in addition to the parameter itself.

        Returns
        -------
        dict
            One entry per named ``__init__`` parameter, keyed by parameter
            name, sorted for stable display; nested entries follow when
            ``deep`` is True.
        """
        return self._collect_params(deep=deep)

    def set_params(self, **params: object) -> "ParameterMixin":
        """
        Set constructor parameters, returning ``self``.

        Parameters
        ----------
        **params
            Parameter names as keywords, with ``name__subparam`` paths for
            nested introspectable parameters. Unknown names are rejected.

        Returns
        -------
        ParameterMixin
            ``self``, with the parameters assigned.

        Raises
        ------
        ValueError
            If a parameter name is unknown to this instance's constructor.
        """
        nested: dict[str, dict[str, object]] = {}
        for key, value in params.items():
            name, separator, subname = key.partition("__")
            if not separator:
                if name not in self._param_names():
                    raise ValueError(
                        f"Invalid parameter {key!r} for "
                        f"{type(self).__name__}; valid parameters are "
                        f"{sorted(self._param_names())}"
                    )
                setattr(self, name, value)
            else:
                target = getattr(self, name, None)
                if not hasattr(target, "get_params"):
                    raise ValueError(
                        f"Invalid nested parameter {key!r} for "
                        f"{type(self).__name__}: {name!r} is not an "
                        f"introspectable parameter"
                    )
                nested.setdefault(name, {})[subname] = value
        for name, subparams in nested.items():
            getattr(self, name).set_params(**subparams)
        return self

    def _param_names(self) -> list[str]:
        """
        The named parameters of this instance's ``__init__``, sorted.

        ``self``, ``*args``, and ``**kwargs`` are excluded: only parameters
        a caller can pass by name participate in introspection.
        """
        init = self.__init__
        if init is object.__init__:
            return []
        signature = inspect.signature(init)
        return sorted(
            name
            for name, parameter in signature.parameters.items()
            if name != "self"
            and parameter.kind
            not in (inspect.Parameter.VAR_POSITIONAL, inspect.Parameter.VAR_KEYWORD)
        )

    def _collect_params(self, deep: bool) -> dict[str, object]:
        """
        Build the ``get_params`` result, expanding nested estimators.

        A value is treated as a nested introspectable parameter exactly when
        it exposes ``get_params`` - the same duck-typed test sklearn applies
        - so user-defined strategies and allocators compose the same way the
        shipped ones do.
        """
        params: dict[str, object] = {}
        for name in self._param_names():
            value = getattr(self, name)
            params[name] = value
            if deep and hasattr(value, "get_params"):
                for subname, subvalue in value.get_params(deep=True).items():
                    params[f"{name}__{subname}"] = subvalue
        return params
