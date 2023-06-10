"""
Ethereum Forks
^^^^^^^^^^^^^^

Detects Python packages that specify Ethereum hardforks.
"""

import importlib
import importlib.util
import pkgutil
from pathlib import PurePath
from pkgutil import ModuleInfo
from types import ModuleType
from typing import Any, Iterator, List, Optional, Type, TypeVar

H = TypeVar("H", bound="Hardfork")


class Hardfork:
    """
    Metadata associated with an Ethereum hardfork.
    """

    mod: ModuleType

    @classmethod
    def discover(cls: Type[H], base: Optional[PurePath] = None) -> List[H]:
        """
        Find packages which contain Ethereum hardfork specifications.
        """
        if base is None:
            ethereum = importlib.import_module("ethereum")
        else:
            spec = importlib.util.spec_from_file_location(
                "ethereum", base / "__init__.py", submodule_search_locations=[]
            )
            ethereum = importlib.util.module_from_spec(spec)
            if spec.loader and hasattr(spec.loader, "exec_module"):
                spec.loader.exec_module(ethereum)  # type: ignore[attr-defined]

        path = getattr(ethereum, "__path__", None)
        if path is None:
            raise ValueError("module `ethereum` has no path information")

        modules = pkgutil.iter_modules(path, ethereum.__name__ + ".")
        modules = (module for module in modules if module.ispkg)
        forks: List[H] = []
        new_package = None

        for pkg in modules:
            if isinstance(pkg.module_finder, importlib.abc.MetaPathFinder):
                found = pkg.module_finder.find_module(pkg.name, None)
            else:
                found = pkg.module_finder.find_module(pkg.name)

            if not found:
                raise Exception(f"unable to load module {pkg.name}")

            mod = found.load_module(pkg.name)
            block = getattr(mod, "MAINNET_FORK_BLOCK", -1)

            if block == -1:
                continue

            # If the fork block is unknown, for example in a
            # new improvement proposal, it will be set as None.
            if block is None:
                if new_package is not None:
                    raise ValueError(
                        "cannot have more than 1 new fork package."
                    )
                else:
                    new_package = cls(mod)
                continue

            forks.append(cls(mod))

        forks.sort(key=lambda fork: fork.block)
        if new_package:
            forks.append(new_package)

        return forks

    def __init__(self, mod: ModuleType) -> None:
        self.mod = mod

    @property
    def block(self) -> int:
        """
        Block number of the first block in this hard fork.
        """
        return getattr(self.mod, "MAINNET_FORK_BLOCK")  # noqa: B009

    @property
    def path(self) -> Optional[str]:
        """
        Path to the module containing this hard fork.
        """
        got = getattr(self.mod, "__path__", None)
        if got is None or isinstance(got, str):
            return got

        try:
            assert isinstance(got[0], str)
            return got[0]
        except IndexError:
            return None

    @property
    def short_name(self) -> str:
        """
        Short name (without the `ethereum.` prefix) of the hard fork.
        """
        return self.mod.__name__.split(".")[-1]

    @property
    def name(self) -> str:
        """
        Name of the hard fork.
        """
        return self.mod.__name__

    @property
    def title_case_name(self) -> str:
        """
        Name of the hard fork.
        """
        return self.short_name.replace("_", " ").title()

    def __repr__(self) -> str:
        """
        Return repr(self).
        """
        return (
            self.__class__.__name__
            + "("
            + f"name={self.name!r}, "
            + f"block={self.block}, "
            + "..."
            + ")"
        )

    def import_module(self) -> ModuleType:
        """
        Return the module containing this specification.
        """
        return self.mod

    def module(self, name: str) -> Any:
        """
        Import if necessary, and return the given module belonging to this hard
        fork.
        """
        return importlib.import_module(self.mod.__name__ + "." + name)

    def optimized_module(self, name: str) -> Any:
        """
        Import if necessary, and return the given module belonging to this hard
        fork's optimized implementation.
        """
        assert self.mod.__name__.startswith("ethereum.")
        module = "ethereum_optimized" + self.mod.__name__[8:] + "." + name
        return importlib.import_module(module)

    def iter_modules(self) -> Iterator[ModuleInfo]:
        """
        Iterate through the (sub-)modules describing this hardfork.
        """
        if self.path is None:
            raise ValueError(f"cannot walk {self.name}, path is None")

        return pkgutil.iter_modules(self.path, self.name + ".")

    def walk_packages(self) -> Iterator[ModuleInfo]:
        """
        Iterate recursively through the (sub-)modules describing this hardfork.
        """
        if self.path is None:
            raise ValueError(f"cannot walk {self.name}, path is None")

        return pkgutil.walk_packages(self.path, self.name + ".")
