# Copyright (C) 2022-2023 Ethereum Foundation
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.

"""
Plugins for docc specific to the Ethereum execution specification.
"""

import logging
from collections import defaultdict
from inspect import getmembers, isfunction, ismethod
from itertools import tee
from pathlib import PurePath
from typing import (
    Dict,
    Final,
    FrozenSet,
    Generic,
    Iterable,
    Iterator,
    List,
    Optional,
    Set,
    Tuple,
    TypeVar,
)

import zss
from docc.context import Context
from docc.discover import Discover, T
from docc.document import BlankNode, Document, Node, Visit, Visitor
from docc.languages import verbatim
from docc.plugins import html
from docc.plugins.cst import PythonBuilder
from docc.plugins.references import Definition, Reference
from docc.settings import PluginSettings
from docc.source import Source
from docc.transform import Transform

from .forks import Hardfork

G = TypeVar("G")


def pairwise(iterable: Iterable[G]) -> Iterable[Tuple[G, G]]:
    """
    ABCDEFG --> AB BC CD DE EF FG
    """
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)


class EthereumDiscover(Discover):
    """
    Creates sources that represent the diff between two other sources, one per
    fork.
    """

    forks: List[Hardfork]
    settings: PluginSettings

    def __init__(self, config: PluginSettings) -> None:
        super().__init__(config)
        self.settings = config
        base = config.resolve_path(PurePath("src") / "ethereum")
        self.forks = Hardfork.discover(base=base)

    def discover(self, known: FrozenSet[T]) -> Iterator[Source]:
        """
        Find sources.
        """
        forks = {f.path: f for f in self.forks if f.path is not None}

        by_fork: Dict[Hardfork, Dict[PurePath, Source]] = defaultdict(dict)

        for source in known:
            if not source.relative_path:
                continue

            absolute_path = self.settings.resolve_path(source.relative_path)

            for fork_path, fork_fork in forks.items():
                try:
                    fork_relative_path = absolute_path.relative_to(fork_path)
                    fork = fork_fork
                    break
                except ValueError:
                    logging.debug(
                        "source `%s` is not part of fork `%s`",
                        source.relative_path,
                        fork_fork.short_name,
                        exc_info=True,
                    )
            else:
                continue

            if fork_relative_path in by_fork[fork]:
                raise Exception(
                    f"two sources claim same path `{fork_relative_path}`"
                )

            by_fork[fork][fork_relative_path] = source

        diff_count = 0
        for (before, after) in pairwise(self.forks):
            paths = set(by_fork[before].keys()) | set(by_fork[after].keys())

            for path in paths:
                diff_count += 1
                before_source = by_fork[before].get(path, None)
                after_source = by_fork[after].get(path, None)

                assert before_source or after_source

                output_path = (
                    PurePath("diffs")
                    / before.short_name
                    / after.short_name
                    / path
                )

                yield DiffSource(
                    before.name,
                    before_source,
                    after.name,
                    after_source,
                    output_path,
                )

        if 0 == diff_count:
            raise Exception("no diff pairs found")

        logging.info("Discovered %s pair(s) of sources to diff", diff_count)


S = TypeVar("S", bound=Source)


class DiffSource(Generic[S], Source):
    """
    A source that represents the difference between two other sources.
    """

    before_name: str
    before: Optional[S]

    after_name: str
    after: Optional[S]
    _output_path: PurePath

    def __init__(
        self,
        before_name: str,
        before: Optional[S],
        after_name: str,
        after: Optional[S],
        output_path: PurePath,
    ) -> None:
        self.before_name = before_name
        self.before = before

        self.after_name = after_name
        self.after = after

        self._output_path = output_path

    @property
    def relative_path(self) -> Optional[PurePath]:
        """
        Path to the Source (if one exists) relative to the project root.
        """
        return None

    @property
    def output_path(self) -> PurePath:
        """
        Where to write the output from this Source relative to the output path.
        """
        return self._output_path


class AfterNode(Node):
    """
    Represents content that was added in a diff.
    """

    child: Node

    def __init__(self, child: Node) -> None:
        self.child = child

    @property
    def children(self) -> Tuple[Node]:
        """
        Child nodes belonging to this node.
        """
        return (self.child,)

    def replace_child(self, old: Node, new: Node) -> None:
        """
        Replace the old node with the given new node.
        """
        if self.child == old:
            self.child = new


class BeforeNode(Node):
    """
    Represents content that was removed in a diff.
    """

    child: Node

    def __init__(self, child: Node) -> None:
        self.child = child

    @property
    def children(self) -> Tuple[Node]:
        """
        Child nodes belonging to this node.
        """
        return (self.child,)

    def replace_child(self, old: Node, new: Node) -> None:
        """
        Replace the old node with the given new node.
        """
        if self.child == old:
            self.child = new


class DiffNode(Node):
    """
    Marks a difference (or patch) with a deletion and an insertion.
    """

    before_name: str
    before: Node

    after_name: str
    after: Node

    def __init__(
        self, before_name: str, before: Node, after_name: str, after: Node
    ) -> None:
        self.before_name = before_name
        self.before = before

        self.after_name = after_name
        self.after = after

    @property
    def children(self) -> Tuple[Node, Node]:
        """
        Child nodes belonging to this node.
        """
        return (self.before, self.after)

    def replace_child(self, old: Node, new: Node) -> None:
        """
        Replace the old node with the given new node.
        """
        if self.before == old:
            self.before = new
        if self.after == old:
            self.after = new

    def __repr__(self) -> str:
        """
        String representation of this object.
        """
        return (
            f"{self.__class__.__name__}(..., "
            f"before_name={self.before_name!r}, "
            f"after_name={self.after_name!r})"
        )


class EthereumBuilder(PythonBuilder):
    """
    A `PythonBuilder` that additionally builds `Document`s from `DiffSource`s.
    """

    def build(
        self,
        unprocessed: Set[Source],
        processed: Dict[Source, Document],
    ) -> None:
        """
        Consume unprocessed Sources and insert their Documents into processed.
        """
        # Build normal Python documents.
        super().build(unprocessed, processed)

        # Build diff documents.
        source_set = set(s for s in unprocessed if isinstance(s, DiffSource))
        unprocessed -= source_set

        before_unprocessed = {s.before for s in source_set if s.before}
        after_unprocessed = {s.after for s in source_set if s.after}

        # Rebuild the sources so we get distinct tree objects.
        before_processed: Dict[Source, Document] = dict()
        after_processed: Dict[Source, Document] = dict()

        super().build(before_unprocessed, before_processed)
        super().build(after_unprocessed, after_processed)

        for diff_source in source_set:
            before: Node = BlankNode()
            if diff_source.before:
                before_document = before_processed[diff_source.before]
                del before_processed[diff_source.before]
                before = BeforeNode(before_document.root)

            after: Node = BlankNode()
            if diff_source.after:
                after_document = after_processed[diff_source.after]
                del after_processed[diff_source.after]
                after = AfterNode(after_document.root)

            root = DiffNode(
                diff_source.before_name, before, diff_source.after_name, after
            )
            document = Document(root)
            processed[diff_source] = document


class FixIndexTransform(Transform):
    """
    Replaces `Definition` and `Reference` identifiers within `DiffNode` with
    identifiers specific to the diff.

    Without fixing these identifiers, every Python class would be defined
    multiples times (the actual definition and then again in each diff),
    cluttering up tables of contents.
    """

    def __init__(self, settings: PluginSettings) -> None:
        pass

    def transform(self, context: Context) -> None:
        """
        Apply the transformation to the given document.
        """
        context[Document].root.visit(_FixIndexVisitor())


class _FixIndexVisitor(Visitor):
    diffs: Final[List[DiffNode]]

    def __init__(self) -> None:
        self.diffs = []

    def enter(self, node: Node) -> Visit:
        if isinstance(node, DiffNode):
            self.diffs.append(node)
            return Visit.TraverseChildren

        if not self.diffs:
            return Visit.TraverseChildren

        diff = self.diffs[-1]
        if isinstance(node, (Reference, Definition)):
            before = diff.before_name
            after = diff.after_name

            # TODO: This is not very elegant. Maybe we make custom
            #       EthereumReference and EthereumDefinition types that resolve
            #       differently depending on where the definition is located?
            in_fork = node.identifier.startswith(
                f"{before}."
            ) or node.identifier.startswith(f"{after}.")

            if in_fork:
                node.identifier = f"diff({before},{after}).{node.identifier}"

        return Visit.TraverseChildren

    def exit(self, node: Node) -> None:
        if isinstance(node, DiffNode):
            popped = self.diffs.pop()
            assert popped == node


class MinimizeDiffsTransform(Transform):
    def __init__(self, settings: PluginSettings) -> None:
        pass

    def transform(self, context: Context) -> None:
        """
        Apply the transformation to the given document.
        """
        context[Document].root.visit(_MinimizeDiffsVisitor())


class _CountChildrenVisitor(Visitor):
    stack: Final[List[Node]]
    nodes: Final[Dict[int, int]]

    def __init__(self, nodes: Optional[Dict[int, int]] = None) -> None:
        self.stack = []
        self.nodes = dict() if nodes is None else nodes

    def enter(self, node: Node) -> Visit:
        if id(node) in self.nodes:
            logging.warning("node `%s` appears in tree twice", node)
            return Visit.SkipChildren

        self.nodes[id(node)] = 1
        self.stack.append(node)
        return Visit.TraverseChildren

    def exit(self, node: Node) -> None:
        popped = self.stack.pop()
        assert popped == node

        if self.stack:
            self.nodes[id(self.stack[-1])] += self.nodes[id(node)]


def _node_eq(a: Node, b: Node) -> bool:
    if a is b:
        return True

    if not isinstance(a, type(b)) or not isinstance(b, type(a)):
        return False

    a_members = {k: v for k, v in getmembers(a) if not k.startswith("_")}
    b_members = {k: v for k, v in getmembers(b) if not k.startswith("_")}

    for k, a_v in a_members.items():
        b_v = b_members[k]

        del b_members[k]

        if isfunction(a_v) or ismethod(a_v):
            continue

        if a_v == b_v:
            continue

        if isinstance(a_v, Node) and isinstance(b_v, Node):
            continue

        if isinstance(a_v, (list, tuple)) and isinstance(b_v, (list, tuple)):
            if all(isinstance(x, Node) for x in list(a_v) + list(b_v)):
                continue

        return False

    return not b_members


class _MinimizeDiffsVisitor(Visitor):
    def enter(self, node: Node) -> Visit:
        if not isinstance(node, DiffNode):
            return Visit.TraverseChildren

        before = node.before
        after = node.after

        if before:
            # TODO: Probably should make this work:
            assert isinstance(before, BeforeNode)
            before = before.child

        if after:
            # TODO: Probably should make this work:
            assert isinstance(after, AfterNode)
            after = after.child

        weights: Dict[int, int] = dict()

        before.visit(_CountChildrenVisitor(weights))
        after.visit(_CountChildrenVisitor(weights))

        print(
            node.before_name,
            "has",
            weights[id(before)],
        )

        print(
            node.after_name,
            "has",
            weights[id(after)],
        )

        def update_cost(a: Node, b: Node) -> int:
            if _node_eq(a, b):
                return 0
            return weights[id(a)] + weights[id(b)] + 2

        _, operations = zss.distance(
            before,
            after,
            get_children=lambda n: []
            if isinstance(n, verbatim.Verbatim)
            else tuple(n.children),
            insert_cost=lambda n: weights[id(n)],
            remove_cost=lambda n: weights[id(n)],
            update_cost=update_cost,
            return_operations=True,
        )

        for operation in operations:
            print(operation)

        return Visit.SkipChildren

    def exit(self, node: Node) -> None:
        pass


def render_diff(
    context: object,
    parent: object,
    diff: object,
) -> html.RenderResult:
    """
    Render a DiffNode.
    """
    assert isinstance(diff, DiffNode)
    assert isinstance(parent, (html.HTMLTag, html.HTMLRoot))
    div = html.HTMLTag("div", {"class": "diff", "style": "display: contents;"})
    parent.append(div)
    return div


def render_before(
    context: object,
    parent: object,
    before: object,
) -> html.RenderResult:
    """
    Render a BeforeNode.
    """
    assert isinstance(before, BeforeNode)
    assert isinstance(parent, (html.HTMLTag, html.HTMLRoot))
    div = html.HTMLTag(
        "div", {"class": "diff-before", "style": "display: contents;"}
    )
    parent.append(div)
    return div


def render_after(
    context: object,
    parent: object,
    after: object,
) -> html.RenderResult:
    """
    Render an AfterNode.
    """
    assert isinstance(after, AfterNode)
    assert isinstance(parent, (html.HTMLTag, html.HTMLRoot))
    div = html.HTMLTag(
        "div", {"class": "diff-after", "style": "display: contents;"}
    )
    parent.append(div)
    return div
