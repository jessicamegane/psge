"""Royal Tree benchmark fitness function.

The evolutionary engine minimizes fitness.  Royal Tree is conventionally a
maximization problem, so this evaluator returns ``optimal_score - score``.
Consequently, a perfect tree has fitness zero and works with the engine's
existing exact-solution stopping condition.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class _Node:
    """A node in a prefix-encoded Royal Tree phenotype."""

    symbol: str
    children: tuple = ()


class RoyalTree:
    """Evaluate a bounded Royal Tree instance.

    ``level`` selects the desired perfect tree: ``"a"`` through ``"z"``.
    The supplied grammar normally contains ``x`` and functions from ``a`` to
    that level, where function ``a`` is unary, ``b`` is binary, and so on.
    ``y`` and ``z`` are accepted as the original benchmark's distractor
    terminals, even though the bundled grammars use only ``x``.
    """

    def __init__(
        self,
        level="d",
        full_bonus=2.0,
        partial_bonus=1.0,
        penalty=1.0 / 3.0,
        complete_bonus=2.0,
        invalid_fitness=None,
    ):
        level = str(level).lower()
        if len(level) != 1 or not ("a" <= level <= "z"):
            raise ValueError("level must be a single letter from 'a' to 'z'")
        if min(full_bonus, partial_bonus, penalty, complete_bonus) < 0:
            raise ValueError("Royal Tree weights must be non-negative")

        self.level = level
        self.full_bonus = float(full_bonus)
        self.partial_bonus = float(partial_bonus)
        self.penalty = float(penalty)
        self.complete_bonus = float(complete_bonus)
        self.level_number = self._arity(level)
        self.optimal_score = self._perfect_score(self.level_number)
        self.invalid_fitness = (
            self.optimal_score + 1.0
            if invalid_fitness is None else float(invalid_fitness)
        )

    @staticmethod
    def _arity(symbol):
        return ord(symbol) - ord("a") + 1

    def _perfect_score(self, level_number):
        """Return the score of the perfect tree at ``level_number``."""
        score = 1.0  # Score(x)
        for arity in range(1, level_number + 1):
            score *= self.complete_bonus * arity * self.full_bonus
        return score

    def parse(self, phenotype):
        """Parse a grammar phenotype in prefix form into a tree.

        For example, the level-B perfect tree ``B(A(x), A(x))`` is encoded as
        ``"baxax"``.
        """
        if isinstance(phenotype, (list, tuple)):
            phenotype = "".join(phenotype)
        if not isinstance(phenotype, str):
            raise ValueError("Royal Tree phenotype must be a string or token list")

        symbols = "".join(phenotype.split()).lower()
        if not symbols:
            raise ValueError("Royal Tree phenotype is empty")

        position = 0

        def parse_node():
            nonlocal position
            if position >= len(symbols):
                raise ValueError("Royal Tree phenotype ends before a tree is complete")

            symbol = symbols[position]
            position += 1
            if symbol in {"x", "y", "z"}:
                return _Node(symbol)
            if not ("a" <= symbol <= self.level):
                raise ValueError("invalid Royal Tree symbol: %s" % symbol)

            return _Node(
                symbol,
                tuple(parse_node() for _ in range(self._arity(symbol))),
            )

        tree = parse_node()
        if position != len(symbols):
            raise ValueError("Royal Tree phenotype contains trailing symbols")
        return tree

    @staticmethod
    def _tree_depth(tree):
        if not tree.children:
            return 1
        return 1 + max(RoyalTree._tree_depth(child) for child in tree.children)

    def _perfect_level(self, tree):
        """Return a perfect tree's level, or ``None`` when it is imperfect."""
        if tree.symbol in {"x", "y", "z"}:
            return 0 if tree.symbol == "x" else None

        level_number = self._arity(tree.symbol)
        if all(self._perfect_level(child) == level_number - 1
               for child in tree.children):
            return level_number
        return None

    def score(self, phenotype):
        """Return the conventional, maximized Royal Tree raw score."""
        tree = phenotype if isinstance(phenotype, _Node) else self.parse(phenotype)
        if self._tree_depth(tree) > self.level_number + 1:
            raise ValueError(
                "Royal Tree phenotype exceeds the configured level-%s depth"
                % self.level.upper()
            )

        perfect_levels = {}

        def perfect_level(node):
            node_id = id(node)
            if node_id not in perfect_levels:
                perfect_levels[node_id] = self._perfect_level(node)
            return perfect_levels[node_id]

        def score_node(node):
            if not node.children:
                return 1.0

            expected_child_root = (
                "x" if node.symbol == "a"
                else chr(ord(node.symbol) - 1)
            )
            result = 0.0
            for child in node.children:
                if child.symbol != expected_child_root:
                    weight = self.penalty
                elif perfect_level(child) is not None:
                    weight = self.full_bonus
                else:
                    weight = self.partial_bonus
                result += weight * score_node(child)

            if perfect_level(node) is not None:
                result *= self.complete_bonus
            return result

        return score_node(tree)

    def evaluate(self, individual):
        """Return minimization fitness and benchmark details for the engine."""
        try:
            raw_score = self.score(individual)
        except (TypeError, ValueError):
            return self.invalid_fitness, {
                "generation": 0,
                "evals": 1,
                "raw_score": None,
                "optimal_score": self.optimal_score,
                "invalid": True,
            }

        return self.optimal_score - raw_score, {
            "generation": 0,
            "evals": 1,
            "raw_score": raw_score,
            "optimal_score": self.optimal_score,
            "invalid": False,
        }


if __name__ == "__main__":
    import sge

    fitness = RoyalTree(level="d")
    sge.evolutionary_algorithm(
        evaluation_function=fitness,
        parameters_file="parameters/royal_tree_d.yml",
    )
