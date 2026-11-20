"""Deceptive Max (DMax) benchmark fitness function.

The benchmark maximizes the real part of a complex expression.  PSGE minimizes
fitness, so ``evaluate`` returns ``optimal_real_score - real_score`` and gives
the known optimum fitness zero.
"""

import cmath
import math
from dataclasses import dataclass


@dataclass(frozen=True)
class _Node:
    """A node in a prefix-encoded DMax phenotype."""

    symbol: str
    children: tuple = ()


class DeceptiveMax:
    """Evaluate the m-ary Deceptive Max benchmark.

    The conventional instance has ``arity=5``, ``root_order=3`` and
    ``max_depth=3``.  Its functions are m-ary ``+`` and ``*``; its terminals
    are 0.95 and lambda, a primitive root of unity.  Only the real component
    of the resulting complex value is the raw benchmark fitness.
    """

    def __init__(
        self,
        arity=5,
        root_order=3,
        max_depth=3,
        constant=0.95,
        invalid_fitness=None,
    ):
        self.arity = int(arity)
        self.root_order = int(root_order)
        self.max_depth = int(max_depth)
        self.constant = float(constant)
        if self.arity < 2:
            raise ValueError("arity must be at least two")
        if self.root_order < 2:
            raise ValueError("root_order must be at least two")
        if self.max_depth < 2:
            raise ValueError("max_depth must be at least two")
        if not 0 < self.constant < 1:
            raise ValueError("constant must be strictly between zero and one")

        self.lambda_value = cmath.exp(2j * math.pi / self.root_order)
        self.optimal_score = self._optimal_score()
        self.invalid_fitness = (
            2 * self._magnitude_upper_bound() + 1
            if invalid_fitness is None else float(invalid_fitness)
        )

    def _optimal_score(self):
        """Compute the known optimum under the benchmark depth limit.

        At the final internal level there are ``arity ** (depth - 2)`` sums.
        To make the product real and positive, the number of lambda sums must
        be divisible by ``root_order``.  Using as many lambda sums as possible
        maximizes the magnitude because lambda has modulus one whereas the
        other terminal is smaller than one.
        """
        number_of_sums = self.arity ** (self.max_depth - 2)
        lambda_sums = number_of_sums - (number_of_sums % self.root_order)
        constant_sums = number_of_sums - lambda_sums
        return float(
            self.arity ** number_of_sums
            * self.constant ** constant_sums
        )

    def _magnitude_upper_bound(self):
        """Bound the magnitude of any tree allowed by ``max_depth``."""
        bound = float(self.arity)
        for _ in range(self.max_depth - 2):
            bound **= self.arity
        return bound

    def parse(self, phenotype):
        """Parse a DMax prefix phenotype.

        The bundled grammar emits ASCII ``lambda``.  The Greek symbol ``λ`` is
        also accepted for direct use of the notation from the literature.
        """
        if isinstance(phenotype, (list, tuple)):
            phenotype = "".join(phenotype)
        if not isinstance(phenotype, str):
            raise ValueError("DMax phenotype must be a string or token list")

        symbols = "".join(phenotype.split())
        if not symbols:
            raise ValueError("DMax phenotype is empty")

        constant_token = str(self.constant)
        position = 0

        def parse_node():
            nonlocal position
            if position >= len(symbols):
                raise ValueError("DMax phenotype ends before a tree is complete")
            if symbols.startswith(constant_token, position):
                position += len(constant_token)
                return _Node("constant")
            if symbols.startswith("lambda", position):
                position += len("lambda")
                return _Node("lambda")
            if symbols[position] == "λ":
                position += 1
                return _Node("lambda")

            operator = symbols[position]
            if operator not in {"+", "*"}:
                raise ValueError("invalid DMax symbol: %s" % operator)
            position += 1
            return _Node(
                operator,
                tuple(parse_node() for _ in range(self.arity)),
            )

        tree = parse_node()
        if position != len(symbols):
            raise ValueError("DMax phenotype contains trailing symbols")
        return tree

    @staticmethod
    def _tree_depth(tree):
        if not tree.children:
            return 1
        return 1 + max(DeceptiveMax._tree_depth(child) for child in tree.children)

    def value(self, phenotype):
        """Return the full complex value of a valid, depth-bounded phenotype."""
        tree = phenotype if isinstance(phenotype, _Node) else self.parse(phenotype)
        if self._tree_depth(tree) > self.max_depth:
            raise ValueError("DMax phenotype exceeds the configured maximum depth")

        def evaluate_node(node):
            if node.symbol == "constant":
                return complex(self.constant)
            if node.symbol == "lambda":
                return self.lambda_value

            values = [evaluate_node(child) for child in node.children]
            if node.symbol == "+":
                return sum(values)

            product = complex(1)
            for child_value in values:
                product *= child_value
            return product

        return evaluate_node(tree)

    def raw_fitness(self, phenotype):
        """Return the conventional maximized fitness: ``Re(value)``."""
        return self.value(phenotype).real

    def evaluate(self, individual):
        """Return PSGE minimization fitness and DMax evaluation details."""
        try:
            raw_score = self.raw_fitness(individual)
        except (TypeError, ValueError, OverflowError):
            return self.invalid_fitness, {
                "generation": 0,
                "evals": 1,
                "raw_score": None,
                "optimal_score": self.optimal_score,
                "invalid": True,
            }

        fitness = self.optimal_score - raw_score
        # Roots of unity are represented numerically, so recognize the known
        # optimum despite harmless floating-point roundoff.
        if math.isclose(raw_score, self.optimal_score, rel_tol=1e-12,
                        abs_tol=1e-12):
            fitness = 0.0
        return fitness, {
            "generation": 0,
            "evals": 1,
            "raw_score": raw_score,
            "optimal_score": self.optimal_score,
            "invalid": False,
        }


if __name__ == "__main__":
    import sge

    fitness = DeceptiveMax(max_depth=3)
    sge.evolutionary_algorithm(
        evaluation_function=fitness,
        parameters_file="parameters/deceptive_max_d3.yml",
    )
