"""General-purpose strong-type GP tree class"""
import math
import random


class GeneticTree:
    """
    A class for representing a tree in a GP program. This class is designed
    to be subclassed to provide a specific implementation of a GP tree
    for a particular problem domain. The subclass should define a set of
    primitives and their associated metadata using the declare_primitive
    """

    primitives = {}
    transitives = {}
    literal_initializers = {}
    DEBUG = False
    local = {}

    @classmethod
    def declare_primitive(
        cls,
        roles,
        output_type,
        input_types,
        *args,
        transitive=False,
        literal_init=False,
        **kwargs,
    ):
        """
        Defines a decorator that can be used to conviently define primitives.
        Primitives have the notion of RPG-style "roles" that can be used to
        assign classes of capabilities to trees of different species. Primitives
        also have output type and expected input type of children stored as
        well. This decorator captures the function of the primitive and all
        associated metadata in the primitives dictionary of the GeneticTree
        class for programatic use by GeneticTree and Node objects at runtime.

        This approach represents a modified approach to a decorator-based
        primitive scheme implemented by Sean Harris. The primary modification
        made here is the addition of the roles mechanism to avoid needing to
        create multiple files of primitives with large amounts of
        duplicate content. This implementation also requires the treatment
        of GP literal nodes in a general manner instead of implementing a
        separate decorator for literals as was done in Sean's implementation.

        Args:
            roles: A string or tuple of strings representing the roles that
            output_type: The output type of the primitive
            input_types: A tuple of input types for the primitive
            transitive: A boolean indicating whether the primitive is transitive
            literal_init: A boolean indicating whether the primitive is a literal
            args: A tuple of positional arguments to be passed to the primitive
            kwargs: A dictionary of keyword arguments to be passed to the primitive

        """
        if isinstance(roles, str):
            roles = (roles,)  # turns the string into a single-element tuple

        def add_primitive(func):
            """
            The decorator function that captures the function and associated
            metadata in the primitives dictionary of the GeneticTree class.
            """
            for role in roles:
                if role not in cls.primitives:
                    cls.primitives[role] = set()
                    cls.transitives[role] = set()
                    cls.literal_initializers[role] = {}
                    cls.local[role] = {}
                cls.primitives[role].add((func, output_type, input_types))
                cls.local[role][func.__name__] = func
                if transitive and len(set(input_types)) == 1:
                    cls.transitives[role].add((func, output_type, input_types))
                if literal_init:
                    key = (func.__name__, output_type, input_types)
                    cls.literal_initializers[role][key] = (args, kwargs)
                if cls.DEBUG:
                    print(
                        f"importing primitive '{func.__name__}' of type "
                        + f"{output_type} for role '{role}'",
                        func,
                    )
            return func

        return add_primitive

    def __init__(self, roles, output_type):
        """
        Initializes a tree object with a set of primitives appropriate for
        the roles assigned to the object and a root node of the desired output
        type.
        """
        self.primitive_set = set()
        self.init_dict = {}
        self.local = {}
        if isinstance(roles, str):
            roles = (roles,)  # turns the string into a single-element tuple
        self.roles = roles
        for role in self.roles:
            if role not in self.__class__.primitives:
                print(f"encountered unknown role: {role}")
            else:
                self.primitive_set |= self.__class__.primitives[role]
                self.init_dict.update(self.__class__.literal_initializers[role])
                self.local.update(self.__class__.local[role])
        assert len(self.primitive_set) > 0, "No valid roles used in tree declaration"
        self.root = Node(output_type)
        self.branching_factor = max(
            len(primitive[2]) for primitive in self.primitive_set
        )
        self.node_tags = None
        self.depth_limit = 0
        self.hard_limit = 0
        self.depth = 0
        self.size = 0
        self.func = None
        self.fitness=None

    def initialize(self, depth=1, hard_limit=0, grow=False, leaf_prob=0.5, full=False):
        """
        Performs tree initialization in the GP sense to the calling tree object

        Args:
            depth: The depth of the tree to initialize
            hard_limit: The maximum depth of the tree to initialize
            grow: A boolean indicating whether to use the grow initialization method
            leaf_prob: The probability of initializing a leaf node in the grow method
            full: A boolean indicating whether to use the full initialization method
        """
        if grow:
            self.grow(depth, leaf_prob)
        elif full:
            self.full(depth)
        self.depth_limit = depth
        if hard_limit < depth:
            self.hard_limit = depth * 2
        else:
            self.hard_limit = hard_limit
        self.root.initialize(self.init_dict)
        self.node_tags = self.root.get_tags(self.branching_factor)
        self.depth = math.ceil(
            math.log(max(list(self.node_tags.keys())), self.branching_factor)
        )
        self.size = len(self.node_tags)
        self.string = self.print_tree()
        # self.build()

    def build(self):
        """
        Builds the calling tree object into a callable function
        """
        local = {}
        for role in self.roles:
            if role not in self.__class__.primitives:
                print(f"encountered unknown role: {role}")
            else:
                local.update(self.__class__.local[role])
        self.func = eval("".join(["lambda context: ", self.string]), local)

    def clean(self):
        """
        Cleans up the calling tree object by deleting the callable function
        """
        if self.func is not None:
            del self.func
        self.func = None

    # Full initialization method
    def full(self, depth=1):
        """
        Performs full initialization on the calling tree object

        Args:
            depth: The depth of the tree to initialize
        """
        self.root.full(self.primitive_set, depth - 1)

    # Grow initialization method
    def grow(self, depth=1, leaf_prob=0.5):
        """
        Performs grow initialization on the calling tree object

        Args:
            depth: The depth of the tree to initialize
            leaf_prob: The probability of initializing a leaf node
        """
        self.root.grow(self.primitive_set, depth - 1, leaf_prob, reach_depth=True)

    # Execute/evaluate the calling tree
    def execute(self, context):
        """
        Executes the calling tree object

        Args:
            context: A dictionary of context variables to be passed to the tree

        Returns:
            The result of the tree execution
        """
        if self.func is None:
            self.build()
        return self.func(context)

    # Return a copy of the calling tree
    def copy(self):
        """
        Returns a copy of the calling tree object
        """
        clone = self.__class__(self.roles, self.root.type)
        clone.root = self.root.copy()
        clone.initialize(self.depth_limit, self.hard_limit)
        return clone

    # Random subtree mutation - intended to be called by a copy of a parent
    def subtree_mutation(self):
        """
        Performs a subtree mutation on the calling tree object
        """
        for _ in range(10):
            target = random.choice(list(self.node_tags.keys()))
            mutant_depth_limit = self.hard_limit - math.ceil(
                math.log(target, self.branching_factor)
            )
            if mutant_depth_limit <= 0:
                depth = 0
            else:
                depth = random.randrange(0, mutant_depth_limit)

            if self.root.find_tag(target, self.branching_factor).mutate(
                self.primitive_set, depth
            ):
                break  # break on successful mutation

        else:  # else of for loop calls grow on a random node if all other mutation attempts fail
            target = random.choice(list(self.node_tags.keys()))
            mutant_depth_limit = self.hard_limit - math.ceil(
                math.log(target, self.branching_factor)
            )
            if mutant_depth_limit <= 0:
                depth = 0
            else:
                depth = random.randrange(0, mutant_depth_limit)
            self.root.find_tag(target, self.branching_factor).grow(
                self.primitive_set, depth
            )
        self.initialize(self.depth_limit, self.hard_limit)

    # Random subtree recombination - intended to be called by a copy of a parent
    def subtree_recombination(self, mate):
        """
        Performs a subtree recombination on the calling tree object

        Args:
            mate: The mate tree object to recombine with
        """
        type_options = set(self.node_tags.values()) & set(mate.node_tags.values())
        if len(type_options) == 0:
            print("No matching types for crossover!")
        local_tag = random.choice(
            [key for key in self.node_tags if self.node_tags[key] in type_options]
        )
        mate_tag = random.choice(
            [
                key
                for key in mate.node_tags
                if mate.node_tags[key] == self.node_tags[local_tag]
            ]
        )
        self.root.assign_at_tag(
            local_tag,
            mate.root.find_tag(mate_tag, mate.branching_factor).copy(),
            self.branching_factor,
        )
        self.initialize(self.depth_limit, self.hard_limit)

    # Returns a string representation of the expression encoded by the GP tree
    def print_tree(self):
        """
        Returns a string representation of the calling tree object

        Returns:
            A string representation of the calling tree object
        """
        return self.root.print_tree()

    def to_dict(self):
        """
        Returns a dictionary representation of the calling tree object

        Returns:
            A dictionary representation of the calling tree object
        """
        return {
            "roles": self.roles,
            "output_type": self.root.type,
            "depthLimit": self.depth_limit,
            "hardLimit": self.hard_limit,
            "root": self.root.to_dict(),
        }

    @classmethod
    def from_dict(cls, _dict):
        """
        Returns a calling tree object from a dictionary representation

        Args:
            _dict: The dictionary representation of the calling tree object

        Returns:
            A calling tree object
        """
        genotype = cls(_dict["roles"], _dict["output_type"])
        genotype.root.from_dict(genotype.primitive_set, _dict["root"])
        genotype.initialize(_dict["depthLimit"], _dict["hardLimit"])
        return genotype


class Node:
    """General-purpose strong-typed GP node class"""

    def __init__(self, output_type):
        """
        Args:
            output_type: The type of the output of the node
        """
        self.type = output_type
        self.func = None  # stores an executable function object
        self.children = []
        self.value = None

    def initialize(self, init_dict):
        """
        Initializes the calling node object

        Args:
            init_dict: A dictionary of pre-initialized values
        """

        key = (
            self.func.__name__,
            self.type,
            tuple([child.type for child in self.children]),
        )
        if self.value == None and key in init_dict:
            args, kwargs = init_dict[key]
            self.value = self.func(*args, **kwargs)
        for child in self.children:
            child.initialize(init_dict)

    def filter_type_primitives(self, primitives):
        """
        Accepts the list of available primitives and filters into leaf and
        internal primitives of acceptable type

        Args:
            primitives: The list of available primitives

        Returns:
            A tuple of lists of leaf and internal primitives of acceptable type
        """
        options = [primitive for primitive in primitives if self.type == primitive[1]]
        if not options:
            print(f"type {self.type} not found in primitives")
            exit()
        leaves = [leaf for leaf in options if leaf[2] == ()]
        internals = [internal for internal in options if internal[2] != ()]
        return leaves, internals

    def full(self, primitives, limit=0):
        """
        Performs a full initialization on the calling node object

        Args:
            primitives: The list of available primitives
            limit: The depth limit of the subtree
        """
        self.value = None
        if self.children:
            self.children.clear()
        leaves, internals = self.filter_type_primitives(primitives)

        if limit > 0 and internals:
            self.func, _, input_types = random.choice(internals)
            self.children = [Node(childType) for childType in input_types]
        else:
            self.func, _, _ = random.choice(leaves)

        for child in self.children:
            child.full(primitives, limit - 1)

    # Grow initialization method for subtree generation
    def grow(self, primitives, limit=0, leaf_prob=0.5, reach_depth=False):
        """
        Performs a grow initialization on the calling node object

        Args:
            primitives: The list of available primitives
            limit: The depth limit of the subtree
            leaf_prob: The probability of selecting a leaf primitive
            reach_depth: Whether to reach the depth limit
        """
        self.value = None
        if self.children:
            self.children.clear()
        leaves, internals = self.filter_type_primitives(primitives)

        if limit > 0 and internals and (reach_depth or random.random() > leaf_prob):
            self.func, _, input_types = random.choice(internals)
            self.children = [Node(childType) for childType in input_types]
        else:
            self.func, _, _ = random.choice(leaves)

        if reach_depth and self.children:
            branch = random.choice(range(len(self.children)))
        else:
            branch = -1
        for i, child in enumerate(self.children):
            if i != branch:
                child.grow(primitives, limit - 1)
            else:
                child.grow(primitives, limit - 1, reach_depth=True)

    def mutate(self, primitives, limit=0):
        """
        Performs a mutation on the calling node object

        Args:
            primitives: The list of available primitives
            limit: The depth limit of the subtree

        Returns:
            A boolean indicating whether the mutation was successful
        """
        if self.func is None:
            self.grow(primitives, limit)
            return True
        name = self.func.__name__
        primitive = None
        leaves, internals = self.filter_type_primitives(primitives)
        if self.children == []:
            options = leaves
        else:
            options = internals

        for option in options:
            if option[0].__name__ == name:
                primitive = option
                break

        if primitive is None:
            return False

        if self.value is None:
            options.remove(primitive)
        options = [option for option in options if option[2] == primitive[2]]
        if len(options) == 0:
            return False

        self.value = None
        self.func, _, _ = random.choice(options)
        return True

    # Execute/evaluate the subtree of the calling node as root
    # def execute(self, context):
    # 	return self.func(self, self.children, context)

    def copy(self):
        """
        Return a copy of the subtree of the calling node

        Returns:
            A copy of the calling node object
        """
        clone = Node(self.type)
        clone.func = self.func
        clone.value = self.value
        clone.children = [child.copy() for child in self.children]
        return clone

    def get_tags(self, branching, index=1):
        """
        Generate dictionary of unique node ID tags and node types
        Args:
            branching: The branching factor of the tree
            index: The ID tag of the calling node

        Returns:
            A dictionary of unique node ID tags and node types
        """
        tags = {}
        tags[index] = self.type
        for child_index, child in enumerate(self.children):
            tags.update(child.get_tags(branching, (index * branching) + child_index))
        return tags

    def find_tag(self, target, branching, index=1):
        """
        Get the node with the input ID tag

        Args:
            target: The ID tag of the target node
            branching: The branching factor of the tree
            index: The ID tag of the calling node

        Returns:
            The node with the input ID tag
        """
        if target == index:
            return self
        if index > target:
            return None

        for childIndex in range(len(self.children)):
            tag = (index * branching) + childIndex
            if tag > target:
                break
            elif tag == target:
                return self.children[childIndex]
            childResult = self.children[childIndex].find_tag(target, branching, tag)
            if childResult is not None:
                return childResult
        if index <= 1:
            print(
                f"invalid/missing target: {target}"
            )  # TODO: make into a proper robust error
        return None

    def assign_at_tag(self, target, payload_node, branching, index=1):
        """
        Modify node at the input ID tag

        Args:
            target: The ID tag of the target node
            payload_node: The node to be inserted
            branching: The branching factor of the tree
            index: The ID tag of the calling node

        Returns:
            A boolean indicating whether the assignment was successful
        """

        if target == index:
            self.type = payload_node.type
            self.func = payload_node.func
            self.value = payload_node.value
            self.children = payload_node.children[:]
            return True

        for child_index, child in enumerate(self.children):
            tag = (index * branching) + child_index
            if tag > target:
                return False
            if tag == target:
                self.children[child_index] = payload_node
                return True
            if child.assign_at_tag(target, payload_node, branching, tag):
                return True
        if index <= 1 and target != index:
            print(
                f"invalid/missing target: {target}"
            )  # TODO: make into a proper robust error
        return False

    def print_tree(self):
        """
        Returns a string representation of the subtree of the calling node for debugging purposes

        Returns:
            A string representation of the subtree of the calling node
        """
        if self.value is not None:
            name = f"{self.value}"
        else:
            name = self.func.__name__
        if self.value is not None and not self.children:
            return name
        if not self.children:
            return f"{name}(context)"

        child_strings = []
        num_children = len(self.children)
        for i in range(num_children):
            child_strings.append(self.children[i].print_tree())
            if i < num_children - 1:
                child_strings.append(",")
        return f'{name}({"".join(child_strings)})'

    def to_dict(self):
        """
        Returns a dictionary representation of the subtree of the calling node

        Returns:
            A dictionary representation of the subtree of the calling node
        """
        return {
            "func": self.func.__name__,
            "type": self.type,
            "value": self.value,
            "children": [child.to_dict() for child in self.children],
        }

    def from_dict(self, primitives, _dict):
        """
        Reconstructs the subtree of the calling node from a dictionary representation

        Args:
            primitives: A list of tuples of the form (function, return type, list of argument types)
            _dict: A dictionary representation of the subtree of the calling node
        """
        self.value = _dict["value"]
        self.type = _dict["type"]
        leaves, internals = self.filter_type_primitives(primitives)
        if _dict["children"] == []:
            options = leaves
            children = tuple()
        else:
            options = internals
            children = tuple([child["type"] for child in _dict["children"]])

        for option in options:
            if option[0].__name__ == _dict["func"] and children == option[2]:
                self.func = option[0]
                break
        else:
            assert (
                False
            ), f'Function {_dict["func"]} of type {_dict["type"]} with children {children} could not be found in options\n{options}'
        self.children.clear()
        for child in _dict["children"]:
            self.children.append(Node(child["type"]))
            self.children[-1].from_dict(primitives, child)
        # self.children = [Node(child['type']).fromDict(primitives, child) for child in _dict['children']]


def main():
    """
    Testbench for debugging - only called if you were to run this file directly
    """
    GENERAL = "General"
    PREY = "Prey"
    PREDATOR = "Predator"
    ANGLE = "Angle"
    DISTANCE = "Distance"

    @GeneticTree.declare_primitive((GENERAL, PREY), ANGLE, ())
    def dummy():
        print("Dummy")

    @GeneticTree.declare_primitive((GENERAL, PREY), ANGLE, (ANGLE, ANGLE))
    def thicc():
        print("thicc")

    # @GeneticTree.declarePrimitive((GENERAL, PREY), (ANGLE,), (ANGLE, ANGLE, ANGLE, ANGLE))
    # def thiccc():
    # print("thiccc")
    print(repr(GeneticTree.primitives))
    fullTree = GeneticTree(PREY, ANGLE)
    fullTree.initialize(4, full=True)
    print("full nodes: " + repr(fullTree.node_tags))
    print(f"levels: {fullTree.depth}")
    growTree = GeneticTree(GENERAL, ANGLE)
    growTree.initialize(5, grow=True)
    print("grow nodes: " + repr(growTree.node_tags))
    growTree.subtree_mutation()
    print("mutated grow nodes: " + repr(growTree.node_tags))
    growTree.subtree_recombination(fullTree)
    print("recombination grow nodes: " + repr(growTree.node_tags))


if __name__ == "__main__":
    main()
