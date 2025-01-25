import numpy as np
import random

# Assuming these lists are predefined
listTerminalInt = [1, 2, 3, 4, 5]
listTerminalFloat = [1.1, 2.2, 3.3, 4.4, 5.5]
listFunctionsInt = [{"type": "function", "name": "sumElementsInt", "parametersTypes": [int, int]},
                    {"type": "function", "name": "multiplyElementsInt", "parametersTypes": [int, int]}]
listFunctionsFloat = [{"type": "function", "name": "sumElements", "parametersTypes": [float, float]},
                      {"type": "function", "name": "multiplyElements", "parametersTypes": [float, float]}]
listFunctions1D = [{"type": "function", "name": "swap1D", "parametersTypes": [list, int, int]},
                   {"type": "function", "name": "sort1D", "parametersTypes": [list, bool]},
                   {"type": "function", "name": "flip1D", "parametersTypes": [list]}]

# Define Node class
class Node:
    def __init__(self, nodeType, attributes, children=None):
        self.nodeType = nodeType
        self.attributes = attributes
        self.children = children or []

    def __repr__(self):
        return f"Node(type={self.nodeType}, attributes={self.attributes}, children={self.children})"

# Generate a node based on the required type
def generateNode(requiredType, isLeaf, rng):
    if isLeaf:
        if requiredType == int:
            terminalAttributes = rng.choice(listTerminalInt)
        elif requiredType == float:
            terminalAttributes = rng.choice(listTerminalFloat)
        else:
            raise ValueError(f"Unsupported requiredType for terminal: {requiredType}")
        return Node("terminal", terminalAttributes, None)
    else:
        if requiredType == int:
            functionAttributes = rng.choice(listFunctionsInt)
        elif requiredType == float:
            functionAttributes = rng.choice(listFunctionsFloat)
        elif requiredType == list:
            functionAttributes = rng.choice(listFunctions1D)
        else:
            raise ValueError(f"Unsupported requiredType for function: {requiredType}")

        return Node(
            nodeType="function" if functionAttributes["type"] == "function" else "terminal",
            attributes=functionAttributes,
            children=None,
        )

# Iteratively generate a tree
def generateTreeIterative(requiredType, maxDepth, rng):
    root = generateNode(requiredType, isLeaf=maxDepth == 1, rng=rng)
    stack = [(root, 1)]  # Stack to hold nodes and their current depth

    while stack:
        currentNode, currentDepth = stack.pop()

        if currentNode.nodeType == "function" and currentDepth < maxDepth:
            paramTypes = currentNode.attributes["parametersTypes"]
            for paramType in paramTypes:
                isLeaf = currentDepth + 1 >= maxDepth
                childNode = generateNode(paramType, isLeaf, rng)
                currentNode.children.append(childNode)
                if not isLeaf:
                    stack.append((childNode, currentDepth + 1))

    return root

# Define sorting functions
def swap1D(array, pos1, pos2):
    try:
        arrayCopy = array.copy()
        arrayCopy[pos1], arrayCopy[pos2] = arrayCopy[pos2], arrayCopy[pos1]
        return arrayCopy
    except IndexError:
        print(f"Error: Indices {pos1} and/or {pos2} are out of bounds.")
        return array  # Return the original array unchanged in case of error

def sort1D(array, reverse):
    return np.sort(array).tolist()[::-1] if reverse else np.sort(array).tolist()

def flip1D(array):
    return np.flip(array).tolist()

# Evaluation function to apply tree operations to the list
def evaluateTree(tree, data):
    if tree.nodeType == "terminal":
        return data
    elif tree.nodeType == "function":
        if tree.attributes["name"] == "swap1D":
            pos1 = random.randint(0, len(data) - 1)
            pos2 = random.randint(0, len(data) - 1)
            data = swap1D(data, pos1, pos2)
        elif tree.attributes["name"] == "sort1D":
            reverse = random.choice([True, False])
            data = sort1D(data, reverse)
        elif tree.attributes["name"] == "flip1D":
            data = flip1D(data)
        for child in tree.children:
            data = evaluateTree(child, data)
    return data

# Example usage
rng = np.random.default_rng()

# Generate a tree with the desired depth
maxDepth = 3
tree = generateTreeIterative(list, maxDepth, rng)

# Unsorted input list
unsorted_list = [5, 3, 8, 4, 2, 7, 1, 6]

# Apply tree operations to sort the list
sorted_list = evaluateTree(tree, unsorted_list)
print(f"Unsorted List: {unsorted_list}")
print(f"Sorted List: {sorted_list}")
