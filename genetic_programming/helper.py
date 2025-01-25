from tree import Node
from function_set import *
from terminal_set import *
import numpy as np


def generateBody(environment, maxPrimitives, rng: np.random.default_rng, maxListLength): # BUG: recursion can cause stack overflow
    noPrimitives = rng.integers(1, maxPrimitives + 2)
    bodyChildren = []
    for _ in range(noPrimitives):
        choice = rng.choice([1, 2], p=[0.7, 0.3])
        if choice == 1:
            attribute = rng.choice(
                [
                    rng.choice(listTerminalInt(maxListLength, rng)),
                    rng.choice(listTerminalFloat(maxListLength, rng)),
                    rng.choice(listTerminalBoolean(rng)),
                ], p=[0.6, 0.35, 0.05]  
            )
            bodyChildNode = Node("terminal", attribute, None)
            bodyChildren.append(bodyChildNode)
            environment[attribute["returnType"]].append({"id": f"var_{uuid.uuid4().hex}", "attributes": attribute})
        else:
            choice = rng.choice([1, 2], p=[0.9, 0.1])
            if choice == 1:
                attribute = rng.choice(
                    [
                        rng.choice(listFunctionsInt()),
                        rng.choice(listFunctionsFloat()), # ADAUGA IN ENVIRONMENT
                        rng.choice(listFunctions1D()),
                    ]
                )
                bodyChildNode = Node("function", attribute, None)
                bodyChildren.append(bodyChildNode)
            else:
                bodyChildNode = generateBody(
                    environment, maxPrimitives, rng, maxListLength
                )
                bodyChildren.append(bodyChildNode)
    return Node(
        "body",
        {"function": "body", "returnType": "body", "parametersTypes": None},
        bodyChildren,
    )
    
    
def getEnvAttributes(environment, attributeType):
    return [item['attributes'] for item in environment[attributeType]]


def generateFunctionNode(
    environment, requiredType, isLeaf, rng: np.random.default_rng, maxListLength
):
    if isLeaf:
        choice = rng.choice([1, 2])
        if choice == 1:
            if requiredType == int:
                terminalAttributes = rng.choice(listTerminalInt(maxListLength, rng))
            elif requiredType == float:
                terminalAttributes = rng.choice(listTerminalFloat(maxListLength, rng))
            elif requiredType == bool:
                terminalAttributes = rng.choice(listTerminalBoolean(rng))
            else:
                raise ValueError("Unsupported requiredType: {}".format(requiredType))
            return Node("terminal", terminalAttributes, None)
        else:
            terminalAttributes = rng.choice(getEnvAttributes(environment, requiredType))
            return Node("terminal", terminalAttributes, None) 
    else:
        if requiredType == int:
            functionAttributes = rng.choice(
                [
                    rng.choice(listTerminalInt(maxListLength, rng)),
                    rng.choice(listFunctionsInt()),
                    rng.choice(getEnvAttributes(environment, int))
                ]
            )
        elif requiredType == float:
            functionAttributes = rng.choice(
                [
                    rng.choice(listTerminalFloat(maxListLength, rng)),
                    rng.choice(listFunctionsFloat()),
                    rng.choice(getEnvAttributes(environment, float))
                ]
            )
        elif requiredType == bool:
            functionAttributes = rng.choice(
                [
                    rng.choice(listTerminalBoolean(rng)),
                    rng.choice(getEnvAttributes(environment, bool))
                ]
            )
        elif requiredType == list:
            functionAttributes = rng.choice(listFunctions1D())
        else:
            raise ValueError("Unsupported requiredType for: {}".format(requiredType))

        return Node(
            nodeType=(
                "function" if functionAttributes["type"] == "function" else "terminal"
            ),
            attributes=functionAttributes,
            children=None,
        )


# def generateTree(rng, currentDepth, maxDepth, requiredType):
#     # Determine if the node should be a leaf
#     isLeaf = currentDepth >= maxDepth

#     # Generate the node
#     node = generateNode(requiredType, isLeaf, rng)

#     # If it's not a leaf, generate children nodes
#     if not isLeaf and node.nodeType == "function":
#         # For simplicity, assuming functions take two parameters
#         paramTypes = node.attributes["parametersTypes"]
#         node.children = [
#             generateTree(rng, currentDepth + 1, maxDepth, paramType)
#             for paramType in paramTypes
#         ]

#     return node


def generateRootNode(rng: np.random.default_rng):
    rootAttributes = rng.choice(
        rng.choice(listFunctions1D()),
        rng.choice(listFunctionsInt()),
        rng.choice(listFunctionsFloat()),
    )
    rootNode = Node("function", rootAttributes, None)
