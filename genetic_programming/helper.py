from tree import Node
from function_set import *
from terminal_set import *
from body_set import *
import numpy as np


def generateBody(
    rng: np.random.default_rng,
    environment: dict,
    maxPrimitives: int,
    maxListLength: int,
    currentBodyNo: int = 0,
    maxBodies: int = 5,
) -> Node:
    noPrimitives = rng.integers(1, maxPrimitives + 2)
    bodyChildren = []

    if currentBodyNo == maxBodies:
        return None  # might be needed to return a Node

    for _ in range(noPrimitives):
        choice = rng.choice([1, 2, 3], p=[0.6, 0.3, 0.1])
        if choice == 1:
            attribute = rng.choice(
                [
                    rng.choice(listTerminalInt(maxListLength, rng)),
                    rng.choice(listTerminalFloat(maxListLength, rng)),
                    rng.choice(listTerminalBoolean(rng)),
                ],
                p=[0.6, 0.35, 0.05],
            )
            bodyChildNode = Node("terminal", attribute, None)
            bodyChildren.append(bodyChildNode)
            environment[attribute["returnType"]].append(
                {"id": f"var_{uuid.uuid4().hex}", "attributes": attribute}
            )
        elif choice == 2:
            choice = rng.choice([1, 2], p=[0.8, 0.2])
            if choice == 1:
                attribute = rng.choice(
                    [
                        rng.choice(listFunctionsInt()),
                        rng.choice(listFunctionsFloat()),
                        rng.choice(listFunctions1D()),
                        rng.choice(listFunctionsBoolean()),
                    ]
                )
                bodyChildNode = Node("function", attribute, None)
                bodyChildren.append(bodyChildNode)
        else:
            bodyChildNode = generateBody(
                rng,
                environment,
                maxPrimitives,
                maxListLength,
                currentBodyNo=currentBodyNo + 1,
                maxBodies=maxBodies,
            )
            bodyChildren.append(bodyChildNode)
    return Node(
        "body",
        rng.choice(listFunctionsBody(), p=[0.3, 0.3, 0.3, 0.1]),
        bodyChildren,
    )


def getEnvAttributes(environment, attributeType):
    return [item["attributes"] for item in environment[attributeType]]


def generateParameter(
    rng: np.random.default_rng,
    environment: dict,
    requiredType,
    maxPrimitives: int,
    maxListLength: int,
    maxDepth: int,
    isLeaf: bool,
) -> Node:  # ADD TO ENVIRONMENT
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
                    rng.choice(getEnvAttributes(environment, int)),
                ]
            )
        elif requiredType == float:
            functionAttributes = rng.choice(
                [
                    rng.choice(listTerminalFloat(maxListLength, rng)),
                    rng.choice(listFunctionsFloat()),
                    rng.choice(getEnvAttributes(environment, float)),
                    rng.choice(listTerminalInt(maxListLength, rng)),
                    rng.choice(listFunctionsInt()),
                    rng.choice(getEnvAttributes(environment, int)),
                ]
            )
        elif requiredType == bool:
            functionAttributes = rng.choice(
                [
                    rng.choice(listTerminalBoolean(rng)),
                    rng.choice(listFunctionsBoolean()),
                    rng.choice(getEnvAttributes(environment, bool)),
                ]
            )
        elif requiredType == list:
            functionAttributes = rng.choice(listFunctions1D())
        elif requiredType == "body":
            # functionAttributes = rng.choice(listFunctionsBody(), p=[0.4, 0.3, 0.2, 0.1])
            bodyAttributes = generateBody(
                rng,
                environment,
                maxPrimitives=maxPrimitives,
                maxListLength=maxListLength,
                maxBodies=maxDepth,
            )
            
        else:
            raise ValueError(f"Unsupported requiredType for: {requiredType}")

        return Node(
            nodeType=(
                "function" if functionAttributes["type"] == "function" else "terminal"
            ),
            attributes=functionAttributes,
            children=None,
        )


def generateTree(
    rng: np.random.default_rng,
    environment: dict,
    maxPrimitives: int,
    maxListLength: int,
    isRoot: bool,
    maxBodies: int = 5,
    currentDepth: int = 0,
    maxDepth: int = 10,
) -> Node:
    root = generateBody(
        rng,
        environment,
        maxPrimitives,
        maxListLength,
        maxBodies=maxDepth,
    )
    currentDepth = 0
    while 
    


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


# def generateTree(
#     rng, environment, maxPrimitives, maxBodies, maxListLength, currentDepth, maxDepth
# ):
#     if currentDepth == 0:
#         return generateBody(
#             rng, environment, maxPrimitives, maxListLength, True, maxBodies=maxBodies
#         )
#     elif currentDepth <maxDepth:


# def generateRootNode(rng: np.random.default_rng):
#     rootAttributes = rng.choice(
#         rng.choice(listFunctions1D())
#         + rng.choice(listFunctionsInt())
#         + rng.choice(listFunctionsFloat())
#     )
#     return Node("function", rootAttributes, None)
