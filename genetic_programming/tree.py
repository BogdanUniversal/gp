class Node:
    def __init__(self, nodeType, attributes, children):
        self.nodeType = nodeType # 'function' / 'terminal' / 'body'
        self.attributes = attributes
        self.children = children # List of child nodes
        
    # def __repr__(self):
    #     if self.nodeType == 'terminal':
    #         return str(self.value)
    #     return f"{self.value}({', '.join(map(str, self.children))})"
    
    
class Tree:
    def __init__(self, rootNode, SEED):
        self.rootNode = rootNode
        self.SEED = SEED
        self.fitness = 0
        