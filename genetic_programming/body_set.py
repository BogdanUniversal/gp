def listFunctionsBody():
    return [
        {
            "type": "body",
            "function": "body",
            "returnType": "body",
            "parametersTypes": None,
        },
        {
            "type": "body",
            "function": "ifElse",
            "returnType": "body",
            "parametersTypes": [bool, "body", "body"],
        },
        {
            "type": "body",
            "function": "forLoop",
            "returnType": "body",
            "parametersTypes": [int, int, int, "body"],
        },
        {
            "type": "body",
            "function": "whileLoop",
            "returnType": "body",
            "parametersTypes": [bool, "body"],
        },
    ]


################################################## NOTE return body ##################################################

# def forLoop(body, start, stop, step):
#     for index in range(start, stop, step):
#         return


# def ifElse(condition, ifBody, elseBody):
#     return ifBody() if condition else elseBody()
