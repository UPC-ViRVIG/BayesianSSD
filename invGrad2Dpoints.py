for obj in all_shapes():
    if type(obj) == SimpleObject:
        tan = [obj.svg_get("opt-tx", False), obj.svg_get("opt-ty", False)]
        if tan[0] != None and tan[1] != None:
            obj.svg_set("opt-tx", -tan[0])
            obj.svg_set("opt-ty", -tan[1])
