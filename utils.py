

def is_same_dim(lst1, lst2) -> bool:
    if not isinstance(lst1, list) or not isinstance(lst2, list):
        return True
    
    if len(lst1) != len(lst2):
        return False

    for sublist1, sublist2 in zip(lst1, lst2):
        if not is_same_dim(sublist1, sublist2):
            return False
    
    return True


def shape(list: list) -> tuple:
    if not isinstance(list, list):
        return ()
    
    return (len(list),) + shape(list[0])