class GaussianSceneDivision:
    def __init__(self,VisibilityMap):
        self.VisibilityMap=VisibilityMap
        return
    
    def division(self):
        GroupA,GroupB,GroupIntersection=GaussianSceneDivision.__division_internel(self.VisibilityMap)
        return
    
    def __division_internel(VisibilityMap:dict):
        keys=list(VisibilityMap.keys())

        #sort len
        length_dict={}
        for key in keys:
            length_dict[key]=len(VisibilityMap[key])
        sorted(length_dict.items(), key = lambda kv:(kv[1], kv[0]))
        keys=list(length_dict.keys())
        keys.reverse()

        #division
        GroupA=set(VisibilityMap[keys[0]])
        GroupB=set(VisibilityMap[keys[1]])
        UnionAB=GroupA.union(GroupB)
        intersectionAB=GroupA.intersection(GroupB)
        for view_id in keys[2:]:
            addition=set(VisibilityMap[view_id])
            A_intersection=GroupA.intersection(GroupB.union(addition))
            B_intersection=GroupB.intersection(GroupA.union(addition))
            AB_intersection=UnionAB.intersection(addition)
            if len(A_intersection)<=len(B_intersection) and len(A_intersection)<=len(AB_intersection):
                GroupB=GroupB.union(addition)
            elif len(B_intersection)<len(A_intersection) and len(B_intersection)<=len(AB_intersection):
                GroupA=GroupA.union(addition)
            else:
                GroupA=addition
                GroupB=UnionAB
            UnionAB=GroupA.union(GroupB)
            intersectionAB=GroupA.intersection(GroupB)
        return