def get_scene_relations(detections):
    relations = []
    objects = detections['name'].tolist()
    boxes = detections[['xmin', 'ymin', 'xmax', 'ymax']].values

    for i, obj1 in enumerate(objects):
        for j, obj2 in enumerate(objects):
            if i != j:
                x1, y1, x2, y2 = boxes[i]
                x1b, y1b, x2b, y2b = boxes[j]

                if y2 < y1b:
                    relation = f"{obj1} is above {obj2}"
                elif y1 > y2b:
                    relation = f"{obj1} is below {obj2}"
                elif x2 < x1b:
                    relation = f"{obj1} is left of {obj2}"
                elif x1 > x2b:
                    relation = f"{obj1} is right of {obj2}"
                else:
                    relation = f"{obj1} is near {obj2}"
                relations.append(relation)
    return relations
