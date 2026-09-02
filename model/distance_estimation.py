def estimate_distance(box_height):

    if box_height == 0:
        return 0

    focal_length = 700
    real_height = 1.7

    distance = (real_height * focal_length) / box_height

    return round(distance, 2)