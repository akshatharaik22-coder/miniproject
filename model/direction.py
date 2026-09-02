def get_direction(center_x, image_width):

    if center_x < image_width / 3:
        return "Left"

    elif center_x > (2 * image_width) / 3:
        return "Right"

    else:
        return "Center"