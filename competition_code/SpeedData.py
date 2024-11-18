class SpeedData:
    def __init__(
        self, distance_to_section, current_speed, target_speed, recommended_speed
    ):
        self.current_speed = current_speed
        self.distance_to_corner = distance_to_section
        self.target_speed = target_speed
        self.recommended_speed_now = recommended_speed
        self.speed_diff = current_speed - recommended_speed
