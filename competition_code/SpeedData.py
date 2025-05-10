class SpeedData:
    def __init__(
        self, distance_to_section, current_speed, target_speed, recommended_speed, name=0, r=0
    ):
        self.current_speed = current_speed
        self.distance_to_section = distance_to_section
        self.target_speed_at_distance = target_speed
        self.recommended_speed_now = recommended_speed
        self.speed_diff = current_speed - recommended_speed
        self.name = name
        self.r = r

    def __str__(self):
        return f"{self.name} d {self.distance_to_section:.0f} sp {self.recommended_speed_now:.1f} tsp {self.target_speed_at_distance:.1f} r {self.r:.0f}"
