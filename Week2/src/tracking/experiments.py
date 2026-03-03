# EXPERIMENTS = [
#     {"tracking_model": "overlap", "matching": "hungarian", "min_iou": 0.3, "max_age": 0, "min_confidence": 0.2}, # worse
#     {"tracking_model": "overlap", "matching": "hungarian", "min_iou": 0.3, "max_age": 3, "min_confidence": 0.4}, # best
#     {"tracking_model": "overlap", "matching": "greedy", "min_iou": 0.3, "max_age": 0, "min_confidence": 0.4}, #worse
#     {"tracking_model": "overlap", "matching": "greedy", "min_iou": 0.4, "max_age": 3, "min_confidence": 0.4} #best
# ]

# EXPERIMENTS = [
#     {"tracking_model": "kalman", "matching": "hungarian", "min_iou": 0.5, "max_age": 0, "min_confidence": 0.4}, # worse
#     {"tracking_model": "kalman", "matching": "hungarian", "min_iou": 0.4, "max_age": 3, "min_confidence": 0.3}, # best
#     {"tracking_model": "kalman", "matching": "greedy", "min_iou": 0.5, "max_age": 0, "min_confidence": 0.4}, #worse
#     {"tracking_model": "kalman", "matching": "greedy", "min_iou": 0.5, "max_age": 3, "min_confidence": 0.3} #best
# ]

EXPERIMENTS = []

tracking_models = ["overlap", "kalman"]
matchings = ["greedy", "hungarian"]

min_ious = [0.3, 0.4, 0.5]
max_ages = [0, 1, 3, 5]
min_confidences = [0.2, 0.3, 0.4]

for tm in tracking_models:
    for match in matchings:
        for iou in min_ious:
            for age in max_ages:
                for conf in min_confidences:
                    EXPERIMENTS.append(
                        dict(
                            tracking_model=tm,
                            matching=match,
                            min_iou=iou,
                            max_age=age,
                            min_confidence=conf,
                        )
                    )