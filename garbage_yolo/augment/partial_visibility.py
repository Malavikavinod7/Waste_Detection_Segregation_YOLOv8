import albumentations as A

def partial_visibility():
    return A.Compose([
        A.CoarseDropout(
            max_holes=3,
            max_height=80,
            max_width=80,
            fill_value=0,
            p=0.5
        ),
        A.RandomShadow(
            shadow_roi=(0, 0.4, 1, 1),
            num_shadows_lower=1,
            num_shadows_upper=2,
            shadow_dimension=5,
            p=0.4
        )
    ])
