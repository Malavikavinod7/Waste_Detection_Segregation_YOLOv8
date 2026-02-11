import albumentations as A

def color_space_transform():
    return A.Compose([
        A.HueSaturationValue(
            hue_shift_limit=8,
            sat_shift_limit=12,
            val_shift_limit=10,
            p=0.5
        ),
        A.RGBShift(
            r_shift_limit=10,
            g_shift_limit=10,
            b_shift_limit=10,
            p=0.3
        ),
    ])
