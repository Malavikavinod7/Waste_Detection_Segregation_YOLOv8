import albumentations as A

def texture_emphasis():
    return A.Compose([
        A.Sharpen(alpha=(0.15, 0.35), lightness=(0.8, 1.2), p=0.4),
        A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.3),
    ])
