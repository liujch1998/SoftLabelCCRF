class Region:

    def __init__ (self, bbox, region_):
        self.bbox = bbox  # [xmin, ymin, xmax, ymax] float
        self.region_ = region_  # (1, d_region)

