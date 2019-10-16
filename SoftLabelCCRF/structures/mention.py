class Mention:

    def __init__ (self, id, cat, pos):
        self.id = id  # str
        self.cat = cat  # str
        self.pos = pos  # (start_pos, end_pos)
        self.bbox = None  # [xmin, ymin, xmax, ymax] float
        self.bbox_pred = None  # [xmin, ymin, xmax, ymax] float

