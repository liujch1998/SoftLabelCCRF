class Mention:

    def __init__ (self, id, cat, raw, pos):
        self.id = id  # str
        self.cat = cat  # str
        self.raw = raw  # str
        self.pos = pos  # (caption_index, start_pos, end_pos)
        self.bbox = None  # [xmin, ymin, xmax, ymax] float
        self.bbox_pred = None  # [xmin, ymin, xmax, ymax] float

