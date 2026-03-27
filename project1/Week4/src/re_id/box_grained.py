class BoxGrainedFilter:
    """
    Evaluates whether the visual ReID feature is trustworthy based on bbox quality.
    """
    def __init__(self, img_sizes, min_area=1000, edge_margin=20):
        self.img_sizes = img_sizes
        self.min_area = min_area
        self.edge_margin = edge_margin

    def is_trustworthy(self, bbox, c_id):
        x_min, y_min, x_max, y_max = bbox
        area = (x_max - x_min) * (y_max - y_min)
        
        # Rejects small or distant boxes representing low resolution
        if area < self.min_area:
            return False
            
        # Rejects boxes truncated by the physical camera boundaries
        if (x_min < self.edge_margin or y_min < self.edge_margin or 
            x_max > self.img_sizes[c_id]["width"] - self.edge_margin or y_max > self.img_sizes[c_id]["height"] - self.edge_margin):
            return False
            
        return True