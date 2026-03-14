class BoxGrainedFilter:
    """
    Evaluates whether the visual ReID feature is trustworthy based on bbox quality.

    TODO: CHECK, THIS IS TEMPTATIVE FOR CREATING THE CODEBASE
    """
    def __init__(self, img_width=1920, img_height=1080, min_area=3000, edge_margin=20):
        self.width = img_width
        self.height = img_height
        self.min_area = min_area
        self.edge_margin = edge_margin

    def is_trustworthy(self, bbox):
        x_min, y_min, x_max, y_max = bbox
        area = (x_max - x_min) * (y_max - y_min)
        
        # Rejects small or distant boxes representing low resolution
        if area < self.min_area:
            return False
            
        # Rejects boxes truncated by the physical camera boundaries
        if (x_min < self.edge_margin or y_min < self.edge_margin or 
            x_max > self.width - self.edge_margin or y_max > self.height - self.edge_margin):
            return False
            
        return True