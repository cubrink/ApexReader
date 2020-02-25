import cv2
import sys
import string
import pytesseract
import numpy as np


class ApexReader():
    """
    Reads information from screenshots of game summaries in Apex Legends
    """
    def __init__(self):
        """
        Initializes an ApexReader
        """
        # Create an object to read information from images
        self.ocr = OCR()

        # Used for reading banner
        # Coordinates are relative to a cropped banner
        self._BANNER_SIZE = (700, 500)
        self._BANNER_INFO_OFFSET = 20
        self._BANNER_INFO_SIZE = (40, 120)
        self._BANNER_NAME_LOC = ((35, 10), (70, 225))

        # Creates a mapping between different attributes and
        # the filepath used to find that attribute onscreen
        self.templates = {'squad_kills': r'.\imgs\squadkills.png',
                          'placement': r'.\imgs\placement.png',
                          'banner': r'.\imgs\banner.png',
                          'kills': r'.\imgs\kills.png',
                          'damage': r'.\imgs\damage.png',
                          'survival_time': r'.\imgs\survival_time.png',
                          'revives': r'.\imgs\revives.png',
                          'respawns': r'.\imgs\respawns.png'}

        # Creates a mapping between different attributes and
        # the OCR function needed to read that attribute
        self.attr_ocr_func = {'squad_kills': self.ocr.read_digits,
                              'placement': self.ocr.read_digits,
                              'banner': None,
                              'kills': self.ocr.read_digits,
                              'damage': self.ocr.read_digits,
                              'survival_time': self.ocr.read_time,
                              'revives': self.ocr.read_digits,
                              'respawns': self.ocr.read_digits,
                              'name': self.ocr.read_text}

        # Attributes attached to each banner
        self.banner_attrs = [ 'name', 'kills', 'damage',
                              'survival_time', 'revives', 'respawns']


        # Initialize values of variables
        self.clear()


    def load(self, filepath):
        """
        Loads a screenshot from the specified filepath

        Args:
            filepath: Filepath to the screenshot to be read
        """
        # Clear state and load new image
        self.clear()
        self.img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
        
        # Create thresholded image
        _, self.thresh = cv2.threshold(self.img, 85, 255, cv2.THRESH_BINARY_INV)

        asp_ratio = round(self.img.shape[1]/self.img.shape[0], 3)

        if asp_ratio != round((16/9), 3):
            print("Warning: Loaded screenshot does not have a 16:9 aspect ratio. ApexReader may not work as intended", file=sys.stderr)
        elif self.img.shape != (1080, 1920):
            print("Warning: Loaded screenshot does not have a resolution of 1920x1080. ApexReader may not work as intended", file=sys.stderr)
            self.thresh = cv2.resize(self.thresh, (1920, 1080), cv2.INTER_AREA)
    


    def clear(self):
        """
        Resets state of the ApexReader 
        """
        self._player_data = dict()
        self._game_data = None
        self._num_players = None
        self._squad_kills = None
        self._placement = None
        self.__banner_coords = None
        
        self.img = None
        self.thresh = None


    def get_player(self, idx):
        """
        Returns all data found about the player with the idx'th banner

        Args:
            idx: Index of player to get data for
        """
        data = None
        if idx not in self._player_data:
            # Get data if not already calculated
            banner = self._get_banner(idx)
            if banner is not None:
                data = dict()
                for attr in self.banner_attrs:
                    # Calculate values for each attribute
                    data[attr] = self._get_info(banner, attr)
            self._player_data[idx] = data
        return self._player_data[idx]


    @property
    def game_data(self):
        """
        Returns a dictionary containing all data found in the screenshot
        """
        if self._game_data is None:
            # Find information for each attribute of game_data
            data = dict()
            data['num_players'] = self.num_players
            data['squad_kills'] = self.squad_kills
            data['placement'] = self.placement
            data['players'] = dict()
            for idx in range(self.num_players):
                data['players'][f'player{idx}'] = self.get_player(idx)
            self._game_data = data
        return self._game_data


    @property
    def num_players(self):
        """
        Returns the number of player found in the screenshot
        """
        if self._num_players is None:
            # Calculate number of players if not already known
            self._num_players = len(self._banner_coords)
        return self._num_players
            

    @property
    def squad_kills(self):
        """
        Returns the squad kills found in the screenshot
        """
        if self._squad_kills is None:
            # Determine amount of squad kills if not already known
            point, shape = self._find_template(self.templates['squad_kills'])
            py, px = point[0]
            h, w = shape

            offset = 10
            read_length = 100
            p1 = (px + w, py)
            p2 = (p1[0] + read_length, p1[1] + h + offset)

            crop = self._crop(p1, p2)
        
            self._squad_kills = self.attr_ocr_func['squad_kills'](crop)
        return self._squad_kills

        
    @property
    def placement(self):
        """
        Returns the placement found in the screenshot
        """
        if self._placement is None:
            # Determine placement of squad if not already known
            point, shape = self._find_template(self.templates['placement'])
            py, px = point[0]
            h, w = shape

            read_length = 140
            p1 = (px + w, py)
            p2 = (p1[0] + read_length, p1[1] + h)

            crop = self._crop(p1, p2)
            padded = self._pad_image(crop)

            self._placement = self.attr_ocr_func['placement'](padded)
        return self._placement


    @property
    def _banner_coords(self):
        """
        Returns a list of locations that banners were found at in the screenshot
        """
        if self.__banner_coords is None:
            # Find locations of banners if not already known
            # Get list of coordinates
            coords, shape = self._find_template(self.templates['banner'])
            # Remove duplicate coordinates
            coords = self._remove_duplicates(coords, shape)
            # Sort by x then y (left to right)
            coords.sort(key = lambda e: (e[1], e[0]))
            self.__banner_coords = coords
        return self.__banner_coords


    def _get_banner(self, index):
        """
        Returns the numpy.ndarray for the banner at the specified index

        Args:
            index: Index of banner to return
        """
        if len(self._banner_coords) <= index or index < 0:
            # Return None if index is invalid
            return None
        else:
            # Initial coords
            y_i, x_i = self._banner_coords[index]
            # Coords plus size of banner
            y_f, x_f = (e1 + e2 for e1, e2 in zip(self._banner_coords[index], self._BANNER_SIZE))
            return self.thresh[y_i:y_f, x_i:x_f]


    def _get_info(self, banner, attr):
        """
        Returns the parsed data for the provided banner for the specified attribute

        Args:
            banner: numpy.ndarray image of banner to search in
            attr: Attribute to search for
        """
        # Get image of attribute in the banner and process it
        ocr_func = self.attr_ocr_func[attr]
        attr_image = self._get_info_image(banner, attr)
        return ocr_func(attr_image)


    def _get_info_image(self, banner, attr):
        """
        Returns an image of the relevant attribute

        Args:
            banner: numpy.ndarray image of banner to search in
            attr: Attribute to search for
        """
        if attr == 'name':
            # Use predfined location of 'name' attribute
            y_i, x_i = self._BANNER_NAME_LOC[0]
            y_f, x_f = self._BANNER_NAME_LOC[1]
        else:
            # Otherwise use templating to find location of attribute
            filename = self.templates[attr]
            # Get best matching coordinates, ignore shape
            coords = self._find_template(filename, banner)[0][0] 
            coords = coords[0] + self._BANNER_INFO_OFFSET, coords[1]
            y_i, x_i = coords
            y_f, x_f = (e1 + e2 for e1, e2 in zip(coords, self._BANNER_INFO_SIZE))
        return self._pad_image(banner[y_i:y_f, x_i:x_f])


    def _find_template(self, filepath, img = None):
        """
        Returns a list of coordinates (in (y, x) format) that match the provided template

        Args:
            filepath: Filepath to template to be used
            img: numpy.ndarray to search for template in
        """
        if img is None:
            # Use self.thresh as default image
            img = self.thresh

        # Read image and match template
        template = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
        template_threshold = 0.8

        # Get values for quality of match of template
        res = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)
        # Filter to locations above set threshold
        loc = np.where(res >= template_threshold)

        # Format to (y, x) coordinate pairs
        coords = [(y, x) for y, x in zip(*loc[0:2])]

        # Return list of coordinates that match template               
        return (coords, template.shape)

   
    def _crop(self, p1, p2):
        """
        Returns a cropped portion of the current thresholded image

        Args:
            p1: Top left pixel (in (x, y) format)
            p2: Bottom right pixel (in (x, y) format)
        """
        return self.thresh[p1[1]:p2[1], p1[0]:p2[0]]


    def _pad_image(self, image, color=[255]):
        """
        Returns the original image padded with empty space

        Args:
            image: numpy.ndarray to be padded
            color: Color used to fill padded space
        """
        return cv2.copyMakeBorder(image, 50, 50, 200, 100, borderType=cv2.BORDER_CONSTANT, value=color)


    def _remove_duplicates(self, coords, shape, min_dist=None):
        """
        Removes near-duplicate points from a list of coordinates

        Args:
            coords: List of coordinates to be processed
            shape: Shape of the template used to produce the coordinates
            min_dist: Minimum distance between any two points to not be removed
        """
        def manhattan_distance(c1, c2):
            """
            Returns distance between two points

            Args:
                c1: First coordinate
                c2: Second coordinate
            """
            return abs(c1[0] - c2[0]) + abs(c1[1] - c2[1])

        if min_dist is None:
            # Rough estimate for minimum distance if none is specified
            min_dist = min(shape)

        # Temporarily hold coordinates known to be unique or duplicates
        good_coords = set()
        bad_coords = set()

        for c1 in coords:
            if c1 in bad_coords:
                continue
            # If not seen before, mark unique
            good_coords.add(c1)
            for c2 in (c for c in coords if c != c1):
                # Compare remaining coordinates, mark bad if they
                # are too close to current unique coordinate
                if manhattan_distance(c1, c2) < min_dist:
                    bad_coords.add(c2)

        # Return a list of good coordinates
        return list(good_coords)
            

        


class OCR():
    """Wrapper around pytesseract to provide specific functionalities
    """
    def __init__(self, tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'):
        """
        Initializes pytesseract with preconfigured settings

        Args:
            tesseract_cmd: Path to Google Tesseract-OCR executable
        """
        pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
        self.text = '-_' + string.ascii_letters + string.digits
        self.digits = string.digits
        self.config_prefix = '--psm 7 -c tessedit_char_whitelist='

    def read_text(self, image):
        """
        Returns text information from an image

        Args:
            image: numpy.ndarray to be read
        """
        config = self.config_prefix + self.text
        return pytesseract.image_to_string(image, config=config)

    def read_digits(self, image):
        """
        Returns digits from an image

        Args:
            image: numpy.ndarray to be read
        """
        config = self.config_prefix + self.digits
        return pytesseract.image_to_string(image, config=config)

    def read_time(self, image):
        """
        Returns time-like information from an image

        Args:
            image: numpy.ndarray to be read
        """
        config = self.config_prefix + self.digits + r':'
        return pytesseract.image_to_string(image, config=config)
