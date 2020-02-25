# ApexReader
Analyze screenshots Apex Legends game summaries using Google's Tesseract OCR

## Details
Uses Google's [Tesseract OCR](https://opensource.google/projects/tesseract) in conjuction with template matching via the [opencv-python](https://pypi.org/project/opencv-python/) library. Parsing a screenshot follows this general pattern:
1. Image is loaded from filepath via the `cv2` module and converted to grayscale
2. Loaded image undergoes [thresholding](https://en.wikipedia.org/wiki/Thresholding_(image_processing)) to aid in parsing
3. When a specific attribute is requested to be parsed, the appropriate template image is selected
4. The selected template is found on screen using template matching, returning coordinates of the template image
5. The location of the template is used to determine the region onscreen of the desired attribute information
6. A cropped image containing the attribute information is finally passed to the `OCR` module
7. The result of the OCR processing is saved for the attribute

Additionally, it's worth noting that attributed aren't calculated until requested. This allows for the `ApexReader` module to feel more responsive, as the entire image isn't parsed upon loading an image. This is done by using the `@property` decorator to allow properties to be calculated on their first use, instead of on initialization.

## Example usage:
Let's extract the information from the following screenshot
<img src="https://github.com/cubrink/ApexReader/blob/master/test_imgs/img4.png" width="75%" height="75%">

We can do this simply with the following script

```python3
from ApexReader import ApexReader
from pprint import pprint

ar = ApexReader()
ar.load(r'.\test_imgs\img4.png') # Or whatever screenshot you want to analyze

pprint(ar.game_data)
```

Which returns the output

```python3
{'num_players': 3,
 'placement': '1',
 'players': {'player0': {'damage': '2250',
                         'kills': '12',
                         'name': 'lfegirltocapture',
                         'respawns': '0',
                         'revives': '1',
                         'survival_time': '16:55'},
             'player1': {'damage': '2799',
                         'kills': '10',
                         'name': 'OGRealStimJimmy',
                         'respawns': '1',
                         'revives': '1',
                         'survival_time': '16:55'},
             'player2': {'damage': '982',
                         'kills': '4',
                         'name': 'syuraponda 4',
                         'respawns': '0',
                         'revives': '0',
                         'survival_time': '1855'}},
 'squad_kills': '26'}
>>> 
```
 We can see the screenshot isn't read *perfectly* but it is *very* close, too which we can thank Google's [Tesseract OCR](https://opensource.google/projects/tesseract).
