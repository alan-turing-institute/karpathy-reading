from ascii_magic import AsciiArt
import sys

# to_terminal(self
#   columns: int = 120,
#   width_ratio: float = 2.2,
#   char: Optional[str] = None,
#   enhance_image: bool = False,
#   monochrome: bool = False,
#   back: Optional[ascii_magic.constants.Back] = None,
#   front: Optional[ascii_magic.constants.Front] = None,
#   debug: bool = False
#)

art = AsciiArt.from_image(sys.argv[1])
art.to_terminal(columns=80)
