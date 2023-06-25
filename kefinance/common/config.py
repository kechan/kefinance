from __future__ import print_function

from pathlib import Path

bOnColab = Path('/content').exists()

if bOnColab:
  home = Path('/content/drive/MyDrive/kefinance')  