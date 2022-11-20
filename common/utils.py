import pickle
from pathlib import Path
import IPython.display as display

Path.ls = lambda x: list(x.iterdir())
Path.lf = lambda pth, pat='*': list(pth.glob(pat))
Path.rlf = lambda pth, pat='*': list(pth.rglob(pat))

def load_from_pickle(filename):
  try:
    with open(str(filename), 'rb') as f:
      obj = pickle.load(f)
  except Exception as ex:
    print(ex)
    return None

  return obj

def save_to_pickle(obj, filename):
  try:
    with open(str(filename), 'wb') as f:
      pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
  except Exception as ex:
    print(ex)


def say_done():
  return display.Audio(url="https://ssl.gstatic.com/dictionary/static/pronunciation/2019-10-21/audio/do/done_en_us_1.mp3", autoplay=True)
