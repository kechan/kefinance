from setuptools import find_packages, setup

with open("README.md", "r", encoding="utf-8") as fh:
  long_description = fh.read()

setup(
  name="kefinance",
  version="0.0.1",
  author="R H",
  author_email="ruffian-hamster0n@icloud.com",
  description="kefinance",
  long_description=long_description,
  long_description_content_type="text/markdown",
  url="https://github.com/kechan/kefinance",
  packages=find_packages(),
)