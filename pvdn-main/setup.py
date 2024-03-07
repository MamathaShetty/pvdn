from setuptools import setup, find_packages, Extension
import os
import warnings
warnings.filterwarnings('ignore')

lib_dir = os.path.dirname(os.path.realpath(__file__))
requirements_path = lib_dir + "/requirements.txt"
install_requires = []
if os.path.isfile(requirements_path):
    with open(requirements_path) as f:
        install_requires = f.read().splitlines()

lib = Extension(name="pvdn.detection.utils.image_operations",
                sources=[
                    "pvdn/detection/utils/image_operations.cpp",
                    "pvdn/detection/utils/HeadLampObject.cpp"
                ])
print("Meena")

setup(
    name="PVDN-MAIN",
    version="0.2.0",
    packages=find_packages(),
    url="https://github.com/larsOhne/pvdn",
    license="Creative Commons Legal Code ",
    author=
    "Lars Ohnemus, Lukas Ewecker, Ebubekir Asan, Stefan Roos, Simon Isele, Jakob Ketterer, Leopold Müller, and Sascha Saralajew",
    author_email="",
    description="Tools for working with the PVDN dataset.",
    install_requires=install_requires,
    ext_modules=[lib],
)
print("end")