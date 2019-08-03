import setuptools

with open("README.md", "r") as fh:
	long_description = fh.read()
	
setuptools.setup(
	name="pyLpov",
	version="0.1.0",
	author="Okba Bekhelifi",
	author_email="okba.bekhelifi@univ-usto.dz",
	description="Python Laresi Processing for OpenVibe",
	long_description = long_description,
	long_description_content_type="text/markdown",	
	packages=setuptools.find_packages(),
	classifiers=[
		"Programming Language :: Python :: 2.7",
		"Licence :: OSI Approved :: Apache License 2.0",
		"Operating System :: OS Indepentdent",
	]
)