pip install wheel twine setuptools
rm -rf dist
python setup.py bdist_wheel
twine upload --repository pypi dist/* --username $PYPI_USER --password $PYPI_PASS
