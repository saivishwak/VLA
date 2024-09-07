from setuptools import setup, find_packages

setup(
    name='VLA',
    version='0.0.1',
    packages=find_packages(),
    include_package_data=True,
    # license=open('LICENSE').read(),
    zip_safe=False,
    description="VLA",
    author='saivishwak40@gmail.com',
    author_email='saivishwak40@gmail.com',
    url='',
    install_requires=[line for line in open(
        'requirements.txt').readlines() if "@" not in line],
    keywords=['VLA', 'Vision Language Action', 'Robotics', 'Manipulation'],
)
