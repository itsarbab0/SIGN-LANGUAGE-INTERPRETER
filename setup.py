from setuptools import setup, find_packages

setup(
    name="sign_language_interpreter",
    version="1.0.0",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "Flask",
        "Flask-Cors",
        "numpy",
        "opencv-python",
        "tensorflow",
        "mediapipe",
        "gunicorn",
        "Pillow",
        "scikit-learn",
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Education",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
) 