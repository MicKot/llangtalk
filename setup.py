from setuptools import setup, find_packages

setup(
    name="llangtalk",
    version="0.1.0",
    description="Language learning assistant with speech capabilities",
    packages=find_packages(),
    install_requires=[
        "coqui-tts>=0.11.0",
        "numpy",
        "sounddevice",
        "scipy",
    ],
    extras_require={
        "dev": [
            "black",
            "flake8",
            "pytest",
            "pytest-cov",
        ],
    },
    python_requires=">=3.8",
)
