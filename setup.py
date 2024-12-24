from setuptools import setup, find_packages

setup(
    name="coffee-bean-classifier",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "streamlit>=1.29.0",
        "torch>=2.1.2",
        "torchvision>=0.16.2",
        "Pillow>=10.1.0",
        "supabase>=2.3.0",
        "python-dotenv>=1.0.0",
        "fastapi>=0.109.0",
        "uvicorn>=0.25.0",
        "python-multipart>=0.0.6",
        "numpy>=1.26.2",
        "scikit-learn>=1.3.2",
    ],
    python_requires=">=3.12",
) 