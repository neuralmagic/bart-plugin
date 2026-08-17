"""Setup script for vLLM BART model plugin."""

from setuptools import find_packages, setup

setup(
    name="vllm-bart-plugin",
    version="0.6.0",
    description="BART model plugin for vLLM",
    author="Nicolò Lucchesi",
    author_email="nick.lucche@redhat.com",
    packages=find_packages(),
    package_data={"vllm_bart_plugin": ["py.typed"]},
    python_requires=">=3.10",
    install_requires=[
        "vllm>=0.24.0",
        "torch>=2.9.0",
        "transformers>=4.56.0",
    ],
    entry_points={
        "vllm.general_plugins": [
            "bart=vllm_bart_plugin:register_bart_model",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
)
