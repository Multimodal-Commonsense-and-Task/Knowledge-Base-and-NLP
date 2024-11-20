# Copyright 2021 The HuggingFace Team and the AdapterHub Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Simple check list from AllenNLP repo: https://github.com/allenai/allennlp/blob/main/setup.py

To create the package for pypi.

1. Run `make pre-release` (or `make pre-patch` for a patch release) then run `make fix-copies` to fix the index of the
   documentation.

   If releasing on a special branch, copy the updated README.md on the main branch for your the commit you will make
   for the post-release and run `make fix-copies` on the main branch as well.

2. Run Tests for Amazon Sagemaker. The documentation is located in `./tests/sagemaker/README.md`, otherwise @philschmid.

3. Unpin specific versions from setup.py that use a git install.

4. Checkout the release branch (v<RELEASE>-release, for example v4.19-release), and commit these changes with the
   message: "Release: <VERSION>" and push.

5. Wait for the tests on main to be completed and be green (otherwise revert and fix bugs)

6. Add a tag in git to mark the release: "git tag v<VERSION> -m 'Adds tag v<VERSION> for pypi' "
   Push the tag to git: git push --tags origin v<RELEASE>-release

7. Build both the sources and the wheel. Do not change anything in setup.py between
   creating the wheel and the source distribution (obviously).

   For the wheel, run: "python setup.py bdist_wheel" in the top level directory.
   (this will build a wheel for the python version you use to build it).

   For the sources, run: "python setup.py sdist"
   You should now have a /dist directory with both .whl and .tar.gz source versions.

8. Check that everything looks correct by uploading the package to the pypi test server:

   twine upload dist/* -r pypitest
   (pypi suggest using twine as other methods upload files via plaintext.)
   You may have to specify the repository url, use the following command then:
   twine upload dist/* -r pypitest --repository-url=https://test.pypi.org/legacy/

   Check that you can install it in a virtualenv by running:
   pip install -i https://testpypi.python.org/pypi transformers

   Check you can run the following commands:
   python -c "from transformers import pipeline; classifier = pipeline('text-classification'); print(classifier('What a nice release'))"
   python -c "from transformers import *"

9. Upload the final version to actual pypi:
   twine upload dist/* -r pypi

10. Copy the release notes from RELEASE.md to the tag in github once everything is looking hunky-dory.

11. Run `make post-release` then run `make fix-copies`. If you were on a branch for the release,
    you need to go back to main before executing this.
"""

import os
import re
import shutil
from distutils.core import Command
from pathlib import Path

from setuptools import find_packages, setup


# Remove stale transformers.egg-info directory to avoid https://github.com/pypa/pip/issues/5466
stale_egg_info = Path(__file__).parent / "transformers.egg-info"
if stale_egg_info.exists():
    print(
        (
            "Warning: {} exists.\n\n"
            "If you recently updated transformers to 3.0 or later, this is expected,\n"
            "but it may prevent transformers from installing in editable mode.\n\n"
            "This directory is automatically generated by Python's packaging tools.\n"
            "I will remove it now.\n\n"
            "See https://github.com/pypa/pip/issues/5466 for details.\n"
        ).format(stale_egg_info)
    )
    shutil.rmtree(stale_egg_info)


# IMPORTANT:
# 1. all dependencies should be listed here with their version requirements if any
# 2. once modified, run: `make deps_table_update` to update src/transformers/dependency_versions_table.py
_deps = [
    "Pillow",
    "accelerate>=0.10.0",
    "black==22.3",
    "codecarbon==1.2.0",
    "cookiecutter==1.7.3",
    "dataclasses",
    "datasets",
    "deepspeed>=0.6.5",
    "dill<0.3.5",
    "docutils==0.16.0",
    "fairscale>0.3",
    "faiss-cpu",
    "fastapi",
    "filelock",
    "flake8>=3.8.3",
    "flax>=0.4.1",
    "ftfy",
    "fugashi>=1.0",
    "GitPython<3.1.19",
    "hf-doc-builder>=0.3.0",
    "huggingface-hub>=0.1.0,<1.0",
    "importlib_metadata",
    "ipadic>=1.0.0,<2.0",
    "isort>=5.5.4",
    "jax>=0.2.8,!=0.3.2,<=0.3.6",
    "jaxlib>=0.1.65,<=0.3.6",
    "jieba",
    "nltk",
    "numpy>=1.17",
    "onnxconverter-common",
    "onnxruntime-tools>=1.4.2",
    "onnxruntime>=1.4.0",
    "optuna",
    "optax>=0.0.8",
    "packaging>=20.0",
    "parameterized",
    "phonemizer",
    "protobuf<=3.20.1",
    "psutil",
    "pyyaml>=5.1",
    "pydantic",
    "pytest",
    "pytest-subtests",
    "pytest-timeout",
    "pytest-xdist",
    "python>=3.7.0",
    "ray[tune]",
    "myst-parser",
    "regex!=2019.12.17",
    "requests",
    "resampy<0.3.1",
    "rjieba",
    "rouge-score",
    "sacrebleu>=1.4.12,<2.0.0",
    "sacremoses",
    "sagemaker>=2.31.0",
    "scikit-learn",
    "sentencepiece>=0.1.91,!=0.1.92",
    "sigopt",
    "librosa",
    "sphinx-copybutton",
    "sphinx-markdown-tables",
    "sphinx-rtd-theme==0.4.3",  # sphinx-rtd-theme==0.5.0 introduced big changes in the style.
    "sphinx==3.2.1",
    "sphinxext-opengraph==0.4.1",
    "sphinx-intl",
    "sphinx-multiversion",
    "starlette",
    "tensorflow-cpu>=2.3",
    "tensorflow>=2.3",
    "tensorflow-text",
    "tf2onnx",
    "timeout-decorator",
    "timm",
    "tokenizers>=0.11.1,!=0.11.3,<0.13",
    "torch>=1.0,<1.12",
    "torchaudio",
    "pyctcdecode>=0.3.0",
    "tqdm>=4.27",
    "unidic>=1.0.2",
    "unidic_lite>=1.0.7",
    "uvicorn",
]


# this is a lookup table with items like:
#
# tokenizers: "tokenizers==0.9.4"
# packaging: "packaging"
#
# some of the values are versioned whereas others aren't.
deps = {b: a for a, b in (re.findall(r"^(([^!=<>~]+)(?:[!=<>~].*)?$)", x)[0] for x in _deps)}

# since we save this data in src/transformers/dependency_versions_table.py it can be easily accessed from
# anywhere. If you need to quickly access the data from this table in a shell, you can do so easily with:
#
# python -c 'import sys; from transformers.dependency_versions_table import deps; \
# print(" ".join([ deps[x] for x in sys.argv[1:]]))' tokenizers datasets
#
# Just pass the desired package names to that script as it's shown with 2 packages above.
#
# If transformers is not yet installed and the work is done from the cloned repo remember to add `PYTHONPATH=src` to the script above
#
# You can then feed this for example to `pip`:
#
# pip install -U $(python -c 'import sys; from transformers.dependency_versions_table import deps; \
# print(" ".join([ deps[x] for x in sys.argv[1:]]))' tokenizers datasets)
#


def deps_list(*pkgs):
    return [deps[pkg] for pkg in pkgs]


class DepsTableUpdateCommand(Command):
    """
    A custom distutils command that updates the dependency table.
    usage: python setup.py deps_table_update
    """

    description = "build runtime dependency table"
    user_options = [
        # format: (long option, short option, description).
        ("dep-table-update", None, "updates src/transformers/dependency_versions_table.py"),
    ]

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        entries = "\n".join([f'    "{k}": "{v}",' for k, v in deps.items()])
        content = [
            "# THIS FILE HAS BEEN AUTOGENERATED. To update:",
            "# 1. modify the `_deps` dict in setup.py",
            "# 2. run `make deps_table_update``",
            "deps = {",
            entries,
            "}",
            "",
        ]
        target = "src/transformers/dependency_versions_table.py"
        print(f"updating {target}")
        with open(target, "w", encoding="utf-8", newline="\n") as f:
            f.write("\n".join(content))


extras = {}

extras["ja"] = deps_list("fugashi", "ipadic", "unidic_lite", "unidic")
extras["sklearn"] = deps_list("scikit-learn")

extras["tf"] = deps_list("tensorflow", "onnxconverter-common", "tf2onnx", "tensorflow-text")
extras["tf-cpu"] = deps_list("tensorflow-cpu", "onnxconverter-common", "tf2onnx", "tensorflow-text")

extras["torch"] = deps_list("torch")
extras["accelerate"] = deps_list("accelerate")

if os.name == "nt":  # windows
    extras["retrieval"] = deps_list("datasets")  # faiss is not supported on windows
    extras["flax"] = []  # jax is not supported on windows
else:
    extras["retrieval"] = deps_list("faiss-cpu", "datasets")
    extras["flax"] = deps_list("jax", "jaxlib", "flax", "optax")

extras["tokenizers"] = deps_list("tokenizers")
extras["ftfy"] = deps_list("ftfy")
extras["onnxruntime"] = deps_list("onnxruntime", "onnxruntime-tools")
extras["onnx"] = deps_list("onnxconverter-common", "tf2onnx") + extras["onnxruntime"]
extras["modelcreation"] = deps_list("cookiecutter")

extras["sagemaker"] = deps_list("sagemaker")
extras["deepspeed"] = deps_list("deepspeed") + extras["accelerate"]
extras["fairscale"] = deps_list("fairscale")
extras["optuna"] = deps_list("optuna")
extras["ray"] = deps_list("ray[tune]")
extras["sigopt"] = deps_list("sigopt")

extras["integrations"] = extras["optuna"] + extras["ray"] + extras["sigopt"]

extras["serving"] = deps_list("pydantic", "uvicorn", "fastapi", "starlette")
extras["audio"] = deps_list("librosa", "pyctcdecode", "phonemizer", "resampy")  # resampy can be removed once unpinned.
# `pip install ".[speech]"` is deprecated and `pip install ".[torch-speech]"` should be used instead
extras["speech"] = deps_list("torchaudio") + extras["audio"]
extras["torch-speech"] = deps_list("torchaudio") + extras["audio"]
extras["tf-speech"] = extras["audio"]
extras["flax-speech"] = extras["audio"]
extras["vision"] = deps_list("Pillow")
extras["timm"] = deps_list("timm")
extras["codecarbon"] = deps_list("codecarbon")

extras["sentencepiece"] = deps_list("sentencepiece", "protobuf")
extras["testing"] = (
    deps_list(
        "pytest",
        "pytest-xdist",
        "timeout-decorator",
        "parameterized",
        "psutil",
        "datasets",
        "dill",
        "pytest-timeout",
        "black",
        "sacrebleu",
        "rouge-score",
        "nltk",
        "GitPython",
        "hf-doc-builder",
        "protobuf",  # Can be removed once we can unpin protobuf
        "sacremoses",
        "rjieba",
    )
    + extras["retrieval"]
    + extras["modelcreation"]
)

extras["deepspeed-testing"] = extras["deepspeed"] + extras["testing"] + extras["optuna"]

extras["quality"] = deps_list("black", "isort", "flake8", "GitPython", "hf-doc-builder")

extras["all"] = (
    extras["tf"]
    + extras["torch"]
    + extras["flax"]
    + extras["sentencepiece"]
    + extras["tokenizers"]
    + extras["torch-speech"]
    + extras["vision"]
    + extras["integrations"]
    + extras["timm"]
    + extras["codecarbon"]
    + extras["accelerate"]
)

extras["docs_specific"] = deps_list(
    "docutils",
    "myst-parser",
    "sphinx",
    "sphinx-markdown-tables",
    "sphinx-rtd-theme",
    "sphinx-copybutton",
    "sphinxext-opengraph",
    "sphinx-intl",
    "sphinx-multiversion",
)
# "docs" needs "all" to resolve all the references
extras["docs"] = extras["all"] + extras["docs_specific"]

extras["dev-torch"] = (
    extras["testing"]
    + extras["torch"]
    + extras["sentencepiece"]
    + extras["tokenizers"]
    + extras["torch-speech"]
    + extras["vision"]
    + extras["integrations"]
    + extras["timm"]
    + extras["codecarbon"]
    + extras["quality"]
    + extras["ja"]
    + extras["docs_specific"]
    + extras["sklearn"]
    + extras["modelcreation"]
    + extras["onnxruntime"]
)
extras["dev-tensorflow"] = (
    extras["testing"]
    + extras["tf"]
    + extras["sentencepiece"]
    + extras["tokenizers"]
    + extras["vision"]
    + extras["quality"]
    + extras["docs_specific"]
    + extras["sklearn"]
    + extras["modelcreation"]
    + extras["onnx"]
    + extras["tf-speech"]
)
extras["dev"] = (
    extras["all"]
    + extras["testing"]
    + extras["quality"]
    + extras["ja"]
    + extras["docs_specific"]
    + extras["sklearn"]
    + extras["modelcreation"]
)

extras["torchhub"] = deps_list(
    "filelock",
    "huggingface-hub",
    "importlib_metadata",
    "numpy",
    "packaging",
    "protobuf",
    "regex",
    "requests",
    "sentencepiece",
    "torch",
    "tokenizers",
    "tqdm",
)

# when modifying the following list, make sure to update src/transformers/dependency_versions_check.py
install_requires = [
    deps["importlib_metadata"] + ";python_version<'3.8'",  # importlib_metadata for Python versions that don't have it
    deps["filelock"],  # filesystem locks, e.g., to prevent parallel downloads
    deps["huggingface-hub"],
    deps["numpy"],
    deps["packaging"],  # utilities from PyPA to e.g., compare versions
    deps["pyyaml"],  # used for the model cards metadata
    deps["regex"],  # for OpenAI GPT
    deps["requests"],  # for downloading models over HTTPS
    deps["tokenizers"],
    deps["tqdm"],  # progress bars in model download and training scripts
]

setup(
    name="adapter-transformers",
    version="3.1.0",
    author="Jonas Pfeiffer, Andreas Rücklé, Clifton Poth, Hannah Sterz, based on work by the HuggingFace team and community",
    author_email="pfeiffer@ukp.tu-darmstadt.de",
    description="A friendly fork of HuggingFace's Transformers, adding Adapters to PyTorch language models",
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    keywords="NLP deep learning transformer pytorch BERT adapters",
    license="Apache",
    url="https://github.com/adapter-hub/adapter-transformers",
    package_dir={"": "src"},
    packages=find_packages("src"),
    package_data={"transformers": ["py.typed"]},
    zip_safe=False,
    extras_require=extras,
    entry_points={"console_scripts": ["transformers-cli=transformers.commands.transformers_cli:main"]},
    python_requires=">=3.7.0",
    install_requires=install_requires,
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    cmdclass={"deps_table_update": DepsTableUpdateCommand},
)