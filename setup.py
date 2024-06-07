from setuptools import setup, find_packages

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name='ncmcm',
    version='1.1.0',
    author='Akshey Kumar, Hofer Michael, Paul Eder, Emilija Mazuraite, Nenad Subat',
    author_email='akshey.kumar@univie.ac.at',
    description='A toolbox to visualize neuronal imaging data and apply the NC-MCM framework to it',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/NC-MCM/NC-MCM',
    packages=find_packages(),
    include_package_data=True,
    package_data={},
    classifiers=[
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='==3.10.12',
    install_requires=[
        'absl-py==2.1.0',
        'aiofiles==23.2.1',
        'altair==5.3.0',
        'annotated-types==0.7.0',
        'anyio==4.4.0',
        'asttokens==2.4.1',
        'astunparse==1.6.3',
        'attrs==23.2.0',
        'certifi==2024.2.2',
        'charset-normalizer==3.3.2',
        'click==8.1.7',
        'contourpy==1.2.1',
        'cycler==0.12.1',
        'decorator==5.1.1',
        'dnspython==2.6.1',
        'email_validator==2.1.1',
        'exceptiongroup==1.2.1',
        'executing==2.0.1',
        'fastapi==0.111.0',
        'fastapi-cli==0.0.4',
        'ffmpy==0.3.2',
        'filelock==3.14.0',
        'flatbuffers==24.3.25',
        'fonttools==4.52.4',
        'fsspec==2024.6.0',
        'gast==0.5.4',
        'google-pasta==0.2.0',
        'gradio==4.35.0',
        'gradio_client==1.0.1',
        'grpcio==1.64.0',
        'h11==0.14.0',
        'h5py==3.11.0',
        'httpcore==1.0.5',
        'httptools==0.6.1',
        'httpx==0.27.0',
        'huggingface-hub==0.23.3',
        'idna==3.7',
        'importlib_metadata==7.1.0',
        'importlib_resources==6.4.0',
        'iniconfig==2.0.0',
        'ipython==8.18.1',
        'jedi==0.19.1',
        'Jinja2==3.1.4',
        'joblib==1.4.2',
        'jsonpickle==3.0.4',
        'jsonschema==4.22.0',
        'jsonschema-specifications==2023.12.1',
        'keras==3.3.3',
        'kiwisolver==1.4.5',
        'libclang==18.1.1',
        'Markdown==3.6',
        'markdown-it-py==3.0.0',
        'MarkupSafe==2.0.1',
        'mat73==0.63',
        'matplotlib==3.9.0',
        'matplotlib-inline==0.1.7',
        'mdurl==0.1.2',
        'ml-dtypes==0.3.2',
        'namex==0.0.8',
        'networkx==3.2.1',
        'numpy==1.26.4',
        'opt-einsum==3.3.0',
        'optree==0.11.0',
        'orjson==3.10.3',
        'packaging==24.0',
        'pandas==2.2.2',
        'parso==0.8.4',
        'patsy==0.5.6',
        'pexpect==4.9.0',
        'pillow==10.3.0',
        'pluggy==1.5.0',
        'prompt_toolkit==3.0.45',
        'protobuf==4.25.3',
        'ptyprocess==0.7.0',
        'pure-eval==0.2.2',
        'pydantic==2.7.3',
        'pydantic_core==2.18.4',
        'pydub==0.25.1',
        'Pygments==2.18.0',
        'pyparsing==3.1.2',
        'pytest==8.2.1',
        'python-dateutil==2.9.0.post0',
        'python-dotenv==1.0.1',
        'python-multipart==0.0.9',
        'pytz==2024.1',
        'pyvis==0.3.2',
        'PyYAML==6.0.1',
        'referencing==0.35.1',
        'requests==2.32.3',
        'rich==13.7.1',
        'rpds-py==0.18.1',
        'ruff==0.4.8',
        'scikit-learn==1.5.0',
        'scipy==1.13.1',
        'semantic-version==2.10.0',
        'shellingham==1.5.4',
        'six==1.16.0',
        'sniffio==1.3.1',
        'stack-data==0.6.3',
        'starlette==0.37.2',
        'statsmodels==0.14.2',
        'tensorboard==2.16.2',
        'tensorboard-data-server==0.7.2',
        'tensorflow==2.16.1',
        'tensorflow-io-gcs-filesystem==0.37.0',
        'termcolor==2.4.0',
        'threadpoolctl==3.5.0',
        'tomli==2.0.1',
        'tomlkit==0.12.0',
        'toolz==0.12.1',
        'tqdm==4.66.4',
        'traitlets==5.14.3',
        'typer==0.12.3',
        'typing_extensions==4.12.2',
        'tzdata==2024.1',
        'ujson==5.10.0',
        'urllib3==2.2.1',
        'uvicorn==0.30.1',
        'uvloop==0.19.0',
        'watchfiles==0.22.0',
        'wcwidth==0.2.13',
        'websockets==11.0.3',
        'Werkzeug==3.0.3',
        'wrapt==1.16.0',
        'zipp==3.19.0',
        '-e git+https://github.com/NC-MCM/NC-MCM.git@df68f06f81f98e19229b5fa2f800764cb93ee8fc#egg=ncmcm'
    ],
    extras_require={
        'dev': [
            'pytest',
            'pytest-cov',
        ],
    },
)
